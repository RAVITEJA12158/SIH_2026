"""
GeoSync AI - XGBoost Confidence Scoring (doc section 21)
------------------------------------------------------------
Trains a binary classifier that estimates match reliability for each
(drone-extracted footprint <-> cadastral parcel) candidate, fused with
supporting evidence from municipal/utility/revenue sources.

Training labels come from many independently generated synthetic
scenes (ground truth is known there, standing in for "historical
officially-verified match decisions" a real deployment would train
on -- see doc section 21/41 Stage 5). The model is then applied to
the actual end-to-end pipeline output for the deployment scene, which
was produced by the real SegFormer -> Specialized -> SAM pipeline, so
the reported numbers reflect genuine upstream extraction noise, not
just clean synthetic geometry.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from shapely.affinity import translate
import geopandas as gpd
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))

from synthetic_data import generate_building_layout, build_government_layers, _px_to_geo
from fusion import build_fused_candidate_table

FEATURE_COLS = [
    "iou", "centroid_dist_m", "area_ratio", "hausdorff_m", "attribute_score",
    "extraction_confidence", "municipal_present", "municipal_iou", "municipal_attr_score",
    "utility_present", "utility_iou", "revenue_present", "revenue_attr_score",
    "cross_source_support_count",
]


def _simulate_extracted_layer(truth_polys, rng):
    """Stand-in for a full SegFormer/Specialized/SAM run at the feature-
    vector-training-corpus stage: injects the SAME categories of error a
    real adaptive extractor produces (positional jitter, occasional missed
    features, occasional spurious/split footprints, variable confidence)
    without re-running CNN training for every synthetic scene (the real
    deployment scene, evaluated later, DOES use the true trained models)."""
    rows = []
    for i, poly in enumerate(truth_polys):
        if rng.random() < 0.08:
            continue  # missed detection (heavy occlusion, etc.)
        geo = _px_to_geo(poly)
        jitter_x, jitter_y = rng.normal(0, 0.4), rng.normal(0, 0.4)
        geo = translate(geo, xoff=jitter_x, yoff=jitter_y)
        if rng.random() < 0.10:
            geo = geo.buffer(-0.3)  # under-segmentation
        conf = float(np.clip(rng.normal(0.82, 0.12), 0.2, 0.99))
        rows.append({"geometry": geo, "extraction_confidence": conf,
                      "extraction_method": rng.choice(["segformer", "specialized", "sam_fallback"], p=[0.75, 0.18, 0.07]),
                      "source": "drone_ai_extraction", "_truth_id": i})
    # a few spurious false-positive footprints with no real counterpart
    for _ in range(rng.integers(0, 3)):
        fake = _px_to_geo(truth_polys[0]).centroid.buffer(rng.uniform(1.0, 2.0))
        rows.append({"geometry": fake, "extraction_confidence": float(rng.uniform(0.3, 0.6)),
                      "extraction_method": "sam_fallback", "source": "drone_ai_extraction", "_truth_id": -1})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32644")


def build_training_corpus(n_scenes=10, seed=123):
    rng = np.random.default_rng(seed)
    all_rows = []
    for s in range(n_scenes):
        polys = generate_building_layout(n=int(rng.integers(14, 26)), size=512)
        gdfs, owners, landuse = build_government_layers(polys)
        extracted = _simulate_extracted_layer(polys, rng)
        if len(extracted) == 0 or len(gdfs["cadastral"]) == 0:
            continue
        table = build_fused_candidate_table(extracted, gdfs["cadastral"], gdfs["municipal"],
                                             gdfs["utility"], gdfs["revenue"])
        if len(table) == 0:
            continue
        # ground-truth label: does the extracted feature's true building id
        # equal the cadastral record's true building id?
        cad_truth = gdfs["cadastral"]["_truth_id"].values
        ext_truth = extracted["_truth_id"].values
        labels = [int(ext_truth[int(r.extracted_idx)] == cad_truth[int(r.gov_idx)]
                       and ext_truth[int(r.extracted_idx)] != -1)
                  for r in table.itertuples()]
        table["label"] = labels
        table["scene"] = s
        all_rows.append(table)
    return pd.concat(all_rows, ignore_index=True)


def train_confidence_model(corpus: pd.DataFrame, test_size=0.25, seed=0):
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(corpus, groups=corpus["scene"]))
    train, test = corpus.iloc[train_idx], corpus.iloc[test_idx]

    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]

    model = xgb.XGBClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", random_state=seed,
        scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()),
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n_train": len(train), "n_test": len(test),
        "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan"),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }
    return model, metrics


def score_candidates(model, table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["confidence"] = model.predict_proba(table[FEATURE_COLS])[:, 1]
    return table


if __name__ == "__main__":
    print("Building multi-scene synthetic training corpus for XGBoost confidence model...")
    corpus = build_training_corpus(n_scenes=10)
    print(f"  Corpus size: {len(corpus)} candidate pairs across {corpus['scene'].nunique()} scenes "
          f"({corpus['label'].sum()} positive / {len(corpus)} total)")

    model, metrics = train_confidence_model(corpus)
    print("Held-out evaluation (grouped by scene, no leakage):")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.3f}" if isinstance(v, float) else f"  {k:10s}: {v}")

    from settings import MODELS_DIR
    model.save_model(MODELS_DIR / "xgb_confidence.json")
    print("\nSaved -> models/xgb_confidence.json")
