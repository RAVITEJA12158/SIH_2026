"""
GeoSync AI - Confidence-Based Decision Framework (doc sections 22-24)
-------------------------------------------------------------------------
Routes every scored candidate match into:
  HIGH       -> high-confidence proposal eligible for an authorised workflow
  MEDIUM     -> flagged prototype proposal
  LOW        -> review queue (`manual_review` label)

These labels model a routing policy for the synthetic demonstration. They do
not authorize an official record change; real deployments require authorized
human review and calibrated, governed thresholds.

Also performs basic conflict detection (section 23):
  - Extracted footprints with no acceptable cadastral candidate at all
    ("feature exists in drone imagery but not in records")
  - Cadastral parcels with no extracted-footprint candidate
    ("feature exists in records but not on the ground / imagery")
  - Multiple cadastral parcels competing for the same extracted
    footprint above the LOW threshold ("duplicate/competing claim")
"""
import pandas as pd
import numpy as np

HIGH_THRESHOLD = 0.75
LOW_THRESHOLD = 0.40


def classify_confidence(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= LOW_THRESHOLD:
        return "medium"
    return "low"


def route_decisions(scored_table: pd.DataFrame) -> pd.DataFrame:
    df = scored_table.copy()
    df["confidence_tier"] = df["confidence"].apply(classify_confidence)
    df["decision"] = df["confidence_tier"].map({
        "high": "proposed_auto_eligible",
        "medium": "proposed_flagged",
        "low": "manual_review",
    })
    return df


def resolve_best_match_per_feature(routed: pd.DataFrame) -> pd.DataFrame:
    """For each extracted footprint, keep only its single best-scoring
    cadastral candidate as the 'proposed' match (others become
    lower-ranked alternates, useful context for a human reviewer)."""
    routed = routed.sort_values("confidence", ascending=False)
    best = routed.groupby("extracted_idx", as_index=False).first()
    return best


def detect_conflicts(extracted_gdf, cadastral_gdf, routed: pd.DataFrame):
    """Returns a dict of conflict categories (doc section 23)."""
    matched_extracted = set(routed.loc[routed["decision"] != "manual_review", "extracted_idx"])
    matched_cadastral = set(routed.loc[routed["decision"] != "manual_review", "gov_idx"])

    unmatched_extracted = [i for i in range(len(extracted_gdf)) if i not in set(routed["extracted_idx"])]
    unmatched_cadastral = [j for j in range(len(cadastral_gdf)) if j not in set(routed["gov_idx"])]

    # competing claims: >1 extracted footprint both scoring 'medium'+ for
    # the SAME cadastral parcel
    competing = (
        routed[routed["confidence_tier"].isin(["high", "medium"])]
        .groupby("gov_idx")["extracted_idx"].nunique()
    )
    competing_parcels = competing[competing > 1].index.tolist()

    return {
        "no_record_match": unmatched_extracted,          # imagery feature, no gov record nearby at all
        "no_imagery_match": unmatched_cadastral,          # gov record, nothing detected on the ground
        "competing_claims_gov_idx": competing_parcels,    # multiple footprints claiming one parcel
    }


def build_harmonized_output(extracted_gdf, cadastral_gdf, routed_best: pd.DataFrame):
    """Final prototype annotated spatial table -- one row per extracted
    footprint with its proposed status, matched cadastral attributes
    (if any), and traceability fields. It is not an official validated
    record or a tamper-evident audit log."""
    rows = []
    matched_idx = set(routed_best["extracted_idx"])
    for i in range(len(extracted_gdf)):
        erow = extracted_gdf.iloc[i]
        if i in matched_idx:
            m = routed_best[routed_best["extracted_idx"] == i].iloc[0]
            crow = cadastral_gdf.iloc[int(m["gov_idx"])]
            rows.append({
                "geometry": erow.geometry,
                "status": m["decision"],
                "confidence": round(float(m["confidence"]), 3),
                "confidence_tier": m["confidence_tier"],
                "matched_survey_no": crow.get("survey_no"),
                "matched_owner_name": crow.get("owner_name"),
                "matched_land_use": crow.get("land_use"),
                "cross_source_support_count": int(m["cross_source_support_count"]),
                "extraction_method": erow.get("extraction_method"),
                "extraction_confidence": erow.get("extraction_confidence"),
            })
        else:
            rows.append({
                "geometry": erow.geometry,
                "status": "unmatched_conflict",
                "confidence": 0.0,
                "confidence_tier": "low",
                "matched_survey_no": None,
                "matched_owner_name": None,
                "matched_land_use": None,
                "cross_source_support_count": 0,
                "extraction_method": erow.get("extraction_method"),
                "extraction_confidence": erow.get("extraction_confidence"),
            })
    import geopandas as gpd
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=extracted_gdf.crs)


if __name__ == "__main__":
    import geopandas as gpd
    import xgboost as xgb
    from fusion import build_fused_candidate_table
    from confidence_model import score_candidates
    from settings import DATA_DIR, MODELS_DIR, OUTPUTS_DIR

    extracted = gpd.read_file(DATA_DIR / "extracted_features.geojson")
    cadastral = gpd.read_file(DATA_DIR / "cadastral.geojson")
    municipal = gpd.read_file(DATA_DIR / "municipal.geojson")
    utility = gpd.read_file(DATA_DIR / "utility.geojson")
    revenue = gpd.read_file(DATA_DIR / "revenue.geojson")

    table = build_fused_candidate_table(extracted, cadastral, municipal, utility, revenue)
    model = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / "xgb_confidence.json")
    scored = score_candidates(model, table)
    routed = route_decisions(scored)
    best = resolve_best_match_per_feature(routed)

    print(best["decision"].value_counts())
    conflicts = detect_conflicts(extracted, cadastral, best)
    print("\nConflicts:")
    for k, v in conflicts.items():
        print(f"  {k}: {v}")

    harmonized = build_harmonized_output(extracted, cadastral, best)
    harmonized.to_file(OUTPUTS_DIR / "harmonized_output.geojson", driver="GeoJSON")
    print(f"\nSaved harmonized output -> outputs/harmonized_output.geojson ({len(harmonized)} features)")
