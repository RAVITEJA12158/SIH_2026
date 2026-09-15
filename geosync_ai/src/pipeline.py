"""
GeoSync AI - Full Pipeline Runner
------------------------------------
Multi-Source Ingestion -> Clean & Standardize -> AI-Based Feature
Extraction (SegFormer -> Specialized -> SAM stand-in) -> Match & Align Records
-> Score Synthetic Candidates (XGBoost) -> Prototype Proposal / Review
Routing -> Annotated Demonstration Output

Mirrors the flow chart on the project's technical-approach slide.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from settings import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_project_directories

DATA = DATA_DIR
MODELS = MODELS_DIR
OUT = OUTPUTS_DIR


def step(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def main():
    ensure_project_directories()
    t0 = time.time()

    step("STAGE 1-2 | MULTI-SOURCE INGESTION + CLEAN & STANDARDIZE")
    import synthetic_data
    synthetic_data.generate_all(out_dir=DATA)
    import data_quality as dq
    raw_layers = {
        name: gpd.read_file(DATA / f"{name}.geojson")
        for name in ("cadastral", "municipal", "utility", "revenue")
    }
    standardised_layers, quality_report = dq.profile_and_standardize_layers(
        raw_layers, OUT / "data_quality_report.json"
    )
    dq.require_quality_gate(quality_report)
    print(f"Quality gate: {quality_report['overall_quality_gate']} ({len(quality_report['layers'])} vector layers profiled)")

    step("STAGE 3 | ADAPTIVE AI-BASED FEATURE EXTRACTION (SegFormer -> Specialized -> SAM)")
    import segformer_model as sf
    import specialized_model as spm
    import feature_extraction as fe

    deploy_img = np.load(DATA / "ortho_image.npy")
    deploy_mask = np.load(DATA / "ortho_mask_truth.npy")

    print("Training primary model (SegFormer, from-scratch architecture, no pretrained download)...")
    segformer = sf.train_segformer(deploy_img, deploy_mask, epochs=6)
    torch.save(segformer.state_dict(), MODELS / "segformer.pt")

    prob0, pred0 = sf.predict_full_scene(segformer, deploy_img)
    iou_primary_only = (np.logical_and(pred0, deploy_mask).sum()) / max(1, np.logical_or(pred0, deploy_mask).sum())
    print(f"SegFormer-only IoU vs. synthetic ground truth: {iou_primary_only:.3f}")

    print("\nTraining specialized fallback model (hard/occluded tiles)...")
    specialized = spm.train_specialized_model(epochs=8)
    torch.save(specialized.state_dict(), MODELS / "specialized_model.pt")

    print("\nRunning adaptive escalation (SegFormer -> Specialized -> SAM-fallback)...")
    final_prob, final_pred, method_map = fe.run_adaptive_extraction(
        segformer, specialized, deploy_img, sf.predict_full_scene)
    iou_final = (np.logical_and(final_pred, deploy_mask).sum()) / max(1, np.logical_or(final_pred, deploy_mask).sum())
    print(f"Final adaptive-pipeline IoU vs. synthetic ground truth: {iou_final:.3f} "
          f"(vs {iou_primary_only:.3f} SegFormer-alone)")

    extracted = fe.mask_to_polygons(final_pred, final_prob, method_map, synthetic_data._px_to_geo)
    extracted.to_file(DATA / "extracted_features.geojson", driver="GeoJSON")
    print(f"Extracted {len(extracted)} candidate footprints.")

    step("STAGE 4 | MATCH & ALIGN RECORDS (spatial candidate search + evidence fusion)")
    import fusion
    cadastral = standardised_layers["cadastral"]
    municipal = standardised_layers["municipal"]
    utility = standardised_layers["utility"]
    revenue = standardised_layers["revenue"]
    fused = fusion.build_fused_candidate_table(extracted, cadastral, municipal, utility, revenue)
    fused.to_csv(DATA / "fused_candidates.csv", index=False)
    print(f"Built {len(fused)} fused multi-source candidate pairs.")

    step("STAGE 5 | SCORE SYNTHETIC CANDIDATES (XGBoost confidence)")
    import confidence_model as cm
    print("Building multi-scene synthetic training corpus...")
    corpus = cm.build_training_corpus(n_scenes=10)
    conf_model, metrics = cm.train_confidence_model(corpus)
    print(f"Held-out confidence-model metrics: {metrics}")
    conf_model.save_model(MODELS / "xgb_confidence.json")

    scored = cm.score_candidates(conf_model, fused)
    scored.to_csv(OUT / "scored_candidates.csv", index=False)

    step("STAGE 6 | PROTOTYPE ROUTING + ANNOTATED HARMONIZED OUTPUT")
    import decision_engine as de
    routed = de.route_decisions(scored)
    best = de.resolve_best_match_per_feature(routed)
    conflicts = de.detect_conflicts(extracted, cadastral, best)
    harmonized = de.build_harmonized_output(extracted, cadastral, best)
    harmonized.to_file(OUT / "harmonized_output.geojson", driver="GeoJSON")

    decision_counts = best["decision"].value_counts().to_dict()
    tier_counts = best["confidence_tier"].value_counts().to_dict()

    summary = {
        "n_extracted_features": int(len(extracted)),
        "n_cadastral_records": int(len(cadastral)),
        "extraction_method_breakdown": extracted["extraction_method"].value_counts().to_dict(),
        "segformer_only_iou": round(float(iou_primary_only), 3),
        "adaptive_pipeline_iou": round(float(iou_final), 3),
        "confidence_model_metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()},
        "decision_counts": decision_counts,
        "confidence_tier_counts": tier_counts,
        "conflicts": {k: (v if not isinstance(v, list) or len(v) < 20 else f"{len(v)} items") for k, v in conflicts.items()},
        "data_quality_gate": quality_report["overall_quality_gate"],
        "prototype_notice": "All metrics and decisions are synthetic demonstration outputs, not official record validations.",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with (OUT / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    step("STAGE 7 | REVIEW PACKAGE + PROTOTYPE TRACEABILITY")
    import governance
    review_queue = governance.create_review_queue(best, conflicts, OUT / "review_queue.json")
    manifest = governance.build_artifact_manifest(
        [
            OUT / "run_summary.json", OUT / "data_quality_report.json", OUT / "scored_candidates.csv",
            OUT / "harmonized_output.geojson", OUT / "review_queue.json",
        ],
        OUT / "artifact_manifest.json",
    )
    governance.append_audit_event(
        OUT / "audit_log.jsonl", "pipeline_run", "prototype_pipeline",
        {"quality_gate": quality_report["overall_quality_gate"], "review_cases": len(review_queue["cases"]), "artifact_count": len(manifest["artifacts"])},
    )
    import review_dashboard
    review_dashboard.write_dashboard(OUT, summary, review_queue)
    print(f"Created review package with {len(review_queue['cases'])} cases and {len(manifest['artifacts'])} artifact hashes.")

    print("\n" + json.dumps(summary, indent=2, default=str))
    print(f"\nTotal pipeline runtime: {summary['runtime_seconds']}s")
    return summary, extracted, cadastral, municipal, utility, revenue, harmonized, best, conflicts, final_pred, method_map, deploy_img


if __name__ == "__main__":
    main()
