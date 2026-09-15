# GeoSync AI — Core Pipeline Prototype
**Team: The Sentinels | SIH26013 | Automated Integration and Intelligent Harmonization of Multi-Source Geospatial Data for Urban Land Record Management**

This is a working, end-to-end implementation of the core AI pipeline described in the project documentation — not a UI mockup. Every stage below runs real code on real (synthetic) data and produces real numbers.

## What's actually implemented and running

| Stage | Real implementation |
|---|---|
| **1. Multi-source ingestion** | Synthetic drone orthomosaic + 4 independent government layers (cadastral, municipal, utility, revenue) generated with realistic, *different* imperfections per source (positional shift, missing records, attribute-name drift, a phantom legacy record) |
| **2. Primary AI extraction** | **Real SegFormer architecture** (HuggingFace `transformers`, mit-b0-scale config, ~3.4M params) instantiated from scratch and trained on the synthetic orthomosaic. **0.925 IoU** stand-alone. |
| **3. Specialized fallback** | A compact dilated U-Net trained specifically on hard/occluded tiles, used when SegFormer's confidence is locally low |
| **4. SAM fallback (final tier)** | ⚠️ *Honest limitation:* this sandbox has no network route to download the real Segment Anything checkpoint, so a classical prompt-driven region-growing segmenter (Felzenszwalb superpixels + edge-aware confidence) plays SAM's architectural role. The call site (`sam_fallback.segment_region()`) is a drop-in replacement point for a real `SamPredictor` in production. |
| **5. Adaptive escalation logic** | Grid-based quality routing: SegFormer confidence **and** an independent image-based occlusion/shadow detector jointly decide whether a region escalates. In the last full run: 968/1024 cells resolved by SegFormer alone, 56 escalated to Specialized. Final pipeline IoU **0.933** (vs 0.925 SegFormer-alone) — the escalation genuinely improves results on the hard regions, at negligible extra cost since only ~5% of the scene needed it. |
| **6. Spatial candidate search** | STRtree-indexed candidate search (not O(n·m)) |
| **7. Geometric evidence** | IoU, Hausdorff distance, centroid distance, area-ratio similarity |
| **8. Progressive alignment** | RANSAC-style robust median-offset estimation implemented and applied when a systematic shift is detected (Stage 1–2 of the doc's 4-stage alignment; ICP/TPS are stubbed extension points for non-rigid legacy-survey distortion) |
| **9. Attribute evidence** | RapidFuzz fuzzy matching on owner-name fields, tolerant of the spelling drift injected into the synthetic municipal records |
| **10. Cross-source fusion** | For every drone-extracted↔cadastral candidate, the pipeline checks whether municipal, utility, and revenue records *independently* corroborate the same location/owner — exactly the "fuses spatial, geometric, and attribute evidence across cadastral, municipal, utility, revenue, and drone data" claim on the poster |
| **11. XGBoost confidence scoring** | Trained on a 10-scene synthetic corpus (405 candidate pairs), evaluated **held-out by scene** (no leakage): **Precision 0.977, Recall 1.0, F1 0.989, AUC 1.0** |
| **12. Decision engine** | High/Medium/Low confidence routing → `auto_validate` / `auto_validate_flagged` / `manual_review`, plus conflict detection (unmatched imagery features, unmatched records, competing claims) |
| **13. Harmonized output** | Final `GeoJSON` with per-feature status, matched cadastral attributes, confidence, and full extraction/matching traceability |

## Results from the latest full run

- 22 buildings in the synthetic scene → **22 footprints extracted** (18 via SegFormer, 4 via the specialized escalation tier)
- 21 of 21 cadastral records successfully matched
- **21 auto-validated, 1 routed to manual review** (a genuinely hard, heavily-occluded case)
- 0 unresolved conflicts in this run (2 in a prior run, correctly flagged as `no_record_match`)
- Total pipeline runtime: **~2 minutes** on CPU only, including training both models and the confidence classifier from scratch

See `outputs/overlay_comparison.png` for the before/after (raw multi-source drift → harmonized, confidence-colored result) and `outputs/extraction_scene.png` for which escalation tier resolved each building.

## Project structure

```
geosync_ai/
├── src/
│   ├── synthetic_data.py       # multi-source data + orthomosaic generator
│   ├── segformer_model.py      # primary extractor (real SegFormer arch)
│   ├── specialized_model.py    # 2nd-tier fallback (dilated mini U-Net)
│   ├── sam_fallback.py         # 3rd-tier fallback (SAM stand-in, documented)
│   ├── feature_extraction.py   # adaptive escalation orchestrator
│   ├── matching.py             # STRtree candidate search + geometric/attribute evidence
│   ├── fusion.py                # cross-source (cadastral/municipal/utility/revenue) evidence fusion
│   ├── confidence_model.py     # XGBoost training + inference
│   ├── decision_engine.py      # confidence routing + conflict detection
│   ├── visualize.py             # overlay + extraction-method figures
│   └── pipeline.py              # runs all of the above end-to-end
├── models/                      # trained weights (segformer.pt, specialized_model.pt, xgb_confidence.json)
├── data/                        # generated synthetic inputs + intermediate artifacts
└── outputs/                     # harmonized_output.geojson, run_summary.json, figures
```

Run the whole thing with:
```bash
pip install torch transformers geopandas rasterio pyproj shapely rapidfuzz xgboost scikit-learn
python3 src/pipeline.py
python3 src/visualize.py
```

## What to build next (in priority order)

1. **Swap in real data**: replace `synthetic_data.py` with actual drone orthomosaic tiles + real cadastral/municipal/utility/revenue exports (the rest of the pipeline doesn't need to change — it already operates on standard GeoDataFrames).
2. **Real SAM**: once you have internet/HF access, swap `sam_fallback.segment_region()` for a real `facebook/sam-vit-b` checkpoint call — the interface is already shaped to match (image patch + prompt → mask).
3. **PostGIS backend**: currently everything runs in-memory with GeoPandas; move storage/indexing to PostGIS for city-scale volumes as documented (Stage 8, gradual expansion).
4. **FastAPI + Celery + React**: wrap `pipeline.py`'s stages as async Celery tasks behind a FastAPI service, with the React review dashboard consuming `outputs/harmonized_output.geojson` and `scored_candidates.csv` for the human-in-the-loop screen.
5. **ICP / TPS alignment tiers**: implement Stage 3–4 of progressive alignment (`matching.py` has the extension point) for non-rigid legacy-survey distortion, which the synthetic data doesn't currently model.
6. **Calibrate thresholds on real pilot data**: `CONF_OK`, `CONF_SPECIALIZED_OK` (extraction) and `HIGH_THRESHOLD`, `LOW_THRESHOLD` (decision engine) are currently tuned to exercise the demo meaningfully — per the doc's own Section 22, these need recalibration against real, officially-verified match outcomes.
