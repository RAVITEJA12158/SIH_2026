# GeoSync AI - Core Pipeline Prototype

**Team: The Sentinels | SIH26013 | Automated Integration and Intelligent Harmonization of Multi-Source Geospatial Data for Urban Land Record Management**

GeoSync AI is a working proof of concept for AI-assisted harmonization of drone-derived building footprints and multiple geospatial record layers. It executes end to end on synthetic geospatial data, producing reproducible intermediate files, proposed matches, confidence scores, and review-routing outputs.

## Prototype boundary

This repository is an SIH prototype, **not a production system and not a legal land-records decision system**. It must not be used to create, amend, validate, or adjudicate official records without an authorized human review process, real-data validation, security controls, and formal deployment approval.

All accuracy, matching, and runtime figures in this README are measurements from the synthetic demonstration corpus only. They are useful for evaluating the pipeline mechanics, but are **not evidence of real-world accuracy, operational reliability, or legal validity**.

## What is implemented and demonstrated

| Stage | Prototype implementation | Evidence boundary |
|---|---|---|
| Multi-source ingestion | A synthetic drone orthomosaic plus cadastral, municipal, utility, and revenue layers. Each source includes deliberately different imperfections such as shifts, missing records, name variation, and a legacy record. | Inputs are generated; they are not official datasets. |
| Primary extraction | A SegFormer architecture (`mit-b0` scale configuration, instantiated and trained from scratch) produces building-footprint masks. | The reported 0.925 IoU is against synthetic ground truth only. |
| Specialized fallback | A compact dilated U-Net is used for low-confidence or occluded tiles. | Trained and evaluated on synthetic scenes. |
| Final fallback interface | A prompt-driven classical region-growing implementation (Felzenszwalb superpixels plus edge-aware confidence) is available behind `sam_fallback.segment_region()`. | This is **not Segment Anything (SAM)**; it is an interface-compatible stand-in. |
| Adaptive routing | SegFormer confidence and an image-based occlusion/shadow signal decide whether a tile is escalated. | In the latest synthetic run, the adaptive mask scored 0.933 IoU versus 0.925 for the primary model alone. |
| Candidate search and evidence | STRtree candidate lookup, geometric features (IoU, Hausdorff and centroid distances, area ratio), and fuzzy owner-name comparison. | Parameters are demo settings, not validated acceptance criteria. |
| Alignment | A robust median translation estimate detects and compensates for a simple whole-layer offset. | This is not full RANSAC, ICP, or TPS registration; it only handles the synthetic translation scenario. |
| Cross-source fusion | Municipal, utility, and revenue evidence is added to a drone-to-cadastral candidate score. | Corroboration is demonstrated only on generated records. |
| Quality gate | Dataset profiling checks CRS presence, geometry validity, empty geometries, duplicate geometries, extents, and missing attribute values before matching. | The current report is generated from synthetic GeoJSON and blocks only missing or invalid geometries; production remediation must be governed. |
| Confidence and routing | An XGBoost scorer assigns high/medium/low prototype confidence and routes records to `proposed_auto_eligible`, `proposed_flagged`, or `manual_review`. | These are proposals only; `proposed_auto_eligible` means eligible for a future authorised workflow, not authorisation to update an official record. Thresholds are uncalibrated. |
| Review package | The pipeline generates an evidence-first JSON queue, local audit event, artifact-hash manifest, and self-contained offline review dashboard. | It is a read-only local demonstration, not an authenticated service or immutable audit system. |
| Traceable output | GeoJSON includes proposed status, matched attributes, confidence, extraction method, and cross-source support count. | Provenance is a useful prototype trace, not a tamper-evident audit record. |

## Latest synthetic-run results

- 22 synthetic buildings; 22 footprints extracted (18 via SegFormer, 4 via the specialized fallback).
- 21 synthetic cadastral records matched; 21 proposals received the high-confidence demo route and 1 was routed to manual review.
- Synthetic segmentation IoU: 0.925 (primary) and 0.933 (adaptive pipeline).
- Synthetic held-out-by-scene confidence experiment: precision 0.977, recall 1.000, F1 0.989, AUC 1.000 (273 train / 126 test candidate pairs across the generated corpus).
- Approximate CPU-only demonstration runtime: 125 seconds, including model training.

These figures should be presented as **synthetic prototype results**, never as field accuracy or a performance guarantee.

See `outputs/overlay_comparison.png` for the synthetic-layer harmonization visualization and `outputs/extraction_scene.png` for fallback routing.

After a pipeline run, open `outputs/review_dashboard.html` to inspect the proposed match queue, confidence evidence, and footprint overview. Supporting artifacts are `data_quality_report.json`, `review_queue.json`, `artifact_manifest.json`, and `audit_log.jsonl`.

## Security and governance design requirements

The following is the required security and governance layer for a real deployment. This repository demonstrates only limited local precursors: a quality report, a role-action policy map, a read-only review queue, artifact hashes, and an append-only local log. Those artifacts are **not security controls** and must not be used with official or personally identifiable data.

| Control area | Production requirement |
|---|---|
| Data classification and minimization | Classify imagery, parcel geometry, ownership data, and derived outputs; ingest only fields necessary for the approved workflow; redact or mask sensitive data in non-production environments. |
| Access control | Enforce least-privilege, role-based access (for example, data steward, analyst, reviewer, administrator, auditor), strong authentication, and separation between approval and system-administration roles. |
| Data protection | Encrypt data in transit and at rest; manage secrets outside source code; restrict backups, exports, and development copies; apply defined retention and secure disposal policies. |
| Provenance and audit | Record immutable, time-stamped events for ingestion, model/version selection, transformations, match proposals, reviewer actions, overrides, and exports. Each proposed change must be attributable to a user and source dataset version. |
| Human authority | AI may prioritize and propose matches only. An authorized official must review required cases and approve any record change; automated outputs cannot be treated as title, ownership, or legal adjudication. |
| Model governance | Version datasets, labels, code, and models; validate on representative, officially verified pilot cases; calibrate confidence thresholds; document known failure modes; monitor drift and reassess after material changes. |
| Quality and appeals | Preserve source geometry and original attributes, show evidence and alternatives to reviewers, flag conflicts, support correction/appeal workflows, and prohibit silent overwrites. |
| Operational security | Conduct threat modelling, vulnerability management, security testing, incident response exercises, backup/restore testing, and periodic access reviews before and during operation. |
| Compliance and hosting | Obtain the responsible authority's approval for data residency, retention, sharing, procurement, and applicable privacy/records obligations before onboarding official data. |

The technical architecture should therefore treat PostGIS, an authenticated API, a reviewer dashboard, a job queue, audit storage, and monitoring as deployment components, not as capabilities already provided by this prototype.

## Project structure

```
geosync_ai/
|- src/
|  |- synthetic_data.py       # generated data and orthomosaic
|  |- segformer_model.py      # primary extractor
|  |- specialized_model.py    # second-tier fallback
|  |- sam_fallback.py         # classical SAM stand-in
|  |- feature_extraction.py   # adaptive extraction routing
|  |- matching.py             # candidate search and matching evidence
|  |- fusion.py               # multi-source evidence fusion
|  |- confidence_model.py     # prototype XGBoost scorer
|  |- decision_engine.py      # prototype routing and conflicts
|  |- data_quality.py         # CRS/geometry/attribute quality gate
|  |- governance.py           # review-policy and traceability helpers
|  |- review_dashboard.py     # offline evidence-review dashboard
|  |- visualize.py            # demonstration figures
|  `- pipeline.py             # end-to-end synthetic run
|- models/                    # generated model weights
|- data/                      # generated inputs and intermediates
`- outputs/                   # generated GeoJSON, summary, and figures
```

Run the synthetic demonstration:

```bash
pip install torch transformers geopandas rasterio pyproj shapely rapidfuzz xgboost scikit-learn
python src/pipeline.py
python src/visualize.py
```

## Path to a pilot deployment

1. Validate with a governed, representative pilot dataset and independently verified outcomes; report error analysis by locality, imagery conditions, parcel type, and data-source quality.
2. Replace the SAM stand-in with an evaluated SAM implementation, if it demonstrably improves pilot results.
3. Implement ICP/TPS or another justified registration method for non-rigid survey distortions, with error bounds and rollback.
4. Move spatial storage and indexing to PostGIS; replace the local review artifacts with an authenticated FastAPI service, job queue, persistent reviewer interface, protected audit trail, monitoring, and backup/restore procedures.
5. Calibrate extraction and decision thresholds from official pilot outcomes, define review policy, and ensure that only authorized users can approve changes.
6. Complete the security and governance controls above, including formal risk assessment and deployment approval, before processing official data.
