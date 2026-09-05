┌──────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — MULTI-SOURCE DATA INPUT                       │
└──────────────────────────────────────────────────────────────────────────────┘

  Drone Imagery ───────────────┐
  Orthorectified Imagery (ORI) ─┤
  DSM / DTM ────────────────────┤
  Existing Cadastral Maps ──────┤
  Municipal GIS Layers ─────────┤
  Utility Network Data ─────────┤
  Building Footprint Datasets ──┤
  Revenue / Land Records ───────┤
  GNSS / CORS Survey Data ──────┤
  Ground Truth (GT) Data ───────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — DATA INGESTION, ETL & PROFILING                                  │
│ GDAL + Rasterio + GeoPandas + API / ETL Connectors                          │
│                                                                              │
│ • Identify Raster / Vector / CAD / Tabular / Scanned data                   │
│ • CRS detection  • Spatial extent  • Resolution  • Schema inspection        │
│ • Missing-data and source-quality checks                                    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — GEOREFERENCING & COORDINATE STANDARDIZATION                       │
│ GDAL / PROJ / PyProj                                                        │
│                                                                              │
│ CRS Transformation → Common Project CRS                                     │
│ GCP / GNSS control points → Affine / Helmert / TPS when required            │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼


═══════════════════════ PARALLEL DATA PROCESSING ═══════════════════════════


 ┌──────────────────────────┐   ┌──────────────────────────┐
 │ A. IMAGE / ELEVATION     │   │ B. CADASTRAL DATA        │
 │ PROCESSING               │   │ HARMONIZATION            │
 └────────────┬─────────────┘   └────────────┬─────────────┘
              │                              │
 ORI / Orthomosaic + DSM/DTM          Vector / Scan / CAD /
              │                        Survey Text / Sheet Index
              ▼                              │
       DSM − DTM = nDSM                      ▼
              │                       Type-specific processing
              ▼                              │
 AI FEATURE EXTRACTION                       ▼
 SegFormer + U-Net++                  Georeferencing /
              │                       COGO / Vectorization
              ▼                              │
 Buildings / Roads /                   Common Parcel Layer
 Walls / Boundaries                           │
              │                              │
              ▼                              │
 Mask Cleaning                           Geometry Validation
 Opening / Closing /                      ST_MakeValid()
 Hole Filling / CCA                              │
              │                              │
              ▼                              │
 Contour Extraction /                         │
 Skeletonization                               │
              │                              │
              ▼                              │
 Polygonization / Centerlines                  │
              │                              │
              ▼                              │
 Douglas–Peucker                              │
              │                              │
              ▼                              │
 AI-DERIVED CURRENT GIS LAYERS                 │
              └──────────────┬───────────────┘
                             ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│              PHASE 4 — AUTOMATED TOPOLOGY CORRECTION                        │
│                                                                              │
│ ST_MakeValid() • ST_Snap() • ST_SnapToGrid()                                │
│ ST_RemoveRepeatedPoints() • ST_UnaryUnion()                                 │
│                                                                              │
│ Correct: invalid geometry / gaps / slivers / overshoots / undershoots        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5 — PARCEL-CENTRIC SPATIAL ASSOCIATION                                │
│                                                                              │
│ Cadastral Parcel Layer + AI-Derived GIS Layers                               │
│                ↓                                                             │
│         R-tree / STRtree                                                     │
│                ↓                                                             │
│         Candidate Generation                                                  │
│                ↓                                                             │
│ ST_Contains • ST_Within • ST_Intersects • ST_Overlaps • ST_Distance          │
│                ↓                                                             │
│         PARCEL-CENTRIC SPATIAL BASE                                          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼


════════════════════ MULTI-SOURCE SPATIAL HARMONIZATION ════════════════════


┌──────────────────────────┐  ┌──────────────────────────┐
│ MUNICIPAL GIS LAYERS     │  │ BUILDING FOOTPRINT DATA  │
└────────────┬─────────────┘  └────────────┬─────────────┘
             ▼                             ▼
      Initial Overlay                  STRtree / R-tree
             │                             │
             ▼                             ▼
 IoU + Distance + Shape             Candidate Pairs
 + Hausdorff Distance                      │
             │                             ▼
             ▼                    IoU + Hausdorff +
     RANSAC if needed              Area Ratio + Shape
             │                             │
             ▼                             ▼
 Similarity / Affine / TPS        MATCH / NEW /
 if alignment correction needed   MODIFIED / REMOVED
             │                             │
             └──────────────┬──────────────┘
                            │


┌──────────────────────────┐  ┌──────────────────────────┐
│ UTILITY NETWORK DATA     │  │ REVENUE / LAND RECORDS   │
└────────────┬─────────────┘  └────────────┬─────────────┘
             ▼                             ▼
 Point / Line Geometry              Attribute Cleaning
             │                             │
             ▼                             ▼
 STRtree / R-tree                   Intelligent Attribute
             │                       Normalization
             ▼                             │
 Visible assets?                         ▼
       │                           Exact Parcel ID
  ┌────┴─────┐                     Survey Number
  │ YES  NO  │                     Crosswalk Table
  ▼      ▼   ▼                     Levenshtein /
 Compare  Direct                  Fuzzy Matching
 with AI  Coordinate                     │
 GIS      Integration                     ▼
  │         │                      ATTRIBUTE MATCH
  ▼         ▼
 RANSAC / GNSS / GT
 ICP only if validation
 alignment is justified
       │
       ▼
 UTILITY GIS LAYER
       │
       └─────────────────────┬────────────────────────────┘
                             ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 6 — GNSS / CORS & GROUND TRUTH VALIDATION                             │
│                                                                              │
│ GNSS/CORS coordinates + Ground Truth observations                            │
│                ↓                                                             │
│ RMSE • Positional Offset • Hausdorff Distance                                │
│                ↓                                                             │
│ Position / Match Validation Score                                            │
│                                                                              │
│ Used especially for uncertain matches and accuracy verification              │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 7 — MULTI-SOURCE MATCH FEATURE ENGINEERING                            │
│                                                                              │
│ SPATIAL:   IoU • Distance • Containment • Intersection • Topology            │
│ GEOMETRY:  Hausdorff • Fréchet • Area Ratio • Shape Similarity               │
│ ATTRIBUTE: Parcel ID • Survey No • Municipal ID • Fuzzy Similarity           │
│ QUALITY:   Source Accuracy • GNSS Error • GT Validation • Temporal Recency   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 8 — XGBOOST CONFIDENCE SCORING                                        │
│                                                                              │
│ All Spatial + Geometric + Attribute + Quality Features                       │
│                              ↓                                               │
│                         XGBOOST                                              │
│                              ↓                                               │
│                   Match Probability / Score                                 │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                       MATCH       UNCERTAIN      NO MATCH
                         │             │             │
                         │             ▼             │
                         │       GNSS / GT /         │
                         │       Manual Review       │
                         └─────────────┬─────────────┘
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 9 — SPATIAL CONFLICT DETECTION & RESOLUTION                           │
│                                                                              │
│ Detect: Positional • Geometry • Attribute • Temporal Conflicts               │
│                              ↓                                               │
│ Evidence Priority:                                                          │
│ GNSS/CORS + Ground Truth + Source Authority + Positional Accuracy            │
│ + XGBoost Confidence + Temporal Recency                                      │
│                              ↓                                               │
│                    AUTO-RESOLVE OR REVIEW                                    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 10 — HARMONIZED SPATIAL DATABASE                                     │
│                                                                              │
│                       PostgreSQL + PostGIS                                  │
│                              ↓                                               │
│                     PARCEL-CENTRIC MODEL                                    │
│                                                                              │
│ Parcel_ID                                                                    │
│ ├── Cadastral Geometry & Attributes                                         │
│ ├── AI-Detected Features                                                    │
│ ├── Municipal GIS Information                                               │
│ ├── Building Footprints & Change Status                                     │
│ ├── Utility Network Relationships                                           │
│ ├── Revenue / Land Records                                                  │
│ ├── GNSS / GT Validation                                                    │
│ ├── XGBoost Confidence Score                                                │
│ ├── Conflict / Review Status                                                │
│ └── Source Metadata & Data Provenance                                       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 11 — WEB GIS + API INTEGRATION                                        │
│                                                                              │
│                         Orthomosaic / ORI                                   │
│                              BASE MAP                                        │
│                                  +                                           │
│ Cadastral Parcels + AI GIS Layers + Municipal GIS + Utility Networks         │
│ + Building Footprints + Change Layer + Validation / Conflict Layer           │
│                                                                              │
│ User Clicks Parcel / Feature                                                 │
│              ↓                                                               │
│      Parcel_ID / Spatial Query                                               │
│              ↓                                                               │
│        PostGIS + API                                                        │
│              ↓                                                               │
│ COMPLETE HARMONIZED URBAN LAND RECORD                                       │
└──────────────────────────────────────────────────────────────────────────────┘