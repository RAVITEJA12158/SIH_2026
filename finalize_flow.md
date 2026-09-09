╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 1 — MULTI-SOURCE DATA INGESTION                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

INPUT DATASETS
│
├── Drone / Aerial Imagery
├── Orthorectified Imagery (ORI / Orthomosaic)
├── DSM / DTM
├── Existing Cadastral Maps
├── Municipal GIS Layers
├── Utility Network Data
├── Revenue / Land Records
├── Existing Building Footprint Data
├── GNSS / CORS Survey Data
└── Ground Truth (GT) Data
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 2 — DATA INGESTION, PROFILING & CLASSIFICATION                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

GDAL + Rasterio + GeoPandas
                              │
                              ▼
Identify and Validate:
• Raster / Vector / CAD / Tabular / Survey data
• CRS
• Coordinate system
• Resolution
• Spatial extent
• Geometry type
• Attribute schema
• Missing values
• Data quality
• Source metadata
                              │
                              ▼
                    DATA TYPE CLASSIFICATION
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 3 — GEOREFERENCING & COORDINATE HARMONIZATION                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

                    Is spatial data georeferenced?
                              │
                   ┌──────────┴──────────┐
                   │                     │
                  YES                    NO
                   │                     │
                   ▼                     ▼
             CRS Check              Identify GCPs /
                   │                GNSS Control Points
                   │                     │
                   ▼                     ▼
           Coordinate Transform     Affine Transformation
                   │                     │
                   │             If local distortion:
                   │                     ▼
                   │             Thin Plate Spline
                   │             (TPS / Rubber Sheeting)
                   │                     │
                   └──────────────┬──────┘
                                  ▼

                        COMMON PROJECT CRS
                     (Metric CRS / UTM where suitable)
                                  │
                                  ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 4 — ORTHOMOSAIC / ORI + DSM/DTM PREPROCESSING                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Orthomosaic / ORI
        +
DSM / DTM
        │
        ▼
Preprocessing
│
├── CRS Alignment
├── Resolution Alignment / Resampling
├── Image Tiling
├── Normalization
├── Noise / Artifact Handling
└── nDSM Generation where required

nDSM = DSM − DTM
        │
        ▼
PREPARED MULTIMODAL INPUT
        │
        ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 5 — PRIMARY GEOAI FEATURE EXTRACTION                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

             ORTHOMOSAIC / ORI + DSM / DTM / nDSM
                              │
                              ▼
                   SEGFORMER — PRIMARY MODEL
                              │
                              ▼
                  MULTI-CLASS FEATURE MASKS
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼

    BUILDINGS               ROADS            WALLS / BOUNDARIES
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼

                     OTHER REQUIRED CLASSES
                              │
                              ▼
                 CLASS-WISE QUALITY ASSESSMENT
                              │
                ┌─────────────┴─────────────┐
                │                           │
               PASS                        FAIL
                │                           │
                │                           ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 6 — FEATURE-SPECIFIC SPECIALIZATION                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

                 FAILED FEATURE CLASS
                           │
                           ▼
                 CLASS-SPECIFIC ROUTING
                           │
       ┌───────────────────┼────────────────────┐
       │                   │                    │
       ▼                   ▼                    ▼

BUILDING / SURFACE      THIN / NARROW       INDIVIDUAL /
BOUNDARY REFINEMENT     LINEAR FEATURES     OVERLAPPING OBJECTS
       │                   │                    │
       ▼                   ▼                    ▼
     U-Net++             U-Net++          Mask R-CNN /
   or specialized      or boundary-       Instance
 segmentation model    aware model        Segmentation
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │
                           ▼

              OTHER FEATURE-SPECIFIC MODELS
              selected according to feature type
                           │
                           ▼
                   QUALITY CHECK AGAIN
                           │
                 ┌─────────┴─────────┐
                 │                   │
                PASS                FAIL
                 │                   │
                 │                   ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 7 — FINAL AI FALLBACK                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

                         SAM 3.1
                              │
                              ▼
               Prompt / Object / Region-Guided
                        Segmentation
                              │
                              ▼
                       QUALITY CHECK
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   PASS                FAIL
                    │                   │
                    ▼                   ▼
                ACCEPT           GT / MANUAL
                                  VALIDATION
                    │                   │
                    └──────────┬────────┘
                               ▼

                       ACCEPTED FEATURE MASKS
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 8 — MASK POST-PROCESSING & GIS VECTORIZATION                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

ACCEPTED FEATURE MASKS
          │
          ▼
Class-Specific Mask Cleaning
│
├── Connected Component Analysis
├── Morphological Opening / Closing
├── Hole Filling
├── Small Object Removal
└── Noise Filtering
          │
          ▼
                 ┌───────────────────────┐
                 │ FEATURE TYPE ROUTING  │
                 └───────────┬───────────┘
                             │
         ┌───────────────────┼─────────────────────┐
         │                   │                     │
         ▼                   ▼                     ▼

      POLYGON              LINE                  POINT
         │                   │                     │
         ▼                   ▼                     ▼

Contour Extraction      Skeletonization       Connected Components
/ Polygonization        / Centerline          / Object Detection
         │                   │                     │
         ▼                   ▼                     ▼

Douglas–Peucker      Line Simplification      Centroid Extraction
Simplification
         │                   │                     │
         └───────────────────┼─────────────────────┘
                             ▼

                    GEOMETRY CLEANING
                             │
├── ST_MakeValid
├── ST_Snap
├── Remove Self-Intersections
├── Remove Slivers
├── Gap / Overlap Detection
└── Topology Correction
                             │
                             ▼

                  AI-DERIVED GIS FEATURE LAYERS
                             │
                             ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 9 — CADASTRAL MAP HARMONIZATION                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

INPUT CADASTRAL DATA
        │
        ▼
TYPE CLASSIFICATION
        │
 ┌──────┼───────────────┬───────────────┬──────────────┐
 │      │               │               │              │
 ▼      ▼               ▼               ▼              ▼

GEOREF. SCANNED         CAD / DWG       TEXT /        SHEET
VECTOR  RASTER          DATA            SURVEY DATA   INDEX
 │      │               │               │              │
 │      ▼               ▼               ▼              ▼
 │   GCP Selection    CRS Check       COGO           Select
 │      │             / Helmert       Traverse       Correct
 │      ▼               │             / Parcel       Sheet
 │ Affine / TPS         ▼             Reconstruction
 │      │           Vector Conversion       │
 │      ▼               │                   │
 │ Raster               └─────────┬─────────┘
 │ Vectorization                  │
 └────────────────────────────────┘
                                  │
                                  ▼

                    COMMON PARCEL GIS LAYER
                                  │
                                  ▼

                     PARCEL TOPOLOGY CHECK
│
├── Self-intersection
├── Gaps
├── Overlaps
├── Slivers
├── Invalid geometry
└── Boundary consistency
                                  │
                                  ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 10 — CADASTRAL ↔ AI FEATURE SPATIAL ASSOCIATION                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

PARCEL GIS LAYER
        +
AI-DERIVED GIS FEATURE LAYERS
        │
        ▼
STRtree / R-tree
Candidate Generation
        │
        ▼
Spatial Relationship Analysis
│
├── ST_Contains
├── ST_Within
├── ST_Intersects
├── ST_Distance
└── Nearest Feature Search
        │
        ▼

PARCEL-CENTRIC SPATIAL BASE

Example:
Building → Which Parcel?
Road → Which Parcels?
Wall → Which Boundary?
Utility → Which Parcel / RoW?
        │
        ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 11 — MUNICIPAL GIS + BUILDING FOOTPRINT INTEGRATION                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Municipal GIS / Existing Building Footprints
                        │
                        ▼
                  CRS Harmonization
                        │
                        ▼
                  Initial Overlay
                        │
                        ▼
             STRtree Candidate Generation
                        │
                        ▼
              MATCHING FEATURE CALCULATION
│
├── IoU / Spatial Overlap
├── Distance
├── Area Ratio
├── Shape Similarity
└── Hausdorff Distance
                        │
                        ▼
                LIKELY CORRESPONDENCES
                        │
                        ▼
                      RANSAC
               Remove Wrong Matches
                        │
                        ▼
               Is Systematic Shift Present?
                        │
                 ┌──────┴──────┐
                 │             │
                NO            YES
                 │             │
                 │             ▼
                 │     Affine / Similarity
                 │       Transformation
                 │             │
                 └──────┬──────┘
                        ▼

                  CHANGE DETECTION
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        MATCHED       MODIFIED      NEW / REMOVED
                        │
                        ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 12 — UTILITY NETWORK INTEGRATION & SPATIAL ANALYTICS                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

UTILITY GIS DATA
        ↓
CRS STANDARDIZATION
        ↓
BUILD STRtree SPATIAL INDEX
        ↓
CANDIDATE FEATURE SEARCH
        ↓
SELECTED CANDIDATE FEATURES
        ↓
CALCULATE GEOMETRIC MATCHING FEATURES
        │
        ├── Distance
        ├── Node Correspondence
        ├── Line-to-Line Comparison
        ├── Hausdorff Distance
        └── Discrete Fréchet Distance
        │
        ↓
GEOMETRIC MATCHING EVIDENCE
        ↓
RANSAC
        ↓
REMOVE WRONG / OUTLIER CORRESPONDENCES
        ↓
ICP
        ↓
REFINE GEOMETRIC ALIGNMENT
        ↓
TPS IF LOCAL DISTORTION EXISTS
        ↓
HARMONIZED UTILITY GIS LAYER
        ↓
UTILITY SPATIAL ANALYTICS
        │
        ├── ST_Buffer → Right-of-Way / Safety Corridor
        ├── ST_Intersects → Encroachment Detection
        ├── ST_Intersection → Conflict Area
        ├── ST_Distance → Clearance Analysis
        └── nDSM → Vertical Clearance
        ↓
UTILITY ↔ PARCEL LINK
        ↓
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 13 — REVENUE RECORD & INTELLIGENT ATTRIBUTE INTEGRATION               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Revenue / Land Records
          │
          ▼
Data Cleaning
          │
          ▼
Attribute Normalization
│
├── Standardize Parcel IDs
├── Standardize Survey Numbers
├── Normalize Owner / Land Attributes
├── Handle Missing Values
└── Standardize Formats
          │
          ▼
       EXACT ATTRIBUTE MATCH
          │
├── Parcel ID
├── Survey Number
└── Plot Number
          │
     ┌────┴────┐
     │         │
   MATCH     NO MATCH
     │         │
     │         ▼
     │   Fuzzy Matching
     │
     │   Levenshtein Distance
     │
     │   Attribute Similarity
     │         │
     └─────────┘
          │
          ▼
      INTELLIGENT ATTRIBUTE MAPPING
          │
          ▼
     PARCEL ↔ REVENUE RECORD LINK
          │
          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 14 — GNSS / CORS + GROUND TRUTH VALIDATION                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

GNSS / CORS Survey Data
          +
Ground Truth Records
          │
          ▼

REFERENCE / VALIDATION DATA
          │
          ▼

Validate:
│
├── Orthomosaic Position Accuracy
├── Cadastral Alignment
├── AI-Extracted Features
├── Municipal GIS Alignment
├── Utility Alignment
└── Conflict / Uncertain Matches
          │
          ▼

Accuracy Metrics:
│
├── RMSE
├── Positional Offset
├── Hausdorff Distance
└── Distance Error
          │
          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 15 — MULTI-SOURCE MATCHING + XGBOOST CONFIDENCE ENGINE                ║
╚══════════════════════════════════════════════════════════════════════════════╝

COLLECT MATCHING EVIDENCE
          │
          ├── Spatial Evidence
          │   • IoU
          │   • Distance
          │   • Intersection
          │   • Containment
          │
          ├── Geometric Evidence
          │   • Area Similarity
          │   • Shape Similarity
          │   • Hausdorff Distance
          │   • Fréchet Distance
          │
          ├── Attribute Evidence
          │   • Parcel ID
          │   • Survey Number
          │   • Plot Number
          │   • Fuzzy Similarity
          │
          └── Quality Evidence
              • GNSS Accuracy
              • GT Validation
              • Source Reliability
              • Geometry Quality
                        │
                        ▼

                       XGBOOST
        ML-BASED MATCH CLASSIFICATION MODEL
                        │
                        ▼

             MATCH CONFIDENCE SCORE
                        │
            ┌───────────┼────────────┐
            ▼           ▼            ▼

          HIGH       MEDIUM         LOW
       CONFIDENCE   CONFIDENCE   CONFIDENCE
            │           │            │
            ▼           ▼            ▼

          ACCEPT     GT / REVIEW   CONFLICT /
                                    NO MATCH
                        │
                        ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 16 — SPATIAL CONFLICT RESOLUTION                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Detect Conflicts
│
├── Positional Conflict
├── Boundary Conflict
├── Geometry Conflict
├── Attribute Conflict
└── Source-to-Source Conflict
                        │
                        ▼

                  EVIDENCE RANKING
│
├── GNSS Accuracy
├── Ground Truth
├── Source Authority
├── Geometry Quality
└── XGBoost Confidence
                        │
              ┌─────────┴─────────┐
              │                   │
             CLEAR              UNCERTAIN
              │                   │
              ▼                   ▼

         AUTO RESOLVE        REVIEW QUEUE
              │                   │
              └─────────┬─────────┘
                        ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 17 — HARMONIZED POSTGIS SPATIAL DATABASE                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

                  PostgreSQL + PostGIS
                            │
                            ▼

                   PARCEL-CENTRIC MODEL

Parcel_ID
│
├── Cadastral Boundary
├── AI Extracted Buildings
├── Roads / Walls / Features
├── Existing Building Footprints
├── Municipal GIS Information
├── Utility Relationships
├── Revenue / Land Records
├── GNSS / CORS Validation
├── Ground Truth Validation
├── Match Confidence
├── Conflict Status
├── Change Status
└── Source Metadata / Version
                            │
                            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 18 — CHANGE DETECTION & SYNCHRONIZATION                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

CURRENT AI-DERIVED GIS
          │
          ▼
Compare With:
│
├── Historical GIS
├── Existing Municipal GIS
├── Existing Building Footprints
└── Previous Survey Data
          │
          ▼

CHANGE METRICS
│
├── IoU Difference
├── Hausdorff Distance
├── Area Difference
├── Geometry Difference
└── Attribute Difference
          │
          ▼

CHANGE CLASSIFICATION
│
├── NO CHANGE
├── MODIFIED
├── NEW
└── REMOVED
          │
          ▼

DATABASE VERSIONING + SYNCHRONIZATION
          │
          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHASE 19 — API INTEGRATION & WEB GIS                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

                         WEB GIS
                            │
                            ▼

BASE MAP
├── Orthomosaic / ORI
│
OVERLAY GIS LAYERS
├── Parcel Boundaries
├── AI-Extracted Buildings
├── Roads
├── Walls / Boundaries
├── Municipal GIS
├── Utility Networks
├── Existing Building Footprints
├── Change Detection Layer
└── Confidence / Conflict Layer
                            │
                            ▼

                      USER CLICKS
                       A PARCEL
                            │
                            ▼

                        PARCEL_ID
                            │
                            ▼

                  POSTGIS SPATIAL QUERY

ST_Intersects
ST_Contains
ST_Within
ST_Distance
Attribute Joins
                            │
                            ▼

                       API RESPONSE
                            │
                            ▼

             COMPLETE INTEGRATED LAND RECORD
