"""
GeoSync AI - Spatial Candidate Search & Multi-Factor Evidence
------------------------------------------------------------------
Implements doc sections 14-20:
  - Spatial candidate search via STRtree (avoid O(n*m) comparison)
  - Geometric evidence: IoU, Hausdorff distance, centroid distance,
    area-ratio similarity
  - Attribute evidence: RapidFuzz fuzzy string matching on
    owner-name / identifier fields
  - Progressive alignment (Stage 1-2 only, in this prototype): if a
    systematic offset is detected between two whole layers, a robust
    median translation estimate is applied before re-scoring.
    (ICP / TPS stages are stubbed with a clear extension point --
    they matter most for large non-rigid legacy-survey distortion,
    which the synthetic layers here don't model.)
"""
import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Point
from rapidfuzz import fuzz

CANDIDATE_BUFFER_M = 6.0   # metres, generous search radius for candidates


def spatial_candidates(source_gdf, target_gdf, buffer_m=CANDIDATE_BUFFER_M):
    """For every feature in target_gdf, return indices of source_gdf
    features whose (buffered) geometry intersects it -- this is the
    STRtree-accelerated 'restrict to nearby/overlapping candidates'
    step described in section 14, avoiding a full O(n*m) comparison."""
    geoms = list(source_gdf.geometry.values)
    tree = STRtree(geoms)
    pairs = []
    for j, tgeom in enumerate(target_gdf.geometry.values):
        buffered = tgeom.buffer(buffer_m)
        idxs = tree.query(buffered)
        for i in idxs:
            i = int(i)
            if geoms[i].intersects(buffered):
                pairs.append((i, j))
    return pairs


def estimate_systematic_offset(source_gdf, target_gdf, pairs):
    """Robust median translation estimate between two layers, using centroid
    differences of the current candidate pairs. It detects a simple whole-
    layer shift (dx, dy); it is not RANSAC and does not model rotation,
    scale, or non-rigid distortion."""
    if not pairs:
        return 0.0, 0.0
    dxs, dys = [], []
    for i, j in pairs:
        c1 = source_gdf.geometry.values[i].centroid
        c2 = target_gdf.geometry.values[j].centroid
        d = c1.distance(c2)
        if d < 5.0:  # only use "close enough to plausibly be the same object"
            dxs.append(c2.x - c1.x)
            dys.append(c2.y - c1.y)
    if not dxs:
        return 0.0, 0.0
    # Robust central estimate for the simple translation scenario.
    return float(np.median(dxs)), float(np.median(dys))


def geometric_evidence(geom_a, geom_b):
    inter = geom_a.intersection(geom_b).area
    union = geom_a.union(geom_b).area
    iou = inter / union if union > 0 else 0.0

    centroid_dist = geom_a.centroid.distance(geom_b.centroid)

    area_a, area_b = geom_a.area, geom_b.area
    area_ratio = min(area_a, area_b) / max(area_a, area_b) if max(area_a, area_b) > 0 else 0.0

    hausdorff = geom_a.hausdorff_distance(geom_b)

    return {
        "iou": iou,
        "centroid_dist_m": centroid_dist,
        "area_ratio": area_ratio,
        "hausdorff_m": hausdorff,
    }


def attribute_evidence(attrs_a: dict, attrs_b: dict, fields=("owner_name",)):
    scores = []
    for f in fields:
        va, vb = attrs_a.get(f), attrs_b.get(f)
        if va and vb:
            scores.append(fuzz.token_sort_ratio(str(va), str(vb)) / 100.0)
    return float(np.mean(scores)) if scores else 0.0


def build_candidate_table(extracted_gdf, gov_gdf, gov_name, attribute_fields=("owner_name",)):
    """Full evidence table for every (extracted feature, government record)
    candidate pair within search range."""
    pairs = spatial_candidates(extracted_gdf, gov_gdf)
    dx, dy = estimate_systematic_offset(extracted_gdf, gov_gdf, pairs)
    if abs(dx) > 0.3 or abs(dy) > 0.3:
        from shapely.affinity import translate
        aligned_geoms = [translate(g, xoff=dx, yoff=dy) for g in extracted_gdf.geometry.values]
    else:
        aligned_geoms = list(extracted_gdf.geometry.values)
        dx = dy = 0.0

    rows = []
    for i, j in pairs:
        ga = aligned_geoms[i]
        gb = gov_gdf.geometry.values[j]
        gev = geometric_evidence(ga, gb)
        aev = attribute_evidence(extracted_gdf.iloc[i].to_dict(), gov_gdf.iloc[j].to_dict(), attribute_fields)
        row = {
            "extracted_idx": i,
            "gov_idx": j,
            "gov_source": gov_name,
            "extraction_confidence": extracted_gdf.iloc[i].get("extraction_confidence", np.nan),
            "extraction_method": extracted_gdf.iloc[i].get("extraction_method", "unknown"),
            "applied_offset_dx": dx,
            "applied_offset_dy": dy,
            **gev,
            "attribute_score": aev,
        }
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import geopandas as gpd
    from settings import DATA_DIR
    extracted = gpd.read_file(DATA_DIR / "extracted_features.geojson")
    cadastral = gpd.read_file(DATA_DIR / "cadastral.geojson")
    df = build_candidate_table(extracted, cadastral, "cadastral", attribute_fields=("owner_name",))
    print(df.head(10))
    print(f"\n{len(df)} candidate pairs found between {len(extracted)} extracted features "
          f"and {len(cadastral)} cadastral records.")
