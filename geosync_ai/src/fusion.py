"""
GeoSync AI - Multi-Source Evidence Fusion
--------------------------------------------
Cadastral is treated as the authoritative parcel/survey layer (as it
is in the real workflow). For every candidate (drone-extracted
footprint <-> cadastral parcel) pair, this module looks for
CORROBORATING evidence from the other independently-collected
government sources (municipal GIS, utility records, revenue records)
near that same location, using each source's own geometry and its
owner-name attribute (fuzzy-matched against the cadastral owner_name
with RapidFuzz).

This produces exactly the feature set the poster describes:
"XGBoost fuses spatial, geometric, and attribute evidence across
cadastral, municipal, utility, revenue, and drone data."
"""
import numpy as np
import pandas as pd
from matching import spatial_candidates, geometric_evidence, attribute_evidence, CANDIDATE_BUFFER_M


def _best_support(cadastral_row, gov_gdf, attribute_fields=("owner_name",), buffer_m=CANDIDATE_BUFFER_M):
    """Best supporting evidence for one cadastral parcel from another
    government source (e.g. municipal GIS looking to confirm a cadastral
    parcel). Returns (present, best_iou, best_attr_score)."""
    if gov_gdf is None or len(gov_gdf) == 0:
        return 0, 0.0, 0.0
    cad_geom = cadastral_row.geometry
    nearby = gov_gdf[gov_gdf.geometry.distance(cad_geom) < buffer_m]
    if len(nearby) == 0:
        return 0, 0.0, 0.0
    best_iou, best_attr = 0.0, 0.0
    for _, r in nearby.iterrows():
        gev = geometric_evidence(cad_geom, r.geometry)
        aev = attribute_evidence(cadastral_row.to_dict(), r.to_dict(), attribute_fields)
        best_iou = max(best_iou, gev["iou"] if gev["iou"] > 0 else max(0, 1 - gev["centroid_dist_m"] / buffer_m))
        best_attr = max(best_attr, aev)
    return 1, float(best_iou), float(best_attr)


def build_fused_candidate_table(extracted_gdf, cadastral_gdf, municipal_gdf, utility_gdf, revenue_gdf):
    """One row per (extracted footprint, cadastral parcel) candidate pair,
    enriched with cross-source support columns."""
    from matching import build_candidate_table
    base = build_candidate_table(extracted_gdf, cadastral_gdf, "cadastral", attribute_fields=("owner_name",))
    if len(base) == 0:
        return base

    muni_present, muni_iou, muni_attr = [], [], []
    util_present, util_iou = [], []
    rev_present, rev_attr = [], []

    for _, row in base.iterrows():
        cad_row = cadastral_gdf.iloc[int(row["gov_idx"])]

        p, iou, attr = _best_support(cad_row, municipal_gdf, ("owner_name",))
        muni_present.append(p); muni_iou.append(iou); muni_attr.append(attr)

        p, iou, _ = _best_support(cad_row, utility_gdf, ())
        util_present.append(p); util_iou.append(iou)

        p, _, attr = _best_support(cad_row, revenue_gdf, ("owner_name",))
        rev_present.append(p); rev_attr.append(attr)

    base["municipal_present"] = muni_present
    base["municipal_iou"] = muni_iou
    base["municipal_attr_score"] = muni_attr
    base["utility_present"] = util_present
    base["utility_iou"] = util_iou
    base["revenue_present"] = rev_present
    base["revenue_attr_score"] = rev_attr
    base["cross_source_support_count"] = (
        base["municipal_present"] + base["utility_present"] + base["revenue_present"]
    )
    return base


if __name__ == "__main__":
    import geopandas as gpd
    from settings import DATA_DIR
    extracted = gpd.read_file(DATA_DIR / "extracted_features.geojson")
    cadastral = gpd.read_file(DATA_DIR / "cadastral.geojson")
    municipal = gpd.read_file(DATA_DIR / "municipal.geojson")
    utility = gpd.read_file(DATA_DIR / "utility.geojson")
    revenue = gpd.read_file(DATA_DIR / "revenue.geojson")

    table = build_fused_candidate_table(extracted, cadastral, municipal, utility, revenue)
    table.to_csv(DATA_DIR / "fused_candidates.csv", index=False)
    print(table[["extracted_idx", "gov_idx", "iou", "attribute_score",
                 "municipal_present", "municipal_iou", "utility_present",
                 "revenue_present", "cross_source_support_count"]].head(12))
    print(f"\n{len(table)} fused candidate rows -> data/fused_candidates.csv")
