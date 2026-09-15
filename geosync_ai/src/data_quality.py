"""Dataset profiling and quality gates for the GeoSync prototype.

This module deliberately reports quality problems instead of silently editing
source records. A production service would retain the original data, the
normalised working copy, and a durable remediation workflow in PostGIS.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Mapping

import geopandas as gpd


def _bbox_as_list(gdf: gpd.GeoDataFrame) -> list[float] | None:
    if gdf.empty:
        return None
    return [round(float(value), 3) for value in gdf.total_bounds]


def profile_layer(name: str, gdf: gpd.GeoDataFrame) -> dict:
    """Return a concise, serialisable quality profile for one vector layer."""
    geometry = gdf.geometry
    wkb = geometry.to_wkb() if len(gdf) else geometry
    missing_by_field = {
        column: int(gdf[column].isna().sum())
        for column in gdf.columns
        if column != gdf.geometry.name
    }
    invalid = int((~geometry.is_valid).sum()) if len(gdf) else 0
    empty = int(geometry.is_empty.sum()) if len(gdf) else 0
    duplicate_geometries = int(wkb.duplicated().sum()) if len(gdf) else 0
    blocking_issues = []
    if gdf.crs is None:
        blocking_issues.append("missing_crs")
    if invalid:
        blocking_issues.append("invalid_geometry")
    if empty:
        blocking_issues.append("empty_geometry")

    return {
        "layer": name,
        "record_count": int(len(gdf)),
        "crs": str(gdf.crs) if gdf.crs else None,
        "geometry_type_counts": {str(k): int(v) for k, v in geometry.geom_type.value_counts().items()},
        "extent": _bbox_as_list(gdf),
        "invalid_geometry_count": invalid,
        "empty_geometry_count": empty,
        "duplicate_geometry_count": duplicate_geometries,
        "missing_values_by_field": missing_by_field,
        "blocking_issues": blocking_issues,
        "quality_gate": "pass" if not blocking_issues else "blocked",
    }


def profile_and_standardize_layers(
    layers: Mapping[str, gpd.GeoDataFrame],
    report_path: Path | None = None,
) -> tuple[dict[str, gpd.GeoDataFrame], dict]:
    """Profile layers and transform compatible layers into a common CRS.

    No geometry is repaired or discarded here. Invalid geometry and missing
    CRS are blocking issues because automatically correcting official data can
    conceal a source-data problem.
    """
    non_null_crs = [gdf.crs for gdf in layers.values() if gdf.crs is not None]
    common_crs = non_null_crs[0] if non_null_crs else None
    standardised: dict[str, gpd.GeoDataFrame] = {}
    profiles = []
    for name, layer in layers.items():
        working = layer.copy()
        if common_crs is not None and working.crs is not None and working.crs != common_crs:
            working = working.to_crs(common_crs)
        standardised[name] = working
        profiles.append(profile_layer(name, working))

    report = {
        "prototype_notice": (
            "Quality checks run on generated demonstration data. A production "
            "workflow must preserve original source files and route failures to "
            "an authorised remediation process."
        ),
        "common_processing_crs": str(common_crs) if common_crs else None,
        "overall_quality_gate": "pass" if all(p["quality_gate"] == "pass" for p in profiles) else "blocked",
        "layers": profiles,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return standardised, report


def require_quality_gate(report: dict) -> None:
    """Stop the pipeline before matching if a source has a blocking defect."""
    if report["overall_quality_gate"] != "pass":
        blocked = [p["layer"] for p in report["layers"] if p["quality_gate"] == "blocked"]
        raise ValueError(f"Quality gate blocked processing for: {', '.join(blocked)}")
