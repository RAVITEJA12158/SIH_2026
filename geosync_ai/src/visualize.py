import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import pandas as pd
import numpy as np
import json
from settings import DATA_DIR, OUTPUTS_DIR, ensure_project_directories

DATA = DATA_DIR
OUT = OUTPUTS_DIR

TIER_COLOR = {"high": "#1a9850", "medium": "#fee08b", "low": "#d73027"}
METHOD_STYLE = {"segformer": "-", "specialized": "--", "sam_fallback": ":"}


def make_overlay_figure():
    extracted = gpd.read_file(f"{DATA}/extracted_features.geojson")
    cadastral = gpd.read_file(f"{DATA}/cadastral.geojson")
    municipal = gpd.read_file(f"{DATA}/municipal.geojson")
    truth = gpd.read_file(f"{DATA}/drone_truth_footprints.geojson")
    harmonized = gpd.read_file(f"{OUT}/harmonized_output.geojson")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ---- Panel 1: raw multi-source overlay (the "before harmonization" problem) ----
    ax = axes[0]
    cadastral.boundary.plot(ax=ax, color="#3182bd", linewidth=1.4, label="Cadastral")
    municipal.boundary.plot(ax=ax, color="#e6550d", linewidth=1.2, linestyle="--", label="Municipal GIS")
    extracted.boundary.plot(ax=ax, color="#31a354", linewidth=1.0, linestyle=":", label="AI-Extracted (drone)")
    ax.set_title("Raw Multi-Source Overlay\n(positional drift + missing/extra records, before harmonization)")
    handles = [
        mpatches.Patch(edgecolor="#3182bd", facecolor="none", label="Cadastral"),
        mpatches.Patch(edgecolor="#e6550d", facecolor="none", label="Municipal GIS"),
        mpatches.Patch(edgecolor="#31a354", facecolor="none", label="AI-Extracted (drone)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, local)")
    ax.set_ylabel("Northing (m, local)")

    # ---- Panel 2: harmonized result, confidence-colored ----
    ax2 = axes[1]
    cadastral.boundary.plot(ax=ax2, color="0.75", linewidth=0.8, zorder=1)
    for tier, color in TIER_COLOR.items():
        sub = harmonized[harmonized["confidence_tier"] == tier]
        if len(sub):
            sub.plot(ax=ax2, color=color, alpha=0.55, edgecolor="black", linewidth=0.6, zorder=2)
    conflicts = harmonized[harmonized["status"] == "unmatched_conflict"]
    if len(conflicts):
        conflicts.boundary.plot(ax=ax2, color="black", linewidth=2.2, linestyle="-.", zorder=3)

    ax2.set_title("Harmonized Result\n(XGBoost confidence tier; hatched outline = unresolved conflict)")
    legend_handles = [mpatches.Patch(facecolor=c, edgecolor="black", alpha=0.55, label=f"{t.title()} confidence")
                       for t, c in TIER_COLOR.items()]
    legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="black", linewidth=2, linestyle="-.",
                                          label="Unresolved conflict"))
    ax2.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Easting (m, local)")
    ax2.set_ylabel("Northing (m, local)")

    plt.tight_layout()
    fig.savefig(f"{OUT}/overlay_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT}/overlay_comparison.png")


def make_extraction_scene_figure():
    """Shows the synthetic orthomosaic + extraction-method map (which
    tier -- SegFormer / Specialized / SAM -- resolved each region)."""
    img = np.load(f"{DATA}/ortho_image.npy")
    extracted = gpd.read_file(f"{DATA}/extracted_features.geojson")
    truth = gpd.read_file(f"{DATA}/drone_truth_footprints.geojson")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img)
    axes[0].set_title("Synthetic Drone Orthomosaic\n(shadows / tree-canopy occlusion / dense layout)")
    axes[0].axis("off")

    axes[1].imshow(img)
    method_colors = {"segformer": "#31a354", "specialized": "#fdae61", "sam_fallback": "#d73027"}
    # reproject extracted (in metres) back to pixel space for overlay isn't
    # trivial here without the inverse transform; instead show truth boxes
    # colored by nearest extracted feature's method for a quick visual.
    from shapely.geometry import box as shp_box
    import synthetic_data as sd
    for _, row in extracted.iterrows():
        geo = row.geometry
        minx, miny, maxx, maxy = geo.bounds
        px0, py0 = minx / sd.GSD, sd.IMG_SIZE - maxy / sd.GSD
        px1, py1 = maxx / sd.GSD, sd.IMG_SIZE - miny / sd.GSD
        c = method_colors.get(row["extraction_method"], "white")
        rect = plt.Rectangle((px0, py0), px1 - px0, py1 - py0, fill=False, edgecolor=c, linewidth=2)
        axes[1].add_patch(rect)
    axes[1].set_title("Extraction Method per Footprint\n(green=SegFormer, orange=Specialized, red=SAM-fallback)")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(f"{OUT}/extraction_scene.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT}/extraction_scene.png")


if __name__ == "__main__":
    ensure_project_directories()
    make_overlay_figure()
    make_extraction_scene_figure()
