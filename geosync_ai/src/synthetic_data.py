"""
GeoSync AI - Synthetic Data Generator
--------------------------------------
Generates:
  1. A synthetic drone orthomosaic (RGB raster) containing building
     footprints, plus a matching ground-truth segmentation mask
     (used to train/evaluate the SegFormer extractor).
  2. Multi-source "government" vector datasets (cadastral, municipal,
     utility, revenue) that describe the SAME physical area but with
     realistic imperfections: positional shift, missing parcels,
     extra/phantom parcels, and attribute-name drift (for RapidFuzz).

This stands in for real drone/CORS/cadastral inputs so the rest of
the pipeline (SegFormer -> specialized model -> SAM fallback ->
matching -> XGBoost confidence scoring -> decision engine) can be
exercised end-to-end without external data access.
"""
import json
import random
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate
import geopandas as gpd

RNG = np.random.default_rng(42)
random.seed(42)

IMG_SIZE = 512          # synthetic orthomosaic size in pixels
GSD = 0.10              # ground sample distance: 10 cm / pixel -> 51.2m x 51.2m tile
N_BUILDINGS = 22

OWNER_NAMES = [
    "Ramesh Kumar", "Sunita Devi", "Anil Sharma", "Priya Reddy",
    "Municipal Corporation", "State Housing Board", "Vijay Rao",
    "Lakshmi Narayana", "Suresh Babu", "Kavitha Iyer",
]
# deliberately drifted spellings used by the *other* department's records
OWNER_NAME_VARIANTS = {
    "Ramesh Kumar": "Ramesh  Kumarr",
    "Sunita Devi": "Sunita D.",
    "Anil Sharma": "Anil Sharma ",
    "Priya Reddy": "Priya  Reddi",
    "Municipal Corporation": "Municipal Corp.",
    "State Housing Board": "St. Housing Board",
    "Vijay Rao": "Vijaya Rao",
    "Lakshmi Narayana": "Lakshmi Narayanan",
    "Suresh Babu": "Suresh  Babu",
    "Kavitha Iyer": "Kavita Iyer",
}
LAND_USE = ["Residential", "Commercial", "Mixed Use", "Institutional"]


def _random_building_polygon(cx, cy, max_size=48, min_size=18):
    """Axis-ish aligned rectangle with slight rotation, in pixel coords."""
    w = RNG.uniform(min_size, max_size)
    h = RNG.uniform(min_size, max_size)
    angle = RNG.uniform(-8, 8)  # slight rotation, degrees
    poly = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    poly = _rotate(poly, angle, (cx, cy))
    return poly


def _rotate(poly, angle_deg, origin):
    from shapely.affinity import rotate
    return rotate(poly, angle_deg, origin=origin)


def generate_building_layout(n=N_BUILDINGS, size=IMG_SIZE, margin=30, min_sep=60):
    """Place non-overlapping building footprints across the tile (pixel coords)."""
    centers = []
    polys = []
    attempts = 0
    while len(polys) < n and attempts < n * 50:
        attempts += 1
        cx = RNG.uniform(margin, size - margin)
        cy = RNG.uniform(margin, size - margin)
        if any(np.hypot(cx - ox, cy - oy) < min_sep for ox, oy in centers):
            continue
        poly = _random_building_polygon(cx, cy)
        centers.append((cx, cy))
        polys.append(poly)
    return polys


def rasterize_mask(polys, size=IMG_SIZE):
    from rasterio.features import rasterize
    mask = rasterize(
        [(p, 1) for p in polys],
        out_shape=(size, size),
        fill=0,
        dtype="uint8",
    )
    return mask


def synthesize_orthomosaic(polys, size=IMG_SIZE, noise_level=18, add_occlusion=True):
    """Render a fake RGB orthomosaic: base terrain texture + building
    footprints with roof texture + shadows + trees/occlusion + sensor noise,
    so extraction is non-trivial (mirrors the real-world "shadows, trees,
    dense construction, unclear boundaries" challenge called out in the
    project documentation)."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    # base ground texture (soil/vegetation mix)
    base = RNG.uniform(80, 130, size=(size, size, 1))
    img[:] = base * np.array([0.85, 0.68, 0.48])  # tan/soil ground -- kept
    # visually distinct (red-dominant, not green-dominant) from the darker
    # saturated-green tree canopy blobs added below, so a simple color
    # heuristic can later distinguish "ground" from "occlusion" without
    # needing a trained model for that step.

    mask = rasterize_mask(polys, size)

    # roof texture per-building, with a directional drop shadow
    roof_colors = RNG.uniform(90, 170, size=(len(polys), 3))
    for i, poly in enumerate(polys):
        pmask = rasterize_mask([poly], size).astype(bool)
        img[pmask] = roof_colors[i]
        # cast shadow: shift mask down-right and darken where it doesn't
        # overlap the roof itself
        shifted = translate(poly, xoff=6, yoff=6)
        smask = rasterize_mask([shifted], size).astype(bool)
        shadow_only = smask & (~pmask)
        img[shadow_only] *= 0.55

    if add_occlusion:
        # random tree-canopy blobs partially occluding some roofs/edges
        for _ in range(14):
            tx, ty = RNG.uniform(0, size, size=2)
            r = RNG.uniform(8, 22)
            yy, xx = np.ogrid[:size, :size]
            blob = (xx - tx) ** 2 + (yy - ty) ** 2 <= r ** 2
            img[blob] = img[blob] * 0.3 + np.array([25, 70, 25]) * 0.7

    img += RNG.normal(0, noise_level, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask


def _px_to_geo(poly, origin=(0.0, 0.0), gsd=GSD, size=IMG_SIZE):
    """Convert a pixel-space polygon to a local metric CRS-like coordinate
    (meters), with y flipped so it behaves like a real georeferenced layer."""
    coords = [(origin[0] + x * gsd, origin[1] + (size - y) * gsd) for x, y in poly.exterior.coords]
    return Polygon(coords)


def build_government_layers(building_polys):
    """From the drone-truth building footprints, derive four independent
    government datasets with realistic, DIFFERENT imperfections each,
    exactly as described in the project doc (CRS/format/accuracy differ
    per source, some features missing, attribute drift, etc.)."""
    n = len(building_polys)
    owners = [random.choice(OWNER_NAMES) for _ in range(n)]
    landuse = [random.choice(LAND_USE) for _ in range(n)]

    records = {"cadastral": [], "municipal": [], "utility": [], "revenue": []}

    # --- Cadastral: fairly accurate, small survey noise, ~1 missing parcel
    drop_cadastral = set(random.sample(range(n), 1))
    for i, poly in enumerate(building_polys):
        if i in drop_cadastral:
            continue
        geo = _px_to_geo(poly)
        geo = translate(geo, xoff=RNG.normal(0, 0.15), yoff=RNG.normal(0, 0.15))
        records["cadastral"].append({
            "geometry": geo,
            "survey_no": f"{100+i}",
            "owner_name": owners[i],
            "land_use": landuse[i],
            "source": "cadastral",
            "_truth_id": i,
        })

    # --- Municipal GIS: systematic positional shift (different CRS datum
    # handling), ~2 missing, attribute name drift
    shift = (RNG.uniform(1.0, 2.2), RNG.uniform(1.0, 2.2))
    drop_municipal = set(random.sample(range(n), 2))
    for i, poly in enumerate(building_polys):
        if i in drop_municipal:
            continue
        geo = _px_to_geo(poly)
        geo = translate(geo, xoff=shift[0] + RNG.normal(0, 0.2), yoff=shift[1] + RNG.normal(0, 0.2))
        records["municipal"].append({
            "geometry": geo,
            "property_id": f"MP-2026-{1000+i}",
            "owner_name": OWNER_NAME_VARIANTS.get(owners[i], owners[i]),
            "building_use": landuse[i] if random.random() > 0.15 else random.choice(LAND_USE),
            "source": "municipal",
            "_truth_id": i,
        })

    # --- Utility records: only ~60% of parcels have utility connections;
    # geometry represented as a slightly buffered/simplified footprint
    utility_subset = random.sample(range(n), int(n * 0.6))
    for i in utility_subset:
        poly = building_polys[i]
        geo = _px_to_geo(poly).buffer(0.3).simplify(0.4)
        geo = translate(geo, xoff=RNG.normal(0, 0.25), yoff=RNG.normal(0, 0.25))
        records["utility"].append({
            "geometry": geo,
            "utility_id": f"UTL-{i:04d}",
            "connection_type": random.choice(["Water", "Electric", "Sewer"]),
            "owner_name": owners[i],
            "source": "utility",
            "_truth_id": i,
        })

    # --- Revenue records: attribute-heavy, occasional duplicate/extra
    # "phantom" record with no real spatial counterpart (legacy data error)
    for i, poly in enumerate(building_polys):
        geo = _px_to_geo(poly)
        geo = translate(geo, xoff=RNG.normal(0, 0.3), yoff=RNG.normal(0, 0.3))
        records["revenue"].append({
            "geometry": geo,
            "record_id": f"RR-2026-{200+i}",
            "owner_name": owners[i],
            "assessment_status": random.choice(["Assessed", "Pending", "Assessed"]),
            "source": "revenue",
            "_truth_id": i,
        })
    # one phantom legacy revenue record, far from any real building
    phantom = _px_to_geo(_random_building_polygon(30, 480, 20, 14))
    records["revenue"].append({
        "geometry": phantom, "record_id": "RR-2026-999", "owner_name": "Unknown Legacy Owner",
        "assessment_status": "Pending", "source": "revenue", "_truth_id": -1,
    })

    gdfs = {k: gpd.GeoDataFrame(v, geometry="geometry", crs="EPSG:32644") for k, v in records.items()}
    return gdfs, owners, landuse


def generate_training_scene(size=256, seed=101):
    """A cleaner reference scene (less occlusion) used only to TRAIN the
    primary SegFormer model -- mirrors training on well-annotated
    reference imagery before deploying on messier field-collected tiles."""
    global RNG
    prev = RNG
    RNG = np.random.default_rng(seed)
    polys = generate_building_layout(n=26, size=size, margin=20, min_sep=42)
    img, mask = synthesize_orthomosaic(polys, size=size, noise_level=14, add_occlusion=True)
    RNG = prev
    return img, mask


def generate_all(out_dir=None):
    if out_dir is None:
        from settings import DATA_DIR
        out_dir = DATA_DIR
    import os
    os.makedirs(out_dir, exist_ok=True)

    # cleaner scene the model is TRAINED on
    train_img, train_mask = generate_training_scene()
    np.save(f"{out_dir}/train_image.npy", train_img)
    np.save(f"{out_dir}/train_mask.npy", train_mask)

    # DEPLOYMENT scene: mostly normal difficulty (so the primary model
    # resolves most of the scene confidently and cheaply), but with a
    # handful of buildings DELIBERATELY placed under heavy tree-canopy
    # occlusion / overlapping shadow -- realistic "hard cases" (dense
    # construction, unclear boundaries) that should genuinely trigger
    # escalation to the specialized model / SAM-fallback tiers.
    polys = generate_building_layout(n=N_BUILDINGS, size=IMG_SIZE, margin=30, min_sep=60)
    img, mask = synthesize_orthomosaic(polys, size=IMG_SIZE, noise_level=18, add_occlusion=True)

    hard_idx = random.sample(range(len(polys)), 4)
    for i in hard_idx:
        cx, cy = polys[i].centroid.x, polys[i].centroid.y
        for _ in range(2):
            tx = cx + RNG.normal(0, 6)
            ty = cy + RNG.normal(0, 6)
            r = RNG.uniform(16, 22)
            yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            blob = (xx - tx) ** 2 + (yy - ty) ** 2 <= r ** 2
            img[blob] = (img[blob].astype(np.float32) * 0.35 + np.array([22, 60, 22]) * 0.65).astype(np.uint8)

    np.save(f"{out_dir}/ortho_image.npy", img)
    np.save(f"{out_dir}/ortho_mask_truth.npy", mask)

    gdfs, owners, landuse = build_government_layers(polys)
    for name, gdf in gdfs.items():
        gdf.to_file(f"{out_dir}/{name}.geojson", driver="GeoJSON")

    # save drone-truth footprints in geo coords too (what a perfect
    # extractor would produce) for reference/evaluation only
    truth_geo = gpd.GeoDataFrame(
        [{"geometry": _px_to_geo(p), "_truth_id": i} for i, p in enumerate(polys)],
        geometry="geometry", crs="EPSG:32644",
    )
    truth_geo.to_file(f"{out_dir}/drone_truth_footprints.geojson", driver="GeoJSON")

    meta = {"n_buildings": len(polys), "gsd": GSD, "img_size": IMG_SIZE}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated {len(polys)} synthetic buildings.")
    for name, gdf in gdfs.items():
        print(f"  {name:10s}: {len(gdf)} records")
    return img, mask, gdfs


if __name__ == "__main__":
    generate_all()
