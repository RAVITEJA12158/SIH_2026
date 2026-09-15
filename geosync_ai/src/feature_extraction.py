"""
GeoSync AI - Adaptive AI Feature Extraction Pipeline
------------------------------------------------------
Implements the escalation logic from the project doc, section 12:

    SegFormer -> if extraction insufficient -> Specialized Model
              -> if still unresolved       -> SAM (fallback)

Rather than running every model on the whole scene, the scene is
divided into a coarse grid; each cell's SegFormer confidence margin
decides whether it needs escalation. This is the "adaptive" part --
most of the scene is resolved cheaply by the primary model, and only
genuinely hard regions pay the cost of the heavier tiers.
"""
import numpy as np
from rasterio.features import shapes as raster_shapes
from shapely.geometry import shape, Polygon
import geopandas as gpd

GRID = 16            # px, coarse-quality grid cell size (fine enough that
                     # a single occluded building doesn't get averaged away
                     # by confident neighboring background)
CONF_OK = 0.5        # min effective quality (confidence margin, penalized
                     # for occlusion) above which SegFormer's own output is
                     # trusted as-is. Per doc section 22, thresholds like
                     # this should be recalibrated against real pilot data.
CONF_SPECIALIZED_OK = 0.55


def _occlusion_mask(img):
    """Cheap image-quality flag independent of the model's own (often
    over-confident) softmax output -- flags dark/green canopy-shadow
    pixels the way operational remote-sensing pipelines flag cloud/shadow
    regions before trusting any downstream model output on them."""
    r, g, b = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    brightness = (r + g + b) / 3.0
    greenish = (g > r + 5) & (g > b + 5)
    dark = brightness < 95
    return (greenish & dark)


def _grid_confidence(prob, occl_mask, grid=GRID):
    """A cell needs escalation if EITHER the model's own confidence margin
    is low OR the underlying imagery in that cell is significantly
    occluded/shadowed (a data-quality issue no classifier confidence
    score would reliably reveal on its own)."""
    H, W = prob.shape
    margin = np.abs(prob - 0.5) * 2  # 0 (ambiguous) .. 1 (confident)
    cells = []
    for y in range(0, H, grid):
        for x in range(0, W, grid):
            m = margin[y:y + grid, x:x + grid].mean()
            occl_frac = occl_mask[y:y + grid, x:x + grid].mean()
            effective_quality = min(m, 1.0 - occl_frac) if occl_frac > 0.3 else m
            cells.append((y, x, effective_quality))
    return cells


def run_adaptive_extraction(segformer_model, specialized_model, img,
                             segformer_predict_fn, device="cpu", verbose=True):
    """Returns final_prob, final_pred, method_map (per-pixel: 0=segformer,
    1=specialized, 2=sam_fallback) over the full scene."""
    from specialized_model import predict_patch as specialized_predict
    from sam_fallback import segment_region

    prob0, pred0 = segformer_predict_fn(segformer_model, img, device=device)
    H, W = prob0.shape
    final_prob = prob0.copy()
    final_pred = pred0.copy()
    method_map = np.zeros((H, W), dtype=np.uint8)

    occl_mask = _occlusion_mask(img)
    cells = _grid_confidence(prob0, occl_mask)
    n_specialized = n_sam = 0

    for (y, x, margin) in cells:
        if margin >= CONF_OK:
            continue  # SegFormer output trusted as-is

        y1, x1 = min(y + GRID, H), min(x + GRID, W)
        # pad a little context around the cell for the specialized model
        py0, px0 = max(0, y - 8), max(0, x - 8)
        py1, px1 = min(H, y1 + 8), min(W, x1 + 8)
        patch = img[py0:py1, px0:px1]

        sp_prob_patch = specialized_predict(specialized_model, patch, device=device)
        sub_y0, sub_x0 = y - py0, x - px0
        sp_prob_cell = sp_prob_patch[sub_y0:sub_y0 + (y1 - y), sub_x0:sub_x0 + (x1 - x)]
        sp_margin = np.abs(sp_prob_cell - 0.5).mean() * 2

        if sp_margin >= CONF_SPECIALIZED_OK:
            final_prob[y:y1, x:x1] = sp_prob_cell
            final_pred[y:y1, x:x1] = (sp_prob_cell > 0.5).astype(np.uint8)
            method_map[y:y1, x:x1] = 1
            n_specialized += 1
        else:
            prior = prob0[py0:py1, px0:px1]
            sam_mask_patch, sam_prob_patch = segment_region(patch, seed_prob=prior)
            final_prob[y:y1, x:x1] = sam_prob_patch[sub_y0:sub_y0 + (y1 - y), sub_x0:sub_x0 + (x1 - x)]
            final_pred[y:y1, x:x1] = sam_mask_patch[sub_y0:sub_y0 + (y1 - y), sub_x0:sub_x0 + (x1 - x)]
            method_map[y:y1, x:x1] = 2
            n_sam += 1

    if verbose:
        n_cells = len(cells)
        print(f"  Grid cells: {n_cells} total | "
              f"{n_cells - n_specialized - n_sam} SegFormer-only | "
              f"{n_specialized} escalated to Specialized | "
              f"{n_sam} escalated to SAM-fallback")

    return final_prob, final_pred, method_map


def mask_to_polygons(pred_mask, prob_map, method_map, px_to_geo_fn, min_area_px=70):
    """Vectorize the final binary prediction into polygons, attaching the
    mean confidence and the dominant extraction method used per polygon."""
    from scipy.ndimage import binary_opening, binary_closing
    pred_mask = binary_opening(pred_mask, structure=np.ones((3, 3)))
    pred_mask = binary_closing(pred_mask, structure=np.ones((3, 3))).astype(np.uint8)
    features = []
    for geom, val in raster_shapes(pred_mask.astype(np.int32), mask=pred_mask.astype(bool)):
        if val != 1:
            continue
        poly = shape(geom)
        if poly.area < min_area_px:
            continue
        # sample confidence / method within the polygon's bounding box
        minx, miny, maxx, maxy = [int(v) for v in poly.bounds]
        maxx, maxy = min(maxx + 1, prob_map.shape[1]), min(maxy + 1, prob_map.shape[0])
        sub_prob = prob_map[miny:maxy, minx:maxx]
        sub_method = method_map[miny:maxy, minx:maxx]
        conf = float(sub_prob.mean()) if sub_prob.size else 0.5
        dominant_method = int(np.bincount(sub_method.ravel(), minlength=3).argmax()) if sub_method.size else 0
        method_name = {0: "segformer", 1: "specialized", 2: "sam_fallback"}[dominant_method]

        geo_poly = px_to_geo_fn(poly)
        features.append({
            "geometry": geo_poly,
            "extraction_confidence": round(conf, 3),
            "extraction_method": method_name,
            "source": "drone_ai_extraction",
        })
    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:32644")
    return gdf


if __name__ == "__main__":
    import torch
    import sys
    sys.path.insert(0, "/home/claude/geosync_ai/src")
    from segformer_model import make_segformer, predict_full_scene
    from specialized_model import MiniUNet
    from synthetic_data import _px_to_geo

    img = np.load("/home/claude/geosync_ai/data/ortho_image.npy")

    sf = make_segformer()
    sf.load_state_dict(torch.load("/home/claude/geosync_ai/models/segformer.pt"))
    sf.eval()

    spec = MiniUNet()
    spec.load_state_dict(torch.load("/home/claude/geosync_ai/models/specialized_model.pt"))
    spec.eval()

    final_prob, final_pred, method_map = run_adaptive_extraction(
        sf, spec, img, predict_full_scene)

    gdf = mask_to_polygons(final_pred, final_prob, method_map, _px_to_geo)
    gdf.to_file("/home/claude/geosync_ai/data/extracted_features.geojson", driver="GeoJSON")
    print(f"Extracted {len(gdf)} candidate footprints -> data/extracted_features.geojson")
    print(gdf["extraction_method"].value_counts())
