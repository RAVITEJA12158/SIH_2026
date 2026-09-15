"""
GeoSync AI - Final Fallback (3rd escalation tier): SAM stand-in
------------------------------------------------------------------
IMPORTANT / HONEST NOTE ON THIS PROTOTYPE:
This sandbox has no network route to huggingface.co or Meta's model
hosting, so the real Segment Anything Model checkpoint cannot be
downloaded here. To keep the ESCALATION LOGIC and PIPELINE STRUCTURE
faithful to the design (SegFormer -> Specialized -> SAM), this module
implements a classical, promptable region-growing segmenter that
plays SAM's architectural role: given a bounding-box/point "prompt"
(the coarse seed from the two prior tiers) it returns a refined mask
for that region, with NO learned weights of its own.

In a real deployment, swap `segment_region()` internals for an actual
`SamPredictor(...).predict(box=..., point_coords=...)` call against a
downloaded `facebook/sam-vit-*` checkpoint -- the call sites in
feature_extraction.py do not need to change.
"""
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.filters import sobel
from skimage.color import rgb2gray


def segment_region(patch_uint8, seed_prob=None, prompt_threshold=0.3):
    """Prompt-style segmentation for one unresolved region.

    patch_uint8: (h,w,3) image crop
    seed_prob:   (h,w) float prior confidence from earlier tiers, used as
                 the "prompt" (like SAM's point/box prompt) telling the
                 segmenter roughly where the object of interest is.
    """
    gray = rgb2gray(patch_uint8)
    edges = sobel(gray)
    segments = felzenszwalb(patch_uint8, scale=60, sigma=0.6, min_size=25)

    if seed_prob is None:
        seed_prob = np.full(patch_uint8.shape[:2], 0.5, dtype=np.float32)

    # "prompt": pick superpixels whose mean prior confidence clears the
    # threshold, i.e. the ones the earlier tiers were pointing at
    mask = np.zeros(patch_uint8.shape[:2], dtype=np.uint8)
    out_prob = np.zeros(patch_uint8.shape[:2], dtype=np.float32)
    for seg_id in np.unique(segments):
        sel = segments == seg_id
        mean_prior = seed_prob[sel].mean()
        # edge-density penalty: superpixels that are mostly a soft edge
        # (shadow boundary / canopy fringe) get down-weighted
        edge_density = edges[sel].mean()
        conf = float(np.clip(mean_prior - 0.3 * edge_density, 0, 1))
        out_prob[sel] = conf
        if conf >= prompt_threshold:
            mask[sel] = 1
    return mask, out_prob


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    patch = rng.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
    seed = np.full((64, 64), 0.4, dtype=np.float32)
    seed[20:44, 20:44] = 0.7
    m, p = segment_region(patch, seed)
    print("SAM-fallback smoke test -> mask coverage:", m.mean())
