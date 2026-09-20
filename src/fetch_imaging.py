"""Phase D: imaging branch.

Pulls H&E tile crops for the TCGA TNBC cohort directly from the GDC slide
image tile server (no multi-GB .svs download needed - GDC serves individual
DeepZoom-style JPEG/PNG tiles per slide), picks the tiles with the most
actual tissue in them, and embeds each with Phikon (Owkin), an open
(non-gated) pathology foundation model pretrained on TCGA histology images
via self-supervised learning - a fitting match since our slides are TCGA
slides too.

Output: data/tcga_imaging_embeddings.csv, one 768-dim embedding per sample
(mean-pooled over each sample's top tissue tiles), plus a handful of the
actual tiles saved under data/tiles/ for provenance/inspection.
"""
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

GDC_API = "https://api.gdc.cancer.gov"
TILE_LEVEL = 13          # empirically: real cellular detail, manageable tile count
COARSE_GRID = 4          # scan a 4x4 grid of candidate tiles per slide
TOP_K_TILES = 3          # keep the 3 most tissue-dense tiles per slide
TILES_TO_SAVE = 2        # save this many example tiles per slide to disk
TIMEOUT = 30


def get_diagnostic_slide_ids(case_ids):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,cases.submitter_id",
        "size": "500",
    }
    r = requests.get(f"{GDC_API}/files", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    mapping = {}
    for hit in r.json()["data"]["hits"]:
        case = hit["cases"][0]["submitter_id"]
        mapping.setdefault(case, hit["file_id"])  # first diagnostic slide per case
    return mapping


def fetch_tile(file_id, level, x, y):
    url = f"{GDC_API}/tile/{file_id}"
    r = requests.get(url, params={"level": level, "x": x, "y": y}, timeout=TIMEOUT)
    if r.status_code != 200 or len(r.content) < 200:
        return None
    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    return img


def tissue_fraction(img):
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return float((arr < 225).mean())  # non-white/background pixels


def pick_tissue_tiles(file_id):
    coords = [(x, y) for x in range(COARSE_GRID) for y in range(COARSE_GRID)]
    scored = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_tile, file_id, TILE_LEVEL, x, y): (x, y) for x, y in coords}
        for fut in as_completed(futures):
            x, y = futures[fut]
            img = fut.result()
            if img is None:
                continue
            scored.append((tissue_fraction(img), x, y, img))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:TOP_K_TILES]


def embed_tiles(images, processor, model):
    if not images:
        return None
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    cls_embeds = out.last_hidden_state[:, 0, :].numpy()  # CLS token per Phikon usage
    return cls_embeds.mean(axis=0)


if __name__ == "__main__":
    os.makedirs("data/tiles", exist_ok=True)

    sample_ids = pd.read_csv("data/tcga_tnbc.csv", index_col=0).index.tolist()
    case_ids = sorted({s[:12] for s in sample_ids})
    print(f"Looking up diagnostic slides for {len(case_ids)} TCGA TNBC cases...")
    slide_map = get_diagnostic_slide_ids(case_ids)
    print(f"  Found slides for {len(slide_map)}/{len(case_ids)} cases.")

    print("\nLoading Phikon (owkin/phikon) pathology foundation model...")
    processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    model = AutoModel.from_pretrained("owkin/phikon")
    model.eval()
    print("  Loaded.")

    records = {}
    for i, (case, file_id) in enumerate(sorted(slide_map.items()), 1):
        print(f"[{i}/{len(slide_map)}] {case} (slide {file_id})...", end=" ", flush=True)
        top_tiles = pick_tissue_tiles(file_id)
        if not top_tiles or top_tiles[0][0] < 0.02:
            print("no usable tissue tiles, skipped.")
            continue

        for j, (frac, x, y, img) in enumerate(top_tiles[:TILES_TO_SAVE]):
            out_dir = f"data/tiles/{case}"
            os.makedirs(out_dir, exist_ok=True)
            img.resize((256, 256)).save(f"{out_dir}/tile_{j}_lvl{TILE_LEVEL}_x{x}_y{y}.png")

        embed = embed_tiles([img for _, _, _, img in top_tiles], processor, model)
        records[case] = embed
        print(f"embedded from {len(top_tiles)} tiles (best tissue frac={top_tiles[0][0]:.2f}).")

    df = pd.DataFrame.from_dict(records, orient="index")
    df.columns = [f"phikon_{i}" for i in range(df.shape[1])]
    df.index.name = "case_id"
    df.to_csv("data/tcga_imaging_embeddings.csv")
    print(f"\nSaved data/tcga_imaging_embeddings.csv: {df.shape[0]} samples x {df.shape[1]} dims.")
