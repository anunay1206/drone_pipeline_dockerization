import os
import glob
import shutil
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.enums import Resampling

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.train import setup_cfg
from detectree2.models.predict import predict_on_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns

from detectron2.engine import DefaultPredictor
import torch



#  DOWNSAMPLE Image

def downsample_image(input_path, output_path, scale=0.3):
    with rasterio.open(input_path) as src:
        new_width = int(src.width * scale)
        new_height = int(src.height * scale)

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        profile = src.profile
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": transform
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    return output_path


def resolve_ortho_path(ortho_path):
    """Resolve input ortho path: if folder is passed, pick the first valid image."""
    if os.path.isdir(ortho_path):
        candidates = sorted(
            glob.glob(os.path.join(ortho_path, "*.tif"))
            + glob.glob(os.path.join(ortho_path, "*.tiff"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .tif / .tiff images found in directory: {ortho_path}"
            )
        if len(candidates) > 1:
            print(
                f"⚠️ Multiple orthomosaic images found in {ortho_path}; using the first one: {candidates[0]}"
            )
        return candidates[0]
    return ortho_path


# MODEL BUILDER (warm cache: build once, reuse across orthos)

def build_predictor(model_path, conf_threshold=0.85):
    """Build a detectron2 ``DefaultPredictor`` from detectree2 weights.

    Separated from the pipeline so the worker can construct it once per
    process and reuse it across every ortho (see app/workers/tasks.py).
    """
    cfg = setup_cfg(update_model=model_path)

    if torch.backends.mps.is_available():
        cfg.MODEL.DEVICE = "mps"
    elif torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 6

    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512

    return DefaultPredictor(cfg)


# MAIN PIPELINE

def run_detectree2_pipeline(
    ortho_path,
    predictor=None,
    output_dir="output",
    tile_size=10,
    buffer=10,
    iou_threshold=0.9,
    conf_threshold=0.85,
    model_path=None,
):

    os.makedirs(output_dir, exist_ok=True)

    tiles_dir = os.path.join(output_dir, "tiles")
    pred_dir = os.path.join(output_dir, "predictions")

    # CLEAN OLD RUNS
    if os.path.exists(tiles_dir):
        shutil.rmtree(tiles_dir)
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)

    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    
    #  STEP 0: DOWNSAMPLE
    ortho_path = resolve_ortho_path(ortho_path)

    print("Step 0: Downsampling image...")
    ortho_path = downsample_image(ortho_path, os.path.join(output_dir, "downsampled.tif"), scale=0.3)

  
    # STEP 1: TILING
    
    print("Step 1: Tiling orthomosaic...")

    tile_data(
        ortho_path,
        tiles_dir,
        buffer=buffer,
        tile_width=tile_size,
        tile_height=tile_size,
        threshold=0.05,
        nan_threshold=0.3,
        full_coverage=False
    )

    tiles = [f for f in os.listdir(tiles_dir) if f.endswith(".png") or f.endswith(".tif")]

    print("Tiles created:", len(tiles))
    print("Sample tiles:", tiles[:5])

    if len(tiles) == 0:
        raise RuntimeError("❌ No tiles generated.")

  # STEP 2: MODEL

    print("Step 2: Loading model...")
    if predictor is None:
        if model_path is None:
            raise ValueError(
                "run_detectree2_pipeline needs either a prebuilt `predictor` "
                "or a `model_path` to build one."
            )
        predictor = build_predictor(model_path, conf_threshold=conf_threshold)

  
    # STEP 3: PREDICTION
   
    print("Step 3: Running predictions...")
    predict_on_data(directory=tiles_dir, predictor=predictor)

    # MOVE predictions
    print("Fixing prediction output path...")
    default_pred_dir = os.path.join(os.getcwd(), "predictions")

    if os.path.exists(default_pred_dir):
        for f in os.listdir(default_pred_dir):
            src = os.path.join(default_pred_dir, f)
            dst = os.path.join(pred_dir, f)
            if os.path.isfile(src):
                shutil.move(src, dst)
    else:
        raise RuntimeError("❌ No predictions folder found!")

    # ================================
    # STEP 4: GEOJSON
    # ================================
    print("Step 4: Convert predictions to GeoJSON...")
    project_to_geojson(tiles_dir, pred_dir, pred_dir)

    geojson_files = glob.glob(os.path.join(pred_dir, "*.geojson"))
    print("GeoJSON files found:", len(geojson_files))

    if len(geojson_files) == 0:
        raise RuntimeError("❌ No GeoJSON generated.")

    
    # STEP 5: STITCH
    
    print("Step 5: Stitch crowns...")
    crowns = stitch_crowns(pred_dir, 1)

   
    # STEP 6: CLEANING (FIXED)
  
    print("Step 6: Cleaning crowns...")

    if isinstance(crowns, gpd.GeoSeries):
        crowns = gpd.GeoDataFrame(geometry=crowns)

    crs = crowns.crs

    crowns = crowns[crowns.is_valid]

    # simplify
    crowns["geometry"] = crowns.geometry.simplify(0.6)

    crowns = gpd.GeoDataFrame(crowns, geometry="geometry", crs=crs)

    # confidence
    if "Confidence_score" in crowns.columns:
        conf_col = "Confidence_score"
    else:
        conf_col = None

    if conf_col:
        crowns = crowns[crowns[conf_col] > conf_threshold]

    #  AREA FILTER 
    crowns["area"] = crowns.geometry.area
    crowns = crowns[(crowns["area"] > 4) & (crowns["area"] < 200)].copy()

    crowns = gpd.GeoDataFrame(crowns, geometry="geometry", crs=crs)

    # final cleaning
    crowns = clean_crowns(crowns, iou_threshold, conf_threshold)

    # SAVE
    geojson_path = os.path.join(output_dir, "tree_crowns.geojson")
    crowns.to_file(geojson_path, driver="GeoJSON")

    print(f"GeoJSON saved: {geojson_path}")

   
    # STEP 7: OVERLAY
    
    print("Step 7: Creating overlay image...")

    with rasterio.open(ortho_path) as src:
        img = src.read([1, 2, 3])
        img = img.transpose(1, 2, 0)

        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, extent=extent)

    crowns.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.8)

    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)

    overlay_path = os.path.join(output_dir, "overlay.png")
    plt.axis("off")
    plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Overlay saved: {overlay_path}")

    return geojson_path, overlay_path, ortho_path



# RUN

if __name__ == "__main__":

    ortho_image = "/Users/jayakrishna/Desktop/untitled folder/detectree2/s1_tree_rgb.tif"
    model_weights = "/Users/jayakrishna/Desktop/untitled folder/detectree2/250711_tropical_closed_canopy.pth"

    run_detectree2_pipeline(
        ortho_path=ortho_image,
        model_path=model_weights,
        output_dir="results"
    )