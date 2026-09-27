import torch
import torchvision
from io import BytesIO

from fastapi import FastAPI
from fastapi.responses import Response

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams

from pathlib import Path


app = FastAPI(title="3DGS Renderer API")

MODEL_PATH = str(Path(__file__).resolve().parent)

gaussians = None
scene = None
pipeline = None
background = None


@app.on_event("startup")
def load_model():
    global gaussians, scene, pipeline, background

    print("Loading 3DGS model...")

    # Build the same argument configuration used by GraphDeco
    from argparse import ArgumentParser

    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    args = parser.parse_args([
    "--model_path", MODEL_PATH,
    "--source_path", MODEL_PATH,
    "--depths", "",
    "--resolution", "2",
    "--sh_degree", "3",
    "--data_device", "cuda"
    ])

    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)

    # Load trained Gaussians and COLMAP cameras
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=3000,
        shuffle=False
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(
        bg_color,
        dtype=torch.float32,
        device="cuda"
    )

    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    print(f"Training cameras: {len(scene.getTrainCameras())}")
    print(f"Device: {gaussians.get_xyz.device}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gaussians": gaussians.get_xyz.shape[0],
        "device": str(gaussians.get_xyz.device),
        "cameras": len(scene.getTrainCameras()),
    }


@app.post("/render")
def render_scene():

    # Use the first COLMAP training camera
    view = scene.getTrainCameras()[0]

    with torch.no_grad():
        result = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=False,
            separate_sh=False
        )

        image = result["render"]

    # Convert tensor to PNG in memory
    buffer = BytesIO()
    torchvision.utils.save_image(image, buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )