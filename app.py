import base64
import logging
import os
import time

import cv2
import gradio as gr
import numpy as np
import requests
from gradio.themes.utils import sizes

# LOGGING
logger = logging.getLogger("TRYON")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

# IMAGE ASSETS
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# API CONFIG
FASHN_ENDPOINT_URL = os.environ.get("FASHN_ENDPOINT_URL", "https://api.fashn.ai/v1")
FASHN_API_KEY = os.environ.get("FASHN_API_KEY")
assert FASHN_ENDPOINT_URL, "Please set the FASHN_ENDPOINT_URL environment variable"
assert FASHN_API_KEY, "Please set the FASHN_API_KEY environment variable"

# ----------------- HELPER FUNCTIONS ----------------- #

CATEGORY_API_MAPPING = {"Top": "tops", "Bottom": "bottoms", "Full-body": "one-pieces"}


def opencv_load_image_from_http(url: str) -> np.ndarray:
    """Loads an image from a given URL using HTTP GET."""
    with requests.get(url) as response:
        response.raise_for_status()
        image_data = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image


def encode_img_to_base64(img: np.array) -> str:
    """Encodes an image as a JPEG in Base64 format."""
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = base64.b64encode(img).decode("utf-8")
    img = f"data:image/jpeg;base64,{img}"
    return img


def parse_checkboxes(checkboxes):
    checkboxes = [checkbox.lower().replace(" ", "_") for checkbox in checkboxes]
    checkboxes = {checkbox: True for checkbox in checkboxes}
    return checkboxes


def make_api_request(session, url, headers, data=None, method="GET", max_retries=3, timeout=60):
    for attempt in range(max_retries):
        try:
            if method.upper() == "GET":
                response = session.get(url, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                response = session.post(url, headers=headers, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # If it's the last attempt
                raise Exception(f"API call failed after {max_retries} attempts: {str(e)}") from e
            print(f"Attempt {attempt + 1} failed. Retrying...")
            time.sleep(2)  # Wait for 2 seconds before retrying


# ----------------- CORE FUNCTION ----------------- #


def get_tryon_result(
    model_image,
    garment_image,
    photo_type,
    category,
    nsfw_filter,
    cover_feet,
    adjust_hands,
    restore_background,
    restore_clothes,
    guidance_scale,
    timesteps,
    seed,
    num_samples,
):
    logger.info("Starting new try-on request...")

    # preprocessing: convert to RGB, resize, encode to base64
    model_image, garment_image = map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [model_image, garment_image])
    model_image, garment_image = map(encode_img_to_base64, [model_image, garment_image])

    # prepare data for API request
    category = CATEGORY_API_MAPPING[category]
    data = {
        "model_image": model_image,
        "garment_image": garment_image,
        "flat_lay": photo_type == "Flat Lay",
        "category": category,
        "nsfw_filter": nsfw_filter,
        "cover_feet": cover_feet,
        "adjust_hands": adjust_hands,
        "restore_background": restore_background,
        "restore_clothes": restore_clothes,
        "guidance_scale": guidance_scale,
        "timesteps": timesteps,
        "seed": seed,
        "num_samples": num_samples,
    }

    # make API request
    session = requests.Session()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {FASHN_API_KEY}"}

    try:
        response_data = make_api_request(
            session, f"{FASHN_ENDPOINT_URL}/run", headers=headers, data=data, method="POST"
        )
        pred_id = response_data.get("id")
        logger.info(f"Prediction ID: {pred_id}")
    except Exception as e:
        raise gr.Error(f"Status check failed: {str(e)}")

    # poll the status of the prediction
    start_time = time.time()
    while True:
        if time.time() - start_time > 180:  # 3 minutes timeout
            raise gr.Error("Maximum polling time exceeded.")

        try:
            status_data = make_api_request(
                session, f"{FASHN_ENDPOINT_URL}/status/{pred_id}", headers=headers, method="GET"
            )
        except Exception as e:
            raise gr.Error(f"Status check failed: {str(e)}")

        if status_data["status"] == "completed":
            logger.info("Prediction completed.")
            break
        elif status_data["status"] not in ["starting", "in_queue", "processing"]:
            raise gr.Error(f"Prediction failed with id {pred_id}: {status_data.get('error')}")

        logger.info(f"Prediction status: {status_data['status']}")
        time.sleep(3)

    # get the result images
    result_imgs = []
    for output_url in status_data["output"]:
        result_img = opencv_load_image_from_http(output_url)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_imgs.append(result_img)

    return result_imgs


# ----------------- GRADIO UI ----------------- #


with open("banner.html", "r") as file:
    banner = file.read()
with open("tips.html", "r") as file:
    tips = file.read()
with open("footer.html", "r") as file:
    footer = file.read()

CUSTOM_CSS = """
.image-container  img {
    max-width: 384px;
    max-height: 576px;
    margin: 0 auto;
    border-radius: 0px;
.gradio-container {background-color: #fafafa}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome(radius_size=sizes.radius_md)) as demo:
    gr.HTML(banner)
    gr.HTML(tips)
    with gr.Row():
        with gr.Column():
            model_image = gr.Image(label="Model Image", type="numpy", format="png")

            with gr.Accordion("Model Image Controls", open=False):
                cover_feet = gr.Checkbox(label="Cover Feet", value=False)
                adjust_hands = gr.Checkbox(label="Adjust Hands", value=False)
                restore_background = gr.Checkbox(label="Restore Background", value=False)
                restore_clothes = gr.Checkbox(label="Restore Clothes", value=False)
                nsfw_filter = gr.Checkbox(label="NSFW Filter", value=True)

            example_model = gr.Examples(
                inputs=model_image,
                examples_per_page=10,
                examples=[
                    os.path.join(ASSETS_DIR, "models", img) for img in os.listdir(os.path.join(ASSETS_DIR, "models"))
                ],
            )
        with gr.Column():
            garment_image = gr.Image(label="Garment Image", type="numpy", format="png")
            photo_type = gr.Radio(choices=["Flat Lay", "Model"], label="Select Photo Type")
            category = gr.Radio(choices=["Top", "Bottom", "Full-body"], label="Select Category")

            example_garment = gr.Examples(
                inputs=garment_image,
                examples_per_page=10,
                examples=[
                    os.path.join(ASSETS_DIR, "garments", img)
                    for img in os.listdir(os.path.join(ASSETS_DIR, "garments"))
                ],
            )

        with gr.Column():
            result_gallery = gr.Gallery(label="Try-on Results", show_label=True, elem_id="gallery")
            run_button = gr.Button("Run")
            with gr.Accordion("Sampling Controls", open=False):
                guidance_scale = gr.Slider(minimum=1.5, maximum=5, value=2.0, step=0.1, label="Guidance Scale")
                timesteps = gr.Slider(minimum=10, maximum=50, step=1, value=50, label="Timesteps")
                seed = gr.Number(label="Seed", value=42, precision=0)
                num_samples = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Number of Samples")

    run_button.click(
        fn=get_tryon_result,
        inputs=[
            model_image,
            garment_image,
            photo_type,
            category,
            nsfw_filter,
            cover_feet,
            adjust_hands,
            restore_background,
            restore_clothes,
            guidance_scale,
            timesteps,
            seed,
            num_samples,
        ],
        outputs=[result_gallery],
    )

    gr.HTML(footer)


if __name__ == "__main__":
    demo.launch()
