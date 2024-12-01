import gradio as gr

from gradio_toggle import Toggle
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging

import imageio
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
import tempfile
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from torchao.quantization import quantize_, int8_weight_only

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257


def load_vae(vae_dir, int8=False):
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    if torch.cuda.is_available():
        vae = vae.cuda()
    if int8:
        print("vae - quantization = true")
        quantize_(vae, int8_weight_only())
    torch.cuda.empty_cache()
    return vae.to(torch.bfloat16)


def load_unet(unet_dir, int8=False):
    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
    unet_config_path = unet_dir / "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    if int8:
        print("unet - quantization = true")
        quantize_(transformer, int8_weight_only())
    torch.cuda.empty_cache()
    return transformer


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)


def load_image_to_tensor_with_resize_and_crop(image_path, target_height=512, target_width=768):
    image = Image.open(image_path).convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


# Generate output video name
def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )


def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main(
        img2vid_image="",
        prompt="",
        txt2vid_analytics_toggle=False,
        negative_prompt="",
        frame_rate=25,
        seed=0,
        num_inference_steps=30,
        guidance_scale=3,
        height=512,
        width=768,
        num_frames=121,
        progress=gr.Progress(),
    ):
    
    logger = logging.get_logger(__name__)
    
    # args = parser.parse_args()
    args = {
                "ckpt_dir": "Lightricks/LTX-Video",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": 0,
                "output_path": os.path.join(tempfile.gettempdir(), "gradio"),
                "num_images_per_prompt": 1,
                "input_image_path": img2vid_image,
                "input_video_path": "",
                "bfloat16": True,
                "disable_load_needed_only": False
            }
    logger.warning(f"Running generation with arguments: {args}")

    seed_everething(args['seed'])

    output_dir = (
        Path(args['output_path'])
        if args['output_path']
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    if args['input_image_path']:
        media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            args['input_image_path'], args['height'], args['width']
        )
    else:
        media_items_prepad = None

    height = args['height'] if args['height'] else media_items_prepad.shape[-2]
    width = args['width'] if args['width'] else media_items_prepad.shape[-1]
    num_frames = args['num_frames']

    if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
        logger.warning(
            f"Input resolution or number of frames {height}x{width}x{num_frames} is too big, it is suggested to use the resolution below {MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES}."
        )

    # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(height, width, height_padded, width_padded)

    logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

    if media_items_prepad is not None:
        media_items = F.pad(
            media_items_prepad, padding, mode="constant", value=-1
        )  # -1 is the value for padding since the image is normalized to -1, 1
    else:
        media_items = None

    # Paths for the separate mode directories
    ckpt_dir = Path(args['ckpt_dir'])
    unet_dir = ckpt_dir / "unet"
    vae_dir = ckpt_dir / "vae"
    scheduler_dir = ckpt_dir / "scheduler"

    # Load models
    vae = load_vae(vae_dir, txt2vid_analytics_toggle)
    unet = load_unet(unet_dir, txt2vid_analytics_toggle)
    scheduler = load_scheduler(scheduler_dir)
    patchifier = SymmetricPatchifier(patch_size=1)
    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
    ).to(torch.bfloat16)

    # if torch.cuda.is_available():
    #     text_encoder = text_encoder.to("cuda")

    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )

    if args['bfloat16'] and unet.dtype != torch.bfloat16:
        unet = unet.to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": unet,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    if torch.cuda.is_available() and args['disable_load_needed_only']:
        pipeline = pipeline.to("cuda")

    # Prepare input for the pipeline
    sample = {
        "prompt": args['prompt'],
        "prompt_attention_mask": None,
        "negative_prompt": args['negative_prompt'],
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
    }

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(args['seed'])

    images = pipeline(
        num_inference_steps=args['num_inference_steps'],
        num_images_per_prompt=args['num_images_per_prompt'],
        guidance_scale=args['guidance_scale'],
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=args['frame_rate'],
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(
            ConditioningMethod.FIRST_FRAME
            if media_items is not None
            else ConditioningMethod.UNCONDITIONAL
        ),
        mixed_precision=not args['bfloat16'],
        load_needed_only=not args['disable_load_needed_only']
    ).images

    # Crop the padded images to the desired resolution and number of frames
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = args['frame_rate']
        height, width = video_np.shape[1:3]
        # In case a single image is generated
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=args['prompt'],
                seed=args['seed'],
                resolution=(height, width, num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            if args['input_image_path']:
                base_filename = f"img_to_vid_{i}"
            else:
                base_filename = f"text_to_vid_{i}"
            output_filename = get_unique_filename(
                base_filename,
                ".mp4",
                prompt=args['prompt'],
                seed=args['seed'],
                resolution=(height, width, num_frames),
                dir=output_dir,
            )

            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

            # Write condition image
            if args['input_image_path']:
                reference_image = (
                    (
                        media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy()
                        + 1.0
                    )
                    / 2.0
                    * 255
                )
                imageio.imwrite(
                    get_unique_filename(
                        base_filename,
                        ".png",
                        prompt=args['prompt'],
                        seed=args['seed'],
                        resolution=(height, width, num_frames),
                        dir=output_dir,
                        endswith="_condition",
                    ),
                    reference_image.astype(np.uint8),
                )
        logger.warning(f"Output saved {output_filename}")
        return output_filename


preset_options = [
    {"label": "1216x704, 41 frames", "width": 1216, "height": 704, "num_frames": 41},
    {"label": "1088x704, 49 frames", "width": 1088, "height": 704, "num_frames": 49},
    {"label": "1056x640, 57 frames", "width": 1056, "height": 640, "num_frames": 57},
    {"label": "992x608, 65 frames", "width": 992, "height": 608, "num_frames": 65},
    {"label": "896x608, 73 frames", "width": 896, "height": 608, "num_frames": 73},
    {"label": "896x544, 81 frames", "width": 896, "height": 544, "num_frames": 81},
    {"label": "832x544, 89 frames", "width": 832, "height": 544, "num_frames": 89},
    {"label": "800x512, 97 frames", "width": 800, "height": 512, "num_frames": 97},
    {"label": "768x512, 97 frames", "width": 768, "height": 512, "num_frames": 97},
    {"label": "800x480, 105 frames", "width": 800, "height": 480, "num_frames": 105},
    {"label": "736x480, 113 frames", "width": 736, "height": 480, "num_frames": 113},
    {"label": "704x480, 121 frames", "width": 704, "height": 480, "num_frames": 121},
    {"label": "704x448, 129 frames", "width": 704, "height": 448, "num_frames": 129},
    {"label": "672x448, 137 frames", "width": 672, "height": 448, "num_frames": 137},
    {"label": "640x416, 153 frames", "width": 640, "height": 416, "num_frames": 153},
    {"label": "672x384, 161 frames", "width": 672, "height": 384, "num_frames": 161},
    {"label": "640x384, 169 frames", "width": 640, "height": 384, "num_frames": 169},
    {"label": "608x384, 177 frames", "width": 608, "height": 384, "num_frames": 177},
    {"label": "576x384, 185 frames", "width": 576, "height": 384, "num_frames": 185},
    {"label": "608x352, 193 frames", "width": 608, "height": 352, "num_frames": 193},
    {"label": "576x352, 201 frames", "width": 576, "height": 352, "num_frames": 201},
    {"label": "544x352, 209 frames", "width": 544, "height": 352, "num_frames": 209},
    {"label": "512x352, 225 frames", "width": 512, "height": 352, "num_frames": 225},
    {"label": "512x352, 233 frames", "width": 512, "height": 352, "num_frames": 233},
    {"label": "544x320, 241 frames", "width": 544, "height": 320, "num_frames": 241},
    {"label": "512x320, 249 frames", "width": 512, "height": 320, "num_frames": 249},
    {"label": "512x320, 257 frames", "width": 512, "height": 320, "num_frames": 257},
]

def create_advanced_options():
    with gr.Accordion("Advanced Options (Optional)", open=False):
        seed = gr.Slider(label="Seed", minimum=0, maximum=1000000, step=1, value=0)
        inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=30)
        guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, step=0.1, value=3.0)

        height_slider = gr.Slider(
            label="Height",
            minimum=256,
            maximum=1024,
            step=64,
            value=512,
            visible=False,
        )
        width_slider = gr.Slider(
            label="Width",
            minimum=256,
            maximum=1024,
            step=64,
            value=768,
            visible=False,
        )
        num_frames_slider = gr.Slider(
            label="Number of Frames",
            minimum=1,
            maximum=200,
            step=1,
            value=97,
            visible=False,
        )

        return [
            seed,
            inference_steps,
            guidance_scale,
            height_slider,
            width_slider,
            num_frames_slider,
        ]
    
def preset_changed(preset):
    if preset != "Custom":
        selected = next(item for item in preset_options if item["label"] == preset)
        return (
            selected["height"],
            selected["width"],
            selected["num_frames"],
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    else:
        return (
            None,
            None,
            None,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )


css="""
#col-container {
    margin: 0 auto;
    max-width: 1220px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            img2vid_image = gr.Image(
                type="filepath",
                label="Upload Input Image",
                elem_id="image_upload",
            )

            txt2vid_prompt = gr.Textbox(
                label="Enter Your Prompt",
                placeholder="Describe the video you want to generate (minimum 50 characters)...",
                value="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage.",
                lines=5,
            )
            
            txt2vid_analytics_toggle = Toggle(
                label="torchao.quantization.",
                value=False,
                interactive=True,
            )

            txt2vid_negative_prompt = gr.Textbox(
                label="Enter Negative Prompt",
                placeholder="Describe what you don't want in the video...",
                value="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                lines=2,
            )

            txt2vid_preset = gr.Dropdown(
                choices=[p["label"] for p in preset_options],
                value="768x512, 97 frames",
                label="Choose Resolution Preset",
            )

            txt2vid_frame_rate = gr.Slider(
                label="Frame Rate",
                minimum=21,
                maximum=30,
                step=1,
                value=25,
            )

            txt2vid_advanced = create_advanced_options()
            
            txt2vid_generate = gr.Button(
                "Generate Video",
                variant="primary",
                size="lg",
            )

        with gr.Column():
            txt2vid_output = gr.Video(label="Generated Output")

    with gr.Row():
        gr.Examples(
            examples=[
                [
                    "A young woman in a traditional Mongolian dress is peeking through a sheer white curtain, her face showing a mix of curiosity and apprehension. The woman has long black hair styled in two braids, adorned with white beads, and her eyes are wide with a hint of surprise. Her dress is a vibrant blue with intricate gold embroidery, and she wears a matching headband with a similar design. The background is a simple white curtain, which creates a sense of mystery and intrigue.ith long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair’s face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage",
                    "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                ],
                [
                    "A young man with blond hair wearing a yellow jacket stands in a forest and looks around. He has light skin and his hair is styled with a middle part. He looks to the left and then to the right, his gaze lingering in each direction. The camera angle is low, looking up at the man, and remains stationary throughout the video. The background is slightly out of focus, with green trees and the sun shining brightly behind the man. The lighting is natural and warm, with the sun creating a lens flare that moves across the man’s face. The scene is captured in real-life footage.",
                    "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                ],
                [
                    "A cyclist races along a winding mountain road. Clad in aerodynamic gear, he pedals intensely, sweat glistening on his brow. The camera alternates between close-ups of his determined expression and wide shots of the breathtaking landscape. Pine trees blur past, and the sky is a crisp blue. The scene is invigorating and competitive.",
                    "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                ],
            ],
            inputs=[txt2vid_prompt, txt2vid_negative_prompt, txt2vid_output],
            label="Example Text-to-Video Generations",
        )

    txt2vid_preset.change(fn=preset_changed, inputs=[txt2vid_preset], outputs=txt2vid_advanced[3:])

    txt2vid_generate.click(
        fn=main,
        inputs=[
            img2vid_image,
            txt2vid_prompt,
            txt2vid_analytics_toggle,
            txt2vid_negative_prompt,
            txt2vid_frame_rate,
            *txt2vid_advanced,
        ],
        outputs=txt2vid_output,
        concurrency_limit=1,
        concurrency_id="generate_video",
        queue=True,
    )
demo.launch()
