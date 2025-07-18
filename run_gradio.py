import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import torch.nn as nn
import os
import re

# 必要なパイプラインやモデルのクラスをインポート
from inference.manganinjia_pipeline import MangaNinjiaPipeline
from diffusers import (
    ControlNetModel,
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.refunet_2d_condition import RefUNet2DConditionModel
from src.point_network import PointNet
from src.annotator.lineart import BatchLineartDetector
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# --- 1. モデルと設定のパスを直接定義 ---
# 設定ファイル(YAML)の代わりに、ダウンロードしたモデルへのパスを直接指定します。
print("Initializing paths and configurations...")
pretrained_model_dir = './checkpoints/StableDiffusion'
image_encoder_path = './checkpoints/models/clip-vit-large-patch14'
controlnet_model_name_or_path = './checkpoints/models/control_v11p_sd15_lineart'
annotator_ckpts_path = './checkpoints/models/Annotators'
manga_reference_unet_path = './checkpoints/MangaNinjia/reference_unet.pth'
manga_main_model_path = './checkpoints/MangaNinjia/denoising_unet.pth'
manga_controlnet_model_path = './checkpoints/MangaNinjia/controlnet.pth'
point_net_path = './checkpoints/MangaNinjia/point_net.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. モデルの読み込み (infer.pyのロジックを移植) ---
print("Loading models...")

# Preprocessor (線画抽出器)
preprocessor = BatchLineartDetector(annotator_ckpts_path)

# モデルの入力チャンネル数を設定
in_channels_reference_unet = 4
in_channels_denoising_unet = 4
in_channels_controlnet = 4

# ベースとなるモデルのアーキテクチャを読み込むためのリポジトリ名
sd15_repo = "runwayml/stable-diffusion-v1-5"

# Stable Diffusionの単一モデルファイル(.safetensors)をディレクトリから探す
sd_model_file = None
for file in os.listdir(pretrained_model_dir):
    if file.endswith((".safetensors", ".ckpt")):
        sd_model_file = file
        break
if sd_model_file is None:
    raise FileNotFoundError(f"Stable Diffusion model file not found in: {pretrained_model_dir}")
sd_model_path = os.path.join(pretrained_model_dir, sd_model_file)
print(f"Loading Stable Diffusion base from: {sd_model_path}")

# from_single_file を使用してベースパイプラインを読み込む (safety_checker=None に変更)
base_pipe = StableDiffusionPipeline.from_single_file(
    sd_model_path,
    torch_dtype=torch.float16,
    safety_checker=None
)

# パイプラインからVAEとSchedulerを取得
base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
noise_scheduler = base_pipe.scheduler
vae = base_pipe.vae
vae.to(dtype=torch.float32)  # データ型不整合を避けるためfloat32にキャスト

# 各U-NetとControlNetのアーキテクチャをHugging Faceリポジトリから読み込む
denoising_unet = UNet2DConditionModel.from_pretrained(
    sd15_repo, subfolder="unet",
    in_channels=in_channels_denoising_unet,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True
)
reference_unet = RefUNet2DConditionModel.from_pretrained(
    sd15_repo, subfolder="unet",
    in_channels=in_channels_reference_unet,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True
)
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_name_or_path,
    in_channels=in_channels_controlnet,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True
)

# TokenizerとText/Image Encoderを読み込む
refnet_tokenizer = CLIPTokenizer.from_pretrained(image_encoder_path)
refnet_text_encoder = CLIPTextModel.from_pretrained(image_encoder_path)
refnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
controlnet_tokenizer = CLIPTokenizer.from_pretrained(image_encoder_path)
controlnet_text_encoder = CLIPTextModel.from_pretrained(image_encoder_path)
controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)

# PointNetを初期化
point_net = PointNet()

# --- 3. ダウンロードした学習済み重みをモデルにロード ---
print("Loading custom weights into models...")
controlnet.load_state_dict(
    torch.load(manga_controlnet_model_path, map_location="cpu"), strict=False
)
point_net.load_state_dict(
    torch.load(point_net_path, map_location="cpu"), strict=False
)
reference_unet.load_state_dict(
    torch.load(manga_reference_unet_path, map_location="cpu"), strict=False
)
denoising_unet.load_state_dict(
    torch.load(manga_main_model_path, map_location="cpu"), strict=False
)

# --- 4. MangaNinjiaパイプラインを初期化 ---
print("Initializing MangaNinjia Pipeline...")
pipe = MangaNinjiaPipeline(
    reference_unet=reference_unet,
    controlnet=controlnet,
    denoising_unet=denoising_unet,
    vae=vae,
    refnet_tokenizer=refnet_tokenizer,
    refnet_text_encoder=refnet_text_encoder,
    refnet_image_encoder=refnet_image_enc,
    controlnet_tokenizer=controlnet_tokenizer,
    controlnet_text_encoder=controlnet_text_encoder,
    controlnet_image_encoder=controlnet_image_enc,
    scheduler=noise_scheduler,
    point_net=point_net
)
pipe = pipe.to(torch.device(device))
preprocessor.to(device, dtype=torch.float32)
print("Pipeline is ready.")

# --- 5. Gradio UIのための関数群 ---
def string_to_np_array(coord_string):
    if not coord_string or '[]' in coord_string:
        return np.array([])
    coord_string = coord_string.strip('[]')
    coords = re.findall(r'\d+', coord_string)
    coords = list(map(int, coords))
    coord_array = np.array(coords).reshape(-1, 2)
    return coord_array

def infer_single(is_lineart, ref_image, target_image, output_coords_ref_str, output_coords_base_str, seed = -1, num_inference_steps=20, guidance_scale_ref = 9, guidance_scale_point =15 ):
    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()
    
    output_coords_ref = string_to_np_array(output_coords_ref_str)
    output_coords_base = string_to_np_array(output_coords_base_str)

    matrix1 = np.zeros((512, 512), dtype=np.uint8)
    matrix2 = np.zeros((512, 512), dtype=np.uint8)

    if len(output_coords_ref) == len(output_coords_base):
        for index, (coords_ref, coords_base) in enumerate(zip(output_coords_ref, output_coords_base)):
            y1, x1 = coords_ref
            y2, x2 = coords_base
            if 0 <= y1 < 512 and 0 <= x1 < 512:
                matrix1[y1, x1] = index + 1
            if 0 <= y2 < 512 and 0 <= x2 < 512:
                matrix2[y2, x2] = index + 1

    point_ref = torch.from_numpy(matrix1).unsqueeze(0).unsqueeze(0)
    point_main = torch.from_numpy(matrix2).unsqueeze(0).unsqueeze(0)

    pipe_out = pipe(
        is_lineart=is_lineart,
        ref1=ref_image,
        raw2=target_image,
        edit2=target_image,
        denosing_steps=num_inference_steps,
        processing_res=512,
        match_input_res=True,
        batch_size=1,
        show_progress_bar=True,
        guidance_scale_ref=guidance_scale_ref,
        guidance_scale_point=guidance_scale_point,
        preprocessor=preprocessor,
        generator=generator,
        point_ref=point_ref,
        point_main=point_main,
    )
    return pipe_out

def inference_single_image(ref_image, tar_image, ddim_steps, scale_ref, scale_point, seed, output_coords1, output_coords2, is_lineart):
    if seed == -1:
        seed = np.random.randint(10000)
    
    ref_pil = Image.fromarray(ref_image)
    tar_pil = Image.fromarray(tar_image)

    pipe_out = infer_single(
        is_lineart, ref_pil, tar_pil, 
        output_coords_ref_str=output_coords1, 
        output_coords_base_str=output_coords2, 
        seed=seed, 
        num_inference_steps=ddim_steps, 
        guidance_scale_ref=scale_ref, 
        guidance_scale_point=scale_point
    )
    return pipe_out

clicked_points_img1 = []
clicked_points_img2 = []
current_img_idx = 0
max_clicks = 14
point_size = 5
colors = [(255, 0, 0, 255), (0, 255, 0, 255)]

def draw_points(image_np, points, color):
    img_pil = Image.fromarray(image_np.astype('uint8')).convert("RGBA")
    draw = ImageDraw.Draw(img_pil)
    for point in points:
        x, y = point
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill=color, outline=color)
    return np.array(img_pil)

def get_select_coords(ref_img_np, base_img_np, evt: gr.SelectData):
    global clicked_points_img1, clicked_points_img2, current_img_idx

    is_ref_image_click = (current_img_idx % 2 == 0)
    click_coords = (evt.index[0], evt.index[1])

    if is_ref_image_click:
        if len(clicked_points_img1) < max_clicks:
            clicked_points_img1.append(click_coords)
    else:
         if len(clicked_points_img2) < max_clicks:
            clicked_points_img2.append(click_coords)

    display_ref = draw_points(ref_img_np, clicked_points_img1, colors[0])
    display_base = draw_points(base_img_np, clicked_points_img2, colors[1])
    
    current_img_idx += 1

    coords1_to_save = np.array([(p[1], p[0]) for p in clicked_points_img1])
    coords2_to_save = np.array([(p[1], p[0]) for p in clicked_points_img2])

    return display_ref, str(coords1_to_save.tolist()), display_base, str(coords2_to_save.tolist())


def undo_last_point(ref_img_np, base_img_np):
    global clicked_points_img1, clicked_points_img2, current_img_idx

    if current_img_idx > 0:
        current_img_idx -= 1
        if (current_img_idx % 2 == 0):
            if clicked_points_img1:
                clicked_points_img1.pop()
        else:
            if clicked_points_img2:
                clicked_points_img2.pop()

    display_ref = draw_points(ref_img_np, clicked_points_img1, colors[0])
    display_base = draw_points(base_img_np, clicked_points_img2, colors[1])
    
    coords1_to_save = np.array([(p[1], p[0]) for p in clicked_points_img1])
    coords2_to_save = np.array([(p[1], p[0]) for p in clicked_points_img2])

    return display_ref, str(coords1_to_save.tolist()), display_base, str(coords2_to_save.tolist())

def clear_all_points(ref_img_np, base_img_np):
    global clicked_points_img1, clicked_points_img2, current_img_idx
    clicked_points_img1 = []
    clicked_points_img2 = []
    current_img_idx = 0
    return ref_img_np, "[]", base_img_np, "[]"

# エラー修正済みの関数
def process_image(ref, base):
    if ref is None or base is None:
        raise gr.Error("Please upload both a reference and a target image.")
    
    ref_resized = cv2.resize(ref, (512, 512))
    base_resized = cv2.resize(base, (512, 512))
    
    global clicked_points_img1, clicked_points_img2, current_img_idx
    clicked_points_img1 = []
    clicked_points_img2 = []
    current_img_idx = 0
    
    # Gradioのoutputsリストの順番に合わせて値を返す
    return ref_resized, "[]", base_resized, "[]", ref_resized, base_resized


def run_local(ref, base, ddim_steps, scale_ref, scale_point, seed, output_coords1, output_coords2, is_lineart):
    if ref is None or base is None:
        raise gr.Error("Please process images first.")
    
    if len(string_to_np_array(output_coords1)) != len(string_to_np_array(output_coords2)):
        gr.Warning("The number of points on the reference and target images must be the same. The points will be ignored.")

    pipe_out = inference_single_image(
        ref, base, ddim_steps, scale_ref, scale_point, seed, 
        output_coords1, output_coords2, is_lineart
    )
    
    return [pipe_out.img_pil, pipe_out.to_save_dict['edge2_black']]


# --- 6. Gradioインターフェースの構築 ---
with gr.Blocks() as demo:
    gr.Markdown("# MangaNinjia: Line Art Colorization with Precise Reference Following")
    gr.Markdown("Official Repository: https://github.com/ali-vilab/MangaNinjia")
    
    with gr.Row():
        with gr.Column(scale=1):
            ref_orig = gr.Image(label="Reference Image (Original)", type="numpy")
            base_orig = gr.Image(label="Target Image (Original)", type="numpy")
            
            with gr.Row():
                process_button = gr.Button("1. Process Images to 512x512")
            
            with gr.Accordion("Advanced Options", open=True):
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                scale_ref = gr.Slider(label="Guidance of Reference", minimum=0, maximum=30.0, value=9.0, step=0.1)
                scale_point = gr.Slider(label="Guidance of Points", minimum=0, maximum=30.0, value=15.0, step=0.1)
                is_lineart = gr.Checkbox(label="Target Image is already a Line-art", value=False)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=-1, info="-1 for random")

            with gr.Accordion("Tutorial", open=False):
                 gr.Markdown("""
                1.  **Upload** a reference image and a target image.
                2.  Click **"1. Process Images to 512x512"**. This will resize the images and prepare the point-matching interface below.
                3.  **(Optional)** To define matching points:
                    * **Alternately** click on the reference and target images in the "Point Matching Interface" section.
                    * Start with a click on the **reference image**, then click the corresponding point on the **target image**.
                    * Repeat for more points.
                    * Use **"Undo Last Point"** to remove the last added point.
                    * Use **"Clear All Points"** to start over.
                4.  Adjust advanced options if needed.
                5.  Click **"3. Generate"** to produce the result.
                """)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Point Matching Interface (Optional)")
            with gr.Row():
                ref_display = gr.Image(label="Reference with Points", type="numpy", tool="select")
                base_display = gr.Image(label="Target with Points", type="numpy", tool="select")
            
            with gr.Row():
                undo_button = gr.Button("Undo Last Point")
                clear_button = gr.Button("Clear All Points")
            
            output_coords1 = gr.Textbox(visible=False)
            output_coords2 = gr.Textbox(visible=False)

            run_local_button = gr.Button("3. Generate", variant="primary")
            
            baseline_gallery = gr.Gallery(label='Output (Result | Extracted Line-art)', show_label=True, elem_id="gallery", columns=2, object_fit="contain", height="auto")

    # --- 7. Gradioイベントリスナーの設定 ---
    
    # 戻り値の順番を修正
    process_button.click(
        fn=process_image, 
        inputs=[ref_orig, base_orig], 
        outputs=[ref_display, output_coords1, base_display, output_coords2, ref_orig, base_orig]
    )

    ref_display.select(
        fn=get_select_coords, 
        inputs=[ref_orig, base_orig],
        outputs=[ref_display, output_coords1, base_display, output_coords2]
    )
    base_display.select(
        fn=get_select_coords, 
        inputs=[ref_orig, base_orig],
        outputs=[ref_display, output_coords1, base_display, output_coords2]
    )

    undo_button.click(
        fn=undo_last_point,
        inputs=[ref_orig, base_orig],
        outputs=[ref_display, output_coords1, base_display, output_coords2]
    )
    clear_button.click(
        fn=clear_all_points,
        inputs=[ref_orig, base_orig],
        outputs=[ref_display, output_coords1, base_display, output_coords2]
    )

    run_local_button.click(
        fn=run_local,
        inputs=[
            ref_orig, base_orig, ddim_steps, scale_ref, scale_point, seed,
            output_coords1, output_coords2, is_lineart
        ],
        outputs=[baseline_gallery]
    )

    gr.Examples(
        examples=[
            ['test_cases/hz0.png', 'test_cases/hz1.png'],
            ['test_cases/more_cases/az0.png', 'test_cases/more_cases/az1.JPG'],
            ['test_cases/more_cases/hi0.png', 'test_cases/more_cases/hi1.jpg'],
            ['test_cases/more_cases/kn0.jpg', 'test_cases/more_cases/kn1.jpg'],
            ['test_cases/more_cases/rk0.jpg', 'test_cases/more_cases/rk1.jpg'],
        ],
        inputs=[ref_orig, base_orig],
        label="Example Cases"
    )

# --- 8. Gradioアプリケーションの起動 ---
print("Launching Gradio interface...")
demo.launch(server_name="0.0.0.0", share=True)
