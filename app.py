import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch
import gradio as gr
from safetensors.torch import load_file, save_file
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_FILE = "genshin_chibi_style_sd15_lora.safetensors"
UNET_ONLY_LORA_FILE = "genshin_chibi_style_sd15_lora_unet_only.safetensors"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

state_dict = load_file(LORA_FILE)
unet_state_dict = {
    k: v for k, v in state_dict.items()
    if k.startswith("lora_unet_")
}
save_file(unet_state_dict, UNET_ONLY_LORA_FILE)

pipe.unet.load_lora_adapter(
    ".",
    weight_name=UNET_ONLY_LORA_FILE,
    adapter_name="chibi",
    prefix="unet"
)

pipe.to(device)

if device == "cuda":
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


def generate(prompt, negative_prompt, lora_weight, steps, guidance_scale, seed):
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    seed = int(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # UNet에만 LoRA를 로드했으므로 UNet에서만 weight 조절
    pipe.unet.set_adapters(["chibi"], weights=[float(lora_weight)])

    full_prompt = (
        "genshin_chibi_style, chibi, big head, small body, simple shading, "
        + prompt
    )

    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        width=512,
        height=512,
    ).images[0]

    return image, seed


with gr.Blocks() as demo:
    gr.Markdown("# Anime Chibi Character Generator")
    gr.Markdown("Stable Diffusion 1.5 LoRA demo.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="1girl, cute, full body, smiling, white background",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="low quality, worst quality, blurry, bad anatomy, extra fingers, watermark, text",
            )
            lora_weight = gr.Slider(
                0.0, 1.5, value=0.7, step=0.05, label="LoRA Weight"
            )
            steps = gr.Slider(
                10, 40, value=25, step=1, label="Steps"
            )
            guidance_scale = gr.Slider(
                1, 15, value=7, step=0.5, label="CFG Scale"
            )
            seed = gr.Number(
                label="Seed (-1 for random)", value=-1, precision=0
            )
            button = gr.Button("Generate")

        with gr.Column():
            output = gr.Image(label="Generated Image")
            used_seed = gr.Number(label="Used Seed")

    button.click(
        fn=generate,
        inputs=[prompt, negative_prompt, lora_weight, steps, guidance_scale, seed],
        outputs=[output, used_seed],
    )

demo.queue()
demo.launch()