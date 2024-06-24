import torch
import os
from diffusers import DPMSolverMultistepScheduler
from torch import Generator
from torchvision import transforms

from transformers import CLIPTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler, ControlNetModel 

from .xadapter.model.unet_adapter import UNet2DConditionModel as UNet2DConditionModel_v2
from .xadapter.model.adapter import Adapter_XL
from .pipeline.pipeline_sd_xl_adapter_controlnet_img2img import StableDiffusionXLAdapterControlnetI2IPipeline
from .pipeline.pipeline_sd_xl_adapter_controlnet import StableDiffusionXLAdapterControlnetPipeline
from omegaconf import OmegaConf

from .utils.single_file_utils import (create_scheduler_from_ldm, create_text_encoders_and_tokenizers_from_ldm, convert_ldm_vae_checkpoint, 
                                      convert_ldm_unet_checkpoint, create_text_encoder_from_ldm_clip_checkpoint, create_vae_diffusers_config, 
                                      create_diffusers_controlnet_model_from_ldm, create_unet_diffusers_config)
from safetensors import safe_open

import comfy.model_management
import comfy.utils
import folder_paths
import math

script_directory = os.path.dirname(os.path.abspath(__file__))

class Diffusers_X_Adapter:
    def __init__(self):
        print("Initializing Diffusers_X_Adapter")
        self.device = comfy.model_management.get_torch_device()    
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(self.device) else torch.float32
        self.current_1_5_checkpoint = None
        self.current_lora = None
        self.current_controlnet_checkpoint = None
        self.original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
        self.sdxl_original_config = OmegaConf.load(os.path.join(script_directory, f"configs/sd_xl_base.yaml"))
        self.controlnet_original_config = OmegaConf.load(os.path.join(script_directory, f"configs/control_v11p_sd15.yaml"))
    @classmethod
    def IS_CHANGED(s):
        return ""
    @classmethod
    def INPUT_TYPES(cls):

        return {"required":
                {
                "width_sd1_5": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height_sd1_5": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "resolution_multiplier": ("INT", {"default": 2, "min": 2, "max": 2, "step": 1}),

                "sd_1_5_model": ("MODEL",),
                "sd_1_5_clip": ("CLIP", ),
                "sd_1_5_vae": ("VAE", ),
                "sdxl_model": ("MODEL",),
                "sdxl_clip": ("CLIP", ),
                "sdxl_vae": ("VAE", ),

                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "positive_sd1_5": ("CONDITIONING", ),
                "negative_sd1_5": ("CONDITIONING", ),

                "controlnet_name": (folder_paths.get_filename_list("controlnet"), ), 
                "guess_mode": ("BOOLEAN", {"default": False}),
                "control_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "controlnet_condition_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adapter_condition_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adapter_guidance_start": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_xformers": ("BOOLEAN", {"default": False}),
                },
                "optional": {
                "controlnet_image" : ("IMAGE",),
                "latent_source_image" : ("IMAGE",),
                },             
            }
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "Diffusers-X-Adapter"

    def load_checkpoint(self, use_xformers, sd_1_5_model, sd_1_5_vae, sdxl_model, sdxl_vae, positive, negative, positive_sd1_5, negative_sd1_5, sdxl_clip, sd_1_5_clip, resolution_multiplier,
                        controlnet_name, seed, steps, cfg, width_sd1_5, height_sd1_5, batch_size, #width_sdxl, height_sdxl, lora_checkpoint, use_lora, prompt_sdxl, prompt_sd1_5, negative_prompt,
                        adapter_condition_scale, adapter_guidance_start, controlnet_condition_scale, guess_mode, control_guidance_start, control_guidance_end, controlnet_image=None, latent_source_image=None):
        
        
        if latent_source_image is not None:
            latent_source_image = latent_source_image.permute(0, 3, 1, 2)

        #model_path_sd1_5 = folder_paths.get_full_path("checkpoints", sd_1_5_checkpoint)
        #lora_path = folder_paths.get_full_path("loras", lora_checkpoint)
        #model_path_sdxl = folder_paths.get_full_path("checkpoints", sdxl_checkpoint)
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
           
        #if not use_lora:
        #    self.current_lora = None

        #if not hasattr(self, 'unet_sd1_5')
        self.pipeline = None
        self.unet_sd1_5 = None
        # sd_1_5_clip.load_model()
            #comfy.model_management.soft_empty_cache()
            #print("Loading SD_1_5 checkpoint: ", sd_1_5_checkpoint)
            #self.current_1_5_checkpoint = sd_1_5_checkpoint
            #self.current_lora = lora_checkpoint
            #if model_path_sd1_5.endswith(".safetensors"):
            #    state_dict_sd1_5 = {}
            #    with safe_open(model_path_sd1_5, framework="pt", device="cpu") as f:
            #        for key in f.keys():
            #            state_dict_sd1_5[key] = f.get_tensor(key)
            #elif model_path_sd1_5.endswith(".ckpt"):
            #    state_dict_sd1_5 = torch.load(model_path_sd1_5, map_location="cpu")
            #    while "state_dict" in state_dict_sd1_5:
            #        state_dict_sd1_5 = state_dict_sd1_5["state_dict"]
        print("patching model sd15...")
        # sd_1_5_model.patch_model()
        comfy.model_management.load_models_gpu([sd_1_5_model], memory_required=0, force_patch_weights=True)
        print("finsihed")
        print("constructing state dictionary sd15...")
        state_dict_sd1_5 = sd_1_5_model.model.state_dict_for_saving(sd_1_5_clip.get_sd(), sd_1_5_vae.get_sd(), None)
        #state_dict_sd1_5 = sd_1_5_model.model.state_dict
        print("finished")

        # 1. vae
        converted_vae_config = create_vae_diffusers_config(self.original_config, image_size=512)
        converted_vae = convert_ldm_vae_checkpoint(state_dict_sd1_5, converted_vae_config)
        self.vae_sd1_5 = AutoencoderKL(**converted_vae_config)
        self.vae_sd1_5.load_state_dict(converted_vae, strict=False)
        self.vae_sd1_5.to(self.dtype)

        # 2. unet
        converted_unet_config = create_unet_diffusers_config(self.original_config, image_size=512)
        converted_unet = convert_ldm_unet_checkpoint(state_dict_sd1_5, converted_unet_config)
        self.unet_sd1_5 = UNet2DConditionModel_v2(**converted_unet_config)
        self.unet_sd1_5.load_state_dict(converted_unet, strict=False)
        self.unet_sd1_5.to(self.dtype)

        # 3. text encoder and tokenizer            
        converted_text_encoder_and_tokenizer = create_text_encoders_and_tokenizers_from_ldm(self.original_config, state_dict_sd1_5)
        self.tokenizer_sd1_5 = converted_text_encoder_and_tokenizer['tokenizer'] 
        self.text_encoder_sd1_5 = converted_text_encoder_and_tokenizer['text_encoder']
        self.text_encoder_sd1_5.to(self.dtype)

        # 4. scheduler
        self.scheduler_sd1_5 = create_scheduler_from_ldm("DPMSolverMultistepScheduler", self.original_config, state_dict_sd1_5, scheduler_type="ddim")['scheduler']

        del state_dict_sd1_5, converted_unet, converted_vae

        #if not self.current_lora != lora_checkpoint:
            # 5. lora
        #    if use_lora:
        #        print("Loading LoRA: ", lora_checkpoint)
        #        self.lora_checkpoint1 = lora_checkpoint
        #        if lora_path.endswith(".safetensors"):
        #            state_dict_lora = {}
        #            with safe_open(lora_path, framework="pt", device="cpu") as f:
        #                for key in f.keys():
        #                    state_dict_lora[key] = f.get_tensor(key)
        #        elif lora_path.endswith(".ckpt"):
        #            state_dict_lora = torch.load(lora_path, map_location="cpu")
        #            while "state_dict" in state_dict_lora:
        #                state_dict_lora = state_dict_lora["state_dict"]

        # load controlnet
        if controlnet_image is not None:
            if not hasattr(self, 'controlnet') or self.current_controlnet_checkpoint != controlnet_name:
                self.pipeline = None
                print("Loading controlnet: ", controlnet_name)
                self.current_controlnet_checkpoint = controlnet_name
    
                if controlnet_path.endswith(".safetensors"):
                    state_dict_controlnet = {}
                    with safe_open(controlnet_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict_controlnet[key] = f.get_tensor(key)
                else:
                    state_dict_controlnet = torch.load(controlnet_path, map_location="cpu")
                    while "state_dict" in state_dict_controlnet:
                        state_dict_controlnet = state_dict_controlnet["state_dict"]
                self.controlnet = create_diffusers_controlnet_model_from_ldm("ControlNet", self.controlnet_original_config, state_dict_controlnet)['controlnet']
                self.controlnet.to(self.dtype)

                del state_dict_controlnet
        else:
            self.controlnet = None
            self.current_controlnet_checkpoint = None

        # load Adapter_XL
        if not hasattr(self, 'adapter'):
            adapter_checkpoint_path = os.path.join(script_directory, "checkpoints","X-Adapter")
            if not os.path.exists(adapter_checkpoint_path):
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id="Lingmin-Ran/X-Adapter", local_dir=adapter_checkpoint_path, local_dir_use_symlinks=False)
                except:
                    raise FileNotFoundError(f"No checkpoint directory found at {adapter_checkpoint_path}")
            adapter_ckpt = torch.load(os.path.join(adapter_checkpoint_path, "X_Adapter_v1.bin"))
            adapter = Adapter_XL()
            adapter.load_state_dict(adapter_ckpt)
            adapter.to(self.dtype)
    
        # load SDXL
        # sdxl_clip.load_model()
        print("patching model sdxl...")
        # sdxl_model.patch_model()
        comfy.model_management.load_models_gpu([sdxl_model], memory_required=0, force_patch_weights=True)
        print("finished")
        print("constructing state dictionary sdxl...")
        state_dict_sdxl = sdxl_model.model.state_dict_for_saving(sdxl_clip.get_sd(), sdxl_vae.get_sd(), None)
        #state_dict_sdxl = sdxl_model.model.state_dict
        print("finished")

        #if not hasattr(self, 'unet_sdxl') or self.current_sdxl_checkpoint != sdxl_checkpoint:
        #    self.pipeline = None
        #    comfy.model_management.soft_empty_cache()
        #    print("Loading SDXL checkpoint: ", sdxl_checkpoint)
        #    self.current_sdxl_checkpoint = sdxl_checkpoint
        #    if model_path_sdxl.endswith(".safetensors"):
        #        state_dict_sdxl = {}
        #        with safe_open(model_path_sdxl, framework="pt", device="cpu") as f:
        #            for key in f.keys():
        #                state_dict_sdxl[key] = f.get_tensor(key)
        #    elif model_path_sdxl.endswith(".ckpt"):
        #        state_dict_sdxl = torch.load(model_path_sdxl, map_location="cpu")
        #        while "state_dict" in state_dict_sdxl:
        #            state_dict_sdxl = state_dict_sdxl["state_dict"]

        # 1. vae
        converted_vae_config = create_vae_diffusers_config(self.sdxl_original_config, image_size=1024)
        converted_vae = convert_ldm_vae_checkpoint(state_dict_sdxl, converted_vae_config)
        self.vae_sdxl = AutoencoderKL(**converted_vae_config)
        self.vae_sdxl.load_state_dict(converted_vae, strict=False)
        self.vae_sdxl.to(self.dtype)

        # 2. unet
        converted_unet_config = create_unet_diffusers_config(self.sdxl_original_config, image_size=1024)
        converted_unet = convert_ldm_unet_checkpoint(state_dict_sdxl, converted_unet_config)
        self.unet_sdxl = UNet2DConditionModel_v2(**converted_unet_config)
        self.unet_sdxl.load_state_dict(converted_unet, strict=False)
        self.unet_sdxl.to(self.dtype)
        #cross_attn_dim = converted_unet_config["cross_attention_dim"]
        #print(f"context_dim: {cross_attn_dim}")

        # 3. text encoders and tokenizers
        converted_sdxl_stuff = create_text_encoders_and_tokenizers_from_ldm(self.sdxl_original_config, state_dict_sdxl)
        self.tokenizer_one = converted_sdxl_stuff['tokenizer'] 
        self.sdxl_text_encoder = converted_sdxl_stuff['text_encoder']
        self.tokenizer_two = converted_sdxl_stuff['tokenizer_2']
        self.sdxl_text_encoder2 = converted_sdxl_stuff['text_encoder_2']
        self.sdxl_text_encoder.to(self.dtype)
        self.sdxl_text_encoder2.to(self.dtype)

        # 4. scheduler
        self.scheduler_sdxl = create_scheduler_from_ldm("DPMSolverMultistepScheduler", self.sdxl_original_config, state_dict_sdxl, scheduler_type="ddim",)['scheduler']

        del state_dict_sdxl, converted_unet, converted_vae

        #xformers
        if use_xformers:
            self.unet_sd1_5.enable_xformers_memory_efficient_attention()
            self.unet_sdxl.enable_xformers_memory_efficient_attention()
            if self.controlnet is not None:
                self.controlnet.enable_xformers_memory_efficient_attention()
        else:
            self.unet_sd1_5.disable_xformers_memory_efficient_attention()
            self.unet_sdxl.disable_xformers_memory_efficient_attention()
            if self.controlnet is not None:
                self.controlnet.disable_xformers_memory_efficient_attention()
        

        self.pipeline = StableDiffusionXLAdapterControlnetPipeline(
            vae=self.vae_sdxl,
            text_encoder=self.sdxl_text_encoder,
            text_encoder_2=self.sdxl_text_encoder2,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet_sdxl,
            scheduler=self.scheduler_sdxl,
            vae_sd1_5=self.vae_sd1_5,
            text_encoder_sd1_5=self.text_encoder_sd1_5,
            tokenizer_sd1_5=self.tokenizer_sd1_5,
            unet_sd1_5=self.unet_sd1_5,
            scheduler_sd1_5=self.scheduler_sd1_5,
            adapter=adapter,
            controlnet=self.controlnet)

        self.pipeline.enable_model_cpu_offload()

        self.pipeline.scheduler_sd1_5.config.timestep_spacing = "leading"
        #self.pipeline.scheduler.config.timestep_spacing = "trailing"
        self.pipeline.unet.to(device=self.device, dtype=self.dtype)

        if controlnet_image is not None:
            control_image = controlnet_image.permute(0, 3, 1, 2)
        else:
            control_image = None

        width_sdxl = resolution_multiplier * width_sd1_5
        height_sdxl = resolution_multiplier * height_sd1_5

        #get prompt embeddings from conditioning
        positive_embed = positive[0][0]
        negative_embed = negative[0][0]
        crossattn_max_len = math.lcm(positive_embed.shape[1], negative_embed.shape[1])
        positive_embed = positive_embed.repeat(1, crossattn_max_len // positive_embed.shape[1], 1)
        negative_embed = negative_embed.repeat(1, crossattn_max_len // negative_embed.shape[1], 1)

        positive_embed_sd1_5 = positive_sd1_5[0][0]
        negative_embed_sd1_5 = negative_sd1_5[0][0]
        crossattn_max_len = math.lcm(positive_embed_sd1_5.shape[1], negative_embed_sd1_5.shape[1])
        positive_embed_sd1_5 = positive_embed_sd1_5.repeat(1, crossattn_max_len // positive_embed_sd1_5.shape[1], 1)
        negative_embed_sd1_5 = negative_embed_sd1_5.repeat(1, crossattn_max_len // negative_embed_sd1_5.shape[1], 1)

        positive_pooled_out = positive[0][1]["pooled_output"]
        negative_pooled_out = positive[0][1]["pooled_output"]

        #run inference
        gen = Generator(self.device)
        gen.manual_seed(seed)
    
        img = \
            self.pipeline(prompt=None, negative_prompt=None, prompt_sd1_5=None,
                    prompt_embeds=positive_embed, negative_prompt_embeds=negative_embed, prompt_embeds_sd_1_5=positive_embed_sd1_5, negative_prompt_embeds_sd_1_5=negative_embed_sd1_5, pooled_prompt_embeds=positive_pooled_out, negative_pooled_prompt_embeds=negative_pooled_out,
                    width=width_sdxl, height=height_sdxl, height_sd1_5=height_sd1_5, width_sd1_5=width_sd1_5,
                    image=control_image,
                    num_inference_steps=steps, guidance_scale=cfg,
                    num_images_per_prompt=batch_size, generator=gen,
                    controlnet_conditioning_scale=controlnet_condition_scale,
                    adapter_condition_scale=adapter_condition_scale,
                    adapter_guidance_start=adapter_guidance_start, guess_mode=guess_mode, control_guidance_start=control_guidance_start, 
                    control_guidance_end=control_guidance_end, source_img=latent_source_image).images
        
        image_tensor = (img - img.min()) / (img.max() - img.min())
        if image_tensor.dim() ==  3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.permute(0,  2,  3,  1)
 
        return (image_tensor,)
        
NODE_CLASS_MAPPINGS = {
    "Diffusers_X_Adapter": Diffusers_X_Adapter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffusers_X_Adapter": "Diffusers_X_Adapter",
}