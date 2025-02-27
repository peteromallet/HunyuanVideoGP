from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from loguru import logger

from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from hyvideo.vae import load_vae
from hyvideo.modules import load_model
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.data_utils import align_to
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline

# Video transforms for conditioning images
video_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]

    return image


class HunyuanVideoI2VSampler:
    """
    Image-to-Video sampler for HunyuanVideo
    """
    def __init__(self, hunyuan_video_sampler):
        """
        Initialize the I2V sampler with an existing HunyuanVideoSampler
        
        Args:
            hunyuan_video_sampler: An initialized HunyuanVideoSampler
        """
        self.hunyuan_video_sampler = hunyuan_video_sampler
        self.pipeline = hunyuan_video_sampler.pipeline
        self.vae = hunyuan_video_sampler.vae
        self.text_encoder = hunyuan_video_sampler.text_encoder
        self.text_encoder_2 = hunyuan_video_sampler.text_encoder_2
        self.model = hunyuan_video_sampler.model
        self.args = hunyuan_video_sampler.args
        self.device = hunyuan_video_sampler.device
        
        # Modify the transformer to accept image conditioning
        self._setup_i2v_model()
        
    def _setup_i2v_model(self):
        """
        Modify the transformer to accept image conditioning by doubling the input channels
        """
        with torch.no_grad():
            transformer = self.model
            initial_input_channels = transformer.in_channels
            
            # Create a new patch embedding with double the input channels
            from hyvideo.modules.embed_layers import PatchEmbed
            
            new_img_in = PatchEmbed(
                patch_size=transformer.patch_size,
                in_chans=initial_input_channels * 2,
                embed_dim=transformer.hidden_size,
                device=transformer.device,
                dtype=transformer.dtype
            )
            
            # Initialize the new weights to zero
            new_img_in.proj.weight.zero_()
            
            # Copy the original weights to the first half of the input channels
            new_img_in.proj.weight[:, :initial_input_channels].copy_(transformer.img_in.proj.weight)
            
            # Copy bias if it exists
            if transformer.img_in.proj.bias is not None:
                new_img_in.proj.bias.copy_(transformer.img_in.proj.bias)
            
            # Replace the original patch embedding
            transformer.img_in = new_img_in
            
            logger.info("Modified transformer for image-to-video generation")
    
    def prepare_image_latents(self, image_or_video, height, width, video_length):
        """
        Prepare latents from conditioning image(s)
        
        Args:
            image_or_video: PIL Image, numpy array, or tensor of shape [B, C, H, W] or [B, F, C, H, W]
            height: Target height
            width: Target width
            video_length: Target video length
            
        Returns:
            torch.Tensor: Conditioning latents
        """
        # Handle different input types
        if isinstance(image_or_video, (Image.Image, np.ndarray)):
            # Single image
            image = resize_image_to_bucket(image_or_video, bucket_reso=(width, height))
            image = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).unsqueeze(0)
            image = video_transforms(image).to(device=self.device, dtype=self.pipeline.transformer.dtype)
            
            # Create a video tensor with the same image repeated
            image_or_video = image.unsqueeze(1).repeat(1, video_length, 1, 1, 1)
            
        elif isinstance(image_or_video, torch.Tensor):
            if image_or_video.ndim == 4:  # [B, C, H, W]
                # Single image
                image_or_video = image_or_video.unsqueeze(1).repeat(1, video_length, 1, 1, 1)
            
            # Apply transforms if needed
            if image_or_video.max() > 1.0:
                image_or_video = torch.stack([video_transforms(x) for x in image_or_video[0]], dim=0).unsqueeze(0)
                
            image_or_video = image_or_video.to(device=self.device, dtype=self.pipeline.transformer.dtype)
        
        # Encode to latents
        with torch.no_grad():
            # [B, F, C, H, W] -> [B, C, F, H, W]
            image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()
            
            # Encode with VAE
            cond_latents = self.vae.encode(image_or_video).latent_dist.sample()
            cond_latents = cond_latents * self.vae.config.scaling_factor
            cond_latents = cond_latents.to(dtype=self.pipeline.transformer.dtype)
            
        return cond_latents
    
    def prepare_multiple_image_latents(self, images, height, width, video_length):
        """
        Prepare latents from multiple conditioning images
        
        Args:
            images: List of PIL Images or numpy arrays
            height: Target height
            width: Target width
            video_length: Target video length
            
        Returns:
            torch.Tensor: Interpolated conditioning latents
        """
        if not isinstance(images, list):
            images = [images]
        
        # If only one image provided, duplicate it
        if len(images) == 1:
            images = [images[0], images[0]]
        
        # Use only first and last if more than 2 images
        if len(images) > 2:
            images = [images[0], images[-1]]
        
        # Process each image to get latents
        latents = []
        for img in images:
            # Resize and convert to tensor
            img = resize_image_to_bucket(img, bucket_reso=(width, height))
            img_tensor = torch.from_numpy(np.array(img).copy()).permute(2, 0, 1).unsqueeze(0)
            img_tensor = video_transforms(img_tensor).to(device=self.device, dtype=self.pipeline.transformer.dtype)
            
            # Encode to latent
            with torch.no_grad():
                latent = self.vae.encode(img_tensor).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
                latent = latent.to(dtype=self.pipeline.transformer.dtype)
                latents.append(latent)
        
        # Interpolate between latents
        return self.interpolate_latents(latents, video_length)

    def interpolate_latents(self, latents, video_length):
        """
        Interpolate between latents to create a video sequence
        
        Args:
            latents: List of latent tensors [first_frame, last_frame]
            video_length: Number of frames to generate
            
        Returns:
            torch.Tensor: Interpolated latents for the full video
        """
        # Get first and last frame latents
        first_latent = latents[0]
        last_latent = latents[1]
        
        # Calculate latent frame size
        t_size = (video_length - 1) // 4 + 1  # Adjust based on VAE temporal compression
        
        # Create interpolation weights
        weights = torch.linspace(0, 1, t_size, device=self.device)
        
        # Create empty tensor for all frames
        batch_size = first_latent.shape[0]
        channels = first_latent.shape[1]
        height = first_latent.shape[2]
        width = first_latent.shape[3]
        
        # Create the interpolated latents
        interpolated = torch.zeros((batch_size, channels, t_size, height, width), 
                                  device=self.device, 
                                  dtype=first_latent.dtype)
        
        # Fill with interpolated frames
        for i, w in enumerate(weights):
            interpolated[:, :, i, :, :] = (1 - w) * first_latent + w * last_latent
        
        return interpolated

    def predict_step(self, image_or_video, video_length, prompt, **kwargs):
        """
        Generate a video from one or more images and a text prompt
        
        Args:
            image_or_video: PIL Image, numpy array, tensor, or list of these
            video_length: Number of frames to generate
            prompt: Text prompt
            **kwargs: Additional arguments for predict
            
        Returns:
            dict: Output dictionary containing the generated video
        """
        # Get parameters from kwargs or use defaults
        height = kwargs.pop('height', 720)
        width = kwargs.pop('width', 1280)
        seed = kwargs.pop('seed', None)
        negative_prompt = kwargs.pop('negative_prompt', None)
        infer_steps = kwargs.pop('infer_steps', 50)
        guidance_scale = kwargs.pop('guidance_scale', 6.0)
        embedded_guidance_scale = kwargs.pop('embedded_guidance_scale', None)
        flow_shift = kwargs.pop('flow_shift', 7.0)
        batch_size = kwargs.pop('batch_size', 1)
        num_videos_per_prompt = kwargs.pop('num_videos_per_prompt', 1)
        enable_riflex = kwargs.pop('enable_riflex', False)
        
        # Handle different input types
        if isinstance(image_or_video, list):
            # Multiple images
            cond_latents = self.prepare_multiple_image_latents(image_or_video, height, width, video_length)
        else:
            # Single image
            cond_latents = self.prepare_image_latents(image_or_video, height, width, video_length)
        
        # Call the regular predict method with modified arguments
        return self.predict(
            prompt=prompt,
            height=height,
            width=width,
            video_length=video_length,
            seed=seed,
            negative_prompt=negative_prompt,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            embedded_guidance_scale=embedded_guidance_scale,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            enable_riflex=enable_riflex,
            image_latents=cond_latents,
            **kwargs
        )
    
    def predict(
        self,
        prompt,
        height=720,
        width=1280,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6.0,
        flow_shift=7.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        enable_riflex=False,
        image_latents=None,
        **kwargs,
    ):
        """
        Predict the video from the given text and image.

        Args:
            prompt (str or List[str]): The input text.
            height (int): The height of the output video. Default is 720.
            width (int): The width of the output video. Default is 1280.
            video_length (int): The frame number of the output video. Default is 129.
            seed (int or List[str]): The random seed for the generation. Default is a random integer.
            negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
            guidance_scale (float): The guidance scale for the generation. Default is 6.0.
            num_videos_per_prompt (int): The number of videos per prompt. Default is 1.
            infer_steps (int): The number of inference steps. Default is 50.
            image_latents (torch.Tensor): Conditioning image latents. Required for I2V.
            **kwargs: Additional arguments.
        """
        import random
        out_dict = dict()

        # Check if image_latents is provided
        if image_latents is None:
            raise ValueError("image_latents must be provided for image-to-video generation")

        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.hunyuan_video_sampler.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver
        )
        self.pipeline.scheduler = scheduler

        # ========================================================================
        # Build Rope freqs
        # ========================================================================
        freqs_cos, freqs_sin = self.hunyuan_video_sampler.get_rotary_pos_embed(
            target_video_length, target_height, target_width, enable_riflex
        )
        n_tokens = freqs_cos.shape[0]

        # ========================================================================
        # Print infer args
        # ========================================================================
        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}"""
        logger.debug(debug_str)

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        import time
        start_time = time.time()
        
        # Prepare latents for the noise
        latents = self.pipeline.prepare_latents(
            batch_size * num_videos_per_prompt,
            self.pipeline.transformer.config.in_channels,
            target_height,
            target_width,
            target_video_length,
            prompt_embeds=None,
            dtype=self.pipeline.transformer.dtype,
            device=self.device,
            generator=generator,
            latents=None,
        )
        
        # Modify the forward call to include image latents
        original_forward = self.model.forward
        
        def modified_forward(self, x, t, text_states=None, text_mask=None, text_states_2=None, 
                            freqs_cos=None, freqs_sin=None, guidance=None, pipeline=None, return_dict=True):
            # Split the input into two parts: noise and image conditioning
            noise_latents, image_latents = torch.chunk(x, 2, dim=1)
            
            # Concatenate them along the channel dimension for the modified patch embedding
            combined_input = torch.cat([noise_latents, image_latents], dim=1)
            
            # Call the original forward with the combined input
            return original_forward(self, combined_input, t, text_states, text_mask, 
                                   text_states_2, freqs_cos, freqs_sin, guidance, pipeline, return_dict)
        
        # Temporarily replace the forward method
        self.model.forward = modified_forward.__get__(self.model, type(self.model))
        
        try:
            # Concatenate the latents with image_latents for the model input
            model_input = torch.cat([latents, image_latents], dim=1)
            
            samples = self.pipeline(
                prompt=prompt,
                height=target_height,
                width=target_width,
                video_length=target_video_length,
                num_inference_steps=infer_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=model_input,  # Use the combined latents
                output_type="pil",
                freqs_cis=(freqs_cos, freqs_sin),
                n_tokens=n_tokens,
                embedded_guidance_scale=embedded_guidance_scale,
                data_type="video" if target_video_length > 1 else "image",
                is_progress_bar=True,
                vae_ver=self.args.vae,
                enable_tiling=self.args.vae_tiling,
                callback=callback,
                callback_steps=callback_steps,
            )[0]
            
            out_dict["samples"] = samples
            out_dict["prompts"] = prompt
            
            gen_time = time.time() - start_time
            logger.info(f"Success, time: {gen_time}")
            
        finally:
            # Restore the original forward method
            self.model.forward = original_forward
        
        return out_dict 