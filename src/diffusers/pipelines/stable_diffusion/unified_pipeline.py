import inspect
import warnings
import gc
import random
import sys
from tqdm.auto import tqdm
from typing import List, Optional, Union

import numpy as np
import torch
import PIL

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask



class UnifiedPipeline(DiffusionPipeline):
    r"""
    General Stable Diffusion Pipeline. Merge of text-to-image, image-to-image and inpaint pipelines, plus some
    other modifications, in particular:

        - Is possible to move _just_ the unet to cuda and have the pipeline still run (reduces VRAM)
        - Negative prompting

        TODO:

        - Remove needing autocast for FP16 (big speedup for FP16)

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            warnings.warn(
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file",
                DeprecationWarning,
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def enable_minimal_memory_usage(self):
        """Moves only unet to fp16 and to CUDA, while keepping lighter models on CPUs"""
        self.unet.to(torch.float16).to(torch.device("cuda"))
        self.enable_attention_slicing(1)

        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        mask_image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        strength: Optional[float] = 0.8,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        When init_image is not provided, runs a text to image process
        When init_image is provided, but a mask isn't, runs an image to image process
        When init_image and mask is provided, runs an inpaint process

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
               `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. The mask image will be
                converted to a single channel (luminance) before use.
            strength (`float`, *optional*, defaults to 0.8):
                Ignored unless init_image is provided.
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.            
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # Calculate operating mode based on arguments
        if mask_image: mode = "inpaint"
        elif init_image: mode = "img2img"
        else: mode = "txt2img"

        if mode == "txt2img":
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        elif strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)


        if mode == "txt2img":
            # get the initial random noise unless the user supplied it

            # Unlike in other pipelines, latents need to be generated in the target device
            # for 1-to-1 results reproducibility with the CompVis implementation.
            # However this currently doesn't work in `mps`.
            latents_device = "cpu" if self.device.type == "mps" else self.device
            latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
            if latents is None:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=latents_device,
                )
            else:
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = latents * self.scheduler.sigmas[0]

            t_start = 0

        else:
            if isinstance(init_image, PIL.Image.Image):
                init_image = preprocess(init_image)

            if mode == "inpaint": init_image = init_image.to(self.device)

            # encode the init image into latents and scale the latents
            init_latent_dist = self.vae.encode(init_image.to(self.device)).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

            # expand init_latents for batch_size
            init_latents = torch.cat([init_latents] * batch_size)

            if mode == "inpaint":
                init_latents_orig = init_latents

                # preprocess mask
                if isinstance(mask_image, PIL.Image.Image):
                    mask_image = preprocess_mask(mask_image)
                mask_image = mask_image.to(self.device)
                mask = torch.cat([mask_image] * batch_size)

                # check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError("The mask and init_image should be the same size!")


            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                timesteps = torch.tensor(
                    [num_inference_steps - init_timestep] * batch_size, dtype=torch.long, device=self.device
                )
            else:
                timesteps = self.scheduler.timesteps[-init_timestep]
                timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

            # add noise to latents using the timesteps
            noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps).to(self.device)

            latents = init_latents
            t_start = max(num_inference_steps - init_timestep + offset, 0)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0].to(self.unet.device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            ucond_tokens: List[str]
            if negative_prompt is None:
                ucond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError("`negative_prompt` should be the same type to `prompt`.")
            elif isinstance(negative_prompt, str):
                ucond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
            else:
                ucond_tokens = negative_prompt

            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                ucond_tokens, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0].to(self.unet.device)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[t_start:])):
            t_index = t_start + i

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                latent_model_input = latent_model_input.to(self.unet.dtype)
                t = t.to(self.unet.dtype)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input.to(self.unet.device), t.to(self.unet.device), encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if mode == "inpaint":
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # masking
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * mask) + (latents * (1 - mask))
            else:
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = self.scheduler.step(noise_pred, t_index, latents.to(self.unet.device), **extra_step_kwargs).prev_sample
                else:
                    latents = self.scheduler.step(noise_pred, t.to(self.unet.device), latents.to(self.unet.device), **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.to(self.vae.device).cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        #safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        #image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)
        has_nsfw_concept = False

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def get_text_latent_space(self, prompt, guidance_scale = 7.5):
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def slerp(self, t, v0, v1, DOT_THRESHOLD=0.9995):
        """ helper function to spherically interpolate two arrays v1 v2 
        from https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355 
        this should be better than lerping for moving between noise spaces """

        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)

        return v2

    def lerp_between_prompts(self, first_prompt, second_prompt, seed = None, length = 10, save=False, guidance_scale: Optional[float] = 7.5, **kwargs):
        first_embedding = self.get_text_latent_space(first_prompt)
        second_embedding = self.get_text_latent_space(second_prompt)
        if not seed:
            seed = random.randint(0, sys.maxsize)
        generator = torch.Generator(self.device)
        generator.manual_seed(seed)
        generator_state = generator.get_state()
        lerp_embed_points = []
        for i in range(length):
            weight = i / length
            tensor_lerp = torch.lerp(first_embedding, second_embedding, weight)
            lerp_embed_points.append(tensor_lerp)
        images = []
        for idx, latent_point in enumerate(lerp_embed_points):
            generator.set_state(generator_state)
            image = self.diffuse_from_inits(latent_point, **kwargs)["image"][0]
            images.append(image)
            if save:
                image.save(f"{first_prompt}-{second_prompt}-{idx:02d}.png", "PNG")
        return {"images": images, "latent_points": lerp_embed_points,"generator_state": generator_state}

    def slerp_through_seeds(self,
        prompt,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        save = False,
        seed = None, steps = 10, **kwargs):

        if not seed:
            seed = random.randint(0, sys.maxsize)
        generator = torch.Generator(self.device)
        generator.manual_seed(seed)
        init_start = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8), 
            generator = generator, device = self.device)
        init_end = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8), 
            generator = generator, device = self.device)
        generator_state = generator.get_state()
        slerp_embed_points = []
        # weight from 0 to 1/(steps - 1), add init_end specifically so that we 
        # have len(images) = steps
        for i in range(steps - 1):
            weight = i / steps
            tensor_slerp = self.slerp(weight, init_start, init_end)
            slerp_embed_points.append(tensor_slerp)
        slerp_embed_points.append(init_end)
        images = []
        embed_point = self.get_text_latent_space(prompt)
        for idx, noise_point in enumerate(slerp_embed_points):
            generator.set_state(generator_state)
            image = self.diffuse_from_inits(embed_point, init = noise_point, **kwargs)["image"][0]
            images.append(image)
            if save:
                image.save(f"{seed}-{idx:02d}.png", "PNG")
        return {"images": images, "noise_samples": slerp_embed_points,"generator_state": generator_state}

    @torch.no_grad()
    def diffuse_from_inits(self, text_embeddings, 
        init = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        **kwargs,):

        batch_size = 1

        if generator == None:
            generator = torch.Generator("cuda")
        generator_state = generator.get_state()
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get the intial random noise
        latents = init if init is not None else torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"image": image, "generator_state": generator_state}

    def variation(self, text_embeddings, generator_state, variation_magnitude = 100, **kwargs):
        # random vector to move in latent space
        rand_t = (torch.rand(text_embeddings.shape, device = self.device) * 2) - 1
        rand_mag = torch.sum(torch.abs(rand_t)) / variation_magnitude
        scaled_rand_t = rand_t / rand_mag
        variation_embedding = text_embeddings + scaled_rand_t

        generator = torch.Generator("cuda")
        generator.set_state(generator_state)
        result = self.diffuse_from_inits(variation_embedding, generator=generator, **kwargs)
        result.update({"latent_point": variation_embedding})
        return result
