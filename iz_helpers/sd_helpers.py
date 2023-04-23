from modules.processing import (
    process_images,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
import modules.shared as shared


def renderTxt2Img(
    prompt, negative_prompt, sampler, steps, cfg_scale, seed, width, height
):
    processed = None
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler_name=sampler,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
    )
    processed = process_images(p)
    newseed = p.seed
    return processed, newseed


def renderImg2Img(
    prompt,
    negative_prompt,
    sampler,
    steps,
    cfg_scale,
    seed,
    width,
    height,
    init_image,
    mask_image,
    inpainting_denoising_strength,
    inpainting_mask_blur,
    inpainting_fill_mode,
    inpainting_full_res,
    inpainting_padding,
):
    processed = None

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_img2img_samples,
        outpath_grids=shared.opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler_name=sampler,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[init_image],
        denoising_strength=inpainting_denoising_strength,
        mask_blur=inpainting_mask_blur,
        inpainting_fill=inpainting_fill_mode,
        inpaint_full_res=inpainting_full_res,
        inpaint_full_res_padding=inpainting_padding,
        mask=mask_image,
    )
    # p.latent_mask = Image.new("RGB", (p.width, p.height), "white")

    processed = process_images(p)
    newseed = p.seed
    return processed, newseed
