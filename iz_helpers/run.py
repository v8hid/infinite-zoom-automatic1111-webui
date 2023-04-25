import math, time, os
import numpy as np
from PIL import Image
from modules.ui import plaintext_to_html
import modules.shared as shared

from .helpers import (
    fix_env_Path_ffprobe,
    closest_upper_divisible_by_eight,
    load_model_from_setting,
    do_upscaleImg,
    predict_upscalesize
)
from .sd_helpers import renderImg2Img, renderTxt2Img
from .image import shrink_and_paste_on_blank
from .video import write_video


def create_zoom(
    prompts_array,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    custom_init_image,
    custom_exit_image,
    video_frame_rate,
    video_zoom_mode,
    video_start_frame_dupe_amount,
    video_last_frame_dupe_amount,
    inpainting_denoising_strength,
    inpainting_mask_blur,
    inpainting_fill_mode,
    inpainting_full_res,
    inpainting_padding,
    zoom_speed,
    seed,
    outputsizeW,
    outputsizeH,
    batchcount,
    sampler,
    upscale_do,
    upscaler_name,
    upscale_by,
    progress=None,
):
    for i in range(batchcount):
        print(f"Batch {i+1}/{batchcount}")
        result = create_zoom_single(
            prompts_array,
            negative_prompt,
            num_outpainting_steps,
            guidance_scale,
            num_inference_steps,
            custom_init_image,
            custom_exit_image,
            video_frame_rate,
            video_zoom_mode,
            video_start_frame_dupe_amount,
            video_last_frame_dupe_amount,
            inpainting_denoising_strength,
            inpainting_mask_blur,
            inpainting_fill_mode,
            inpainting_full_res,
            inpainting_padding,
            zoom_speed,
            seed,
            outputsizeW,
            outputsizeH,
            sampler,
            upscale_do,
            upscaler_name,
            upscale_by,
            progress,
        )
    return result


def create_zoom_single(
    prompts_array,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    custom_init_image,
    custom_exit_image,
    video_frame_rate,
    video_zoom_mode,
    video_start_frame_dupe_amount,
    video_last_frame_dupe_amount,
    inpainting_denoising_strength,
    inpainting_mask_blur,
    inpainting_fill_mode,
    inpainting_full_res,
    inpainting_padding,
    zoom_speed,
    seed,
    outputsizeW,
    outputsizeH,
    sampler,
    upscale_do,
    upscaler_name,
    upscale_by,
    progress=None,
):
    # try:
    #     if gr.Progress() is not None:
    #         progress = gr.Progress()
    #         progress(0, desc="Preparing Initial Image")
    # except Exception:
    #     pass
    fix_env_Path_ffprobe()

    prompts = {}
    for x in prompts_array:
        try:
            key = int(x[0])
            value = str(x[1])
            prompts[key] = value
        except ValueError:
            pass
    assert len(prompts_array) > 0, "prompts is empty"

    width = closest_upper_divisible_by_eight(outputsizeW)
    height = closest_upper_divisible_by_eight(outputsizeH)

    # smart upscale: only keyframes to upscale, 
    # therefore change size as state for the hole process
    if upscale_do:
        widthInterp,heightInterp,void = predict_upscalesize(outputsizeW, outputsizeH,upscale_by)
    else:
        widthInterp = width
        heightInterp = height

    current_image = Image.new(mode="RGBA", size=(width, height))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    current_image = current_image.convert("RGB")
    current_seed = seed

    if custom_init_image:
        if upscale_do:
            print(f"using Custom Initial Image, upscaling using {upscaler_name}")
            current_image = do_upscaleImg(custom_init_image,True,upscaler_name,(width,height))

        else:
            print(f"using Custom Initial Image, simple resizing to {width} x {height}")
            current_image = custom_init_image.resize(
                (width, height), resample=Image.LANCZOS
            )
    else:
        load_model_from_setting(
            "infzoom_txt2img_model", progress, "Loading Model for txt2img: "
        )

        processed, newseed = renderTxt2Img(
            prompts[min(k for k in prompts.keys() if k >= 0)],
            negative_prompt,
            sampler,
            num_inference_steps,
            guidance_scale,
            current_seed,
            width,
            height,
        )

        if(len(processed.images) > 0):
            current_image = processed.images[0]
        current_seed = newseed


    num_interpol_frames = round(video_frame_rate * zoom_speed)

    all_frames = []

    if upscale_do:
        if progress: 
            progress(0, desc="upscaling inital txt2image")
            all_frames.append(do_upscaleImg(current_image, upscale_do, upscaler_name, upscale_by))
    else:
        all_frames.append(current_image)
### end of txt2img

## begin img2img
    load_model_from_setting(
        "infzoom_inpainting_model", progress, "Loading Model for inpainting/img2img: "
    )

    for i in range(num_outpainting_steps):
        mask_width = math.trunc(outputsizeH / 4)  # was initially 512px => 128px
        mask_height = math.trunc(outputsizeH / 4)  # was initially 512px => 128px
        print_out = (
            "Outpaint step: "
            + str(i + 1)
            + " / "
            + str(num_outpainting_steps)
            + " Seed: "
            + str(current_seed)
        )
        print(print_out)
        if progress:
            progress(((i + 1) / num_outpainting_steps), desc=print_out)

        prev_image_fix = current_image
        prev_image = shrink_and_paste_on_blank(current_image, mask_width, mask_height)
        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")

        if custom_exit_image and ((i + 1) == num_outpainting_steps):
            if upscale_do:
                if progress: 
                    progress(0, desc=f"upscaling Custom Inital image using {upscaler_name}")

                print (f"upscaling Custom Inital image using {upscaler_name}")
                current_image = do_upscaleImg(custom_exit_image, upscale_do, upscaler_name, (outputsizeW, outputsizeH))
            else:
                print("using Custom Exit Image, simple resizing")
                current_image = custom_exit_image.resize(
                    (outputsizeW, outputsizeH), resample=Image.LANCZOS
                )

        else:
            processed, newseed = renderImg2Img(
                prompts[max(k for k in prompts.keys() if k <= i)],
                negative_prompt,
                sampler,
                num_inference_steps,
                guidance_scale,
                current_seed,
                outputsizeW,
                outputsizeH,
                current_image,
                mask_image,
                inpainting_denoising_strength,
                inpainting_mask_blur,
                inpainting_fill_mode,
                inpainting_full_res,
                inpainting_padding,
            )

            if(len(processed.images) > 0):
                    current_image = processed.images[0]
            current_seed = newseed

        if(len(processed.images) > 0):
            current_image.paste(prev_image, mask=prev_image)
### end img2img

### begin interpolation
        # from here in case of upscale everything is XL:
        if upscale_do:
            if progress: 
                progress(0, desc="upscaling curr+prevImage")

            current_imageXL  = do_upscaleImg(current_image, upscale_do, upscaler_name, upscale_by)
            prev_image_fixXL = do_upscaleImg(prev_image_fix, upscale_do, upscaler_name, upscale_by)
            mask_widthXL = math.trunc(widthInterp / 4)  # was initially 512px => 128px
            mask_heightXL = math.trunc(heightInterp / 4)  # was initially 512px => 128px
        else:
            current_imageXL = current_image
            mask_widthXL = math.trunc(outputsizeW / 4)  # was initially 512px => 128px
            mask_heightXL = math.trunc(outputsizeH / 4)  # was initially 512px => 128px

        # interpolation steps between 2 inpainted (upscaled) images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_imageXL

            interpol_width = round(
                (
                    1
                    - (1 - 2 * mask_widthXL / widthInterp)
                    ** (1 - (j + 1) / num_interpol_frames)
                )
                * widthInterp
                / 2
            )

            interpol_height = round(
                (
                    1
                    - (1 - 2 * mask_heightXL / heightInterp)
                    ** (1 - (j + 1) / num_interpol_frames)
                )
                * heightInterp
                / 2
            )

            interpol_image = interpol_image.crop(
                (
                    interpol_width,
                    interpol_height,
                    widthInterp - interpol_width,
                    heightInterp - interpol_height,
                )
            )

   ## WHY?, should be end size?  interpol_image = interpol_image.resize((width, height))

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round(
                (1 - (widthInterp - 2 * mask_widthXL) / (widthInterp - 2 * interpol_width))
                / 2
                * widthInterp
            )

            interpol_height2 = round(
                (1 - (heightInterp - 2 * mask_heightXL) / (heightInterp - 2 * interpol_height))
                / 2
                * heightInterp
            )

            prev_image_fix_crop = shrink_and_paste_on_blank(
                prev_image_fixXL, interpol_width2, interpol_height2
            )

            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

            all_frames.append(interpol_image)

        all_frames.append(current_image)

    video_file_name = "infinite_zoom_" + str(int(time.time())) + ".mp4"
    output_path = shared.opts.data.get(
        "infzoom_outpath", shared.opts.data.get("outdir_img2img_samples")
    )
    save_path = os.path.join(
        output_path, shared.opts.data.get("infzoom_outSUBpath", "infinite-zooms")
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    out = os.path.join(save_path, video_file_name)
    write_video(
        out,
        all_frames,
        video_frame_rate,
        video_zoom_mode,
        int(video_start_frame_dupe_amount),
        int(video_last_frame_dupe_amount),
    )

    return (
        out,
        processed.images,
        processed.js(),
        plaintext_to_html(processed.info),
        plaintext_to_html(""),
    )
