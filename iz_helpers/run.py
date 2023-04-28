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
)
from .sd_helpers import renderImg2Img, renderTxt2Img
from .image import shrink_and_paste_on_blank
from .video import write_video


def outpaint_steps(
    width,
    height,
    common_prompt_pre,
    common_prompt_suf,
    prompts,
    negative_prompt,
    seed,
    sampler,
    num_inference_steps,
    guidance_scale,
    inpainting_denoising_strength,
    inpainting_mask_blur,
    inpainting_fill_mode,
    inpainting_full_res,
    inpainting_padding,
    init_img,
    outpaint_steps,
    out_config,
    mask_width,
    mask_height,
    custom_exit_image,
    frame_correction=True,
):
    main_frames = [init_img.convert("RGB")]
    for i in range(outpaint_steps):
        print_out = (
            "Outpaint step: "
            + str(i + 1)
            + " / "
            + str(outpaint_steps)
            + " Seed: "
            + str(seed)
        )
        print(print_out)
        current_image = main_frames[-1]
        prev_image_fix = current_image
        # save2Collect(prev_image_fix, out_config, f"prev_image_fix_{i}.png")

        prev_image = shrink_and_paste_on_blank(current_image, mask_width, mask_height)
        # save2Collect(prev_image, out_config, f"prev_image_{i}.png")

        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")
        # save2Collect(mask_image, out_config, f"mask_image_{i}.png")

        if custom_exit_image and ((i + 1) == outpaint_steps):
            current_image = custom_exit_image.resize(
                (width, height), resample=Image.LANCZOS
            )
            main_frames.append(current_image.convert("RGB"))
            print("using Custom Exit Image")
            # save2Collect(current_image, out_config, f"exit_img.png")
        else:
            pr = prompts[max(k for k in prompts.keys() if k <= i)]
            processed, newseed = renderImg2Img(
                f"{common_prompt_pre}\n{pr}\n{common_prompt_suf}".strip(),
                negative_prompt,
                sampler,
                num_inference_steps,
                guidance_scale,
                seed,
                width,
                height,
                current_image,
                mask_image,
                inpainting_denoising_strength,
                inpainting_mask_blur,
                inpainting_fill_mode,
                inpainting_full_res,
                inpainting_padding,
            )
            if len(processed.images) > 0:
                main_frames.append(processed.images[0].convert("RGB"))
            seed = newseed
        if frame_correction and inpainting_mask_blur > 0:
            corrected_frame = crop_inner_image(
                main_frames[i + 1], mask_width, mask_height
            )
            save2Collect(current_image, out_config, f"corrected_{i}")
            main_frames[i] = corrected_frame

        # else TEST pasting differance
        # current_image.paste(prev_image, mask=prev_image)
    frames2Collect(main_frames, out_config)
    print(out_config)
    print("length: ", len(main_frames))
    return main_frames


def create_zoom(
    common_prompt_pre,
    prompts_array,
    common_prompt_suf,
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
            common_prompt_pre,
            prompts_array,
            common_prompt_suf,
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


def prepare_output_path():
    isCollect = shared.opts.data.get("infzoom_collectAllResources", False)
    output_path = shared.opts.data.get("infzoom_outpath", "output")

    save_path = os.path.join(
        output_path, shared.opts.data.get("infzoom_outSUBpath", "infinite-zooms")
    )

    if isCollect:
        save_path = os.path.join(save_path, "iz_collect" + str(int(time.time())))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    video_filename = os.path.join(
        save_path, "infinite_zoom_" + str(int(time.time())) + ".mp4"
    )

    return {
        "isCollect": isCollect,
        "save_path": save_path,
        "video_filename": video_filename,
    }


def save2Collect(img, out_config, name):
    if out_config["isCollect"]:
        img.save(f'{out_config["save_path"]}/{name}.png')


def frame2Collect(all_frames, out_config):
    save2Collect(all_frames[-1], out_config, f"frame_{len(all_frames)}")


def frames2Collect(all_frames, out_config):
    for i, f in enumerate(all_frames):
        save2Collect(f, out_config, f"frame_{i}")


def crop_inner_image(outpainted_img, width_offset, height_offset):
    width, height = outpainted_img.size

    center_x, center_y = int(width / 2), int(height / 2)

    # Crop the image to the center
    cropped_img = outpainted_img.crop(
        (
            center_x - width_offset,
            center_y - height_offset,
            center_x + width_offset,
            center_y + height_offset,
        )
    )
    prev_step_img = cropped_img.resize((width, height), resample=Image.LANCZOS)
    # resized_img = resized_img.filter(ImageFilter.SHARPEN)

    return prev_step_img


def create_zoom_single(
    common_prompt_pre,
    prompts_array,
    common_prompt_suf,
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
    out_config = prepare_output_path()

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

    current_image = Image.new(mode="RGBA", size=(width, height))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    current_image = current_image.convert("RGB")
    current_seed = seed

    if custom_init_image:
        current_image = custom_init_image.resize(
            (width, height), resample=Image.LANCZOS
        )
        # save2Collect(current_image, out_config, f"init_img.png")
        print("using Custom Initial Image")

    else:
        load_model_from_setting(
            "infzoom_txt2img_model", progress, "Loading Model for txt2img: "
        )

        pr = prompts[min(k for k in prompts.keys() if k >= 0)]
        processed, newseed = renderTxt2Img(
            f"{common_prompt_pre}\n{pr}\n{common_prompt_suf}".strip(),
            negative_prompt,
            sampler,
            num_inference_steps,
            guidance_scale,
            current_seed,
            width,
            height,
        )
        if len(processed.images) > 0:
            current_image = processed.images[0]
            # save2Collect(current_image, out_config, f"txt2img.png")
        current_seed = newseed

    mask_width = math.trunc(width / 4)  # was initially 512px => 128px
    mask_height = math.trunc(height / 4)  # was initially 512px => 128px

    num_interpol_frames = round(video_frame_rate * zoom_speed)

    all_frames = []

    if upscale_do and progress:
        progress(0, desc="upscaling inital image")

    load_model_from_setting(
        "infzoom_inpainting_model", progress, "Loading Model for inpainting/img2img: "
    )
    main_frames = outpaint_steps(
        width,
        height,
        common_prompt_pre,
        common_prompt_suf,
        prompts,
        negative_prompt,
        seed,
        sampler,
        num_inference_steps,
        guidance_scale,
        inpainting_denoising_strength,
        inpainting_mask_blur,
        inpainting_fill_mode,
        inpainting_full_res,
        inpainting_padding,
        current_image,
        num_outpainting_steps,
        out_config,
        mask_width,
        mask_height,
        custom_exit_image,
    )
    for i in range(len(main_frames) - 1):
        # TODO: fix upscale
        # all_frames.append(
        #     do_upscaleImg(
        #         prev_image_fix.convert("RGB"), upscale_do, upscaler_name, upscale_by
        #     )
        #     if upscale_do
        #     else prev_image_fix.convert("RGB")
        # )
        # interpolation steps between 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            current_image = main_frames[i + 1]
            interpol_image = current_image
            # save2Collect(interpol_image, out_config, f"interpol_img_{i}_{j}].png")

            interpol_width = round(
                (
                    1
                    - (1 - 2 * mask_width / width)
                    ** (1 - (j + 1) / num_interpol_frames)
                )
                * width
                / 2
            )

            interpol_height = round(
                (
                    1
                    - (1 - 2 * mask_height / height)
                    ** (1 - (j + 1) / num_interpol_frames)
                )
                * height
                / 2
            )

            interpol_image = interpol_image.crop(
                (
                    interpol_width,
                    interpol_height,
                    width - interpol_width,
                    height - interpol_height,
                )
            )
            # save2Collect(interpol_image, out_config, f"interpol_crop_{i}_{j}.png")

            interpol_image = interpol_image.resize((width, height))
            # save2Collect(interpol_image, out_config, f"interpol_resize_{i}_{j}.png")

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round(
                (1 - (width - 2 * mask_width) / (width - 2 * interpol_width))
                / 2
                * width
            )

            interpol_height2 = round(
                (1 - (height - 2 * mask_height) / (height - 2 * interpol_height))
                / 2
                * height
            )

            prev_image_fix_crop = shrink_and_paste_on_blank(
                main_frames[i], interpol_width2, interpol_height2
            )
            # save2Collect(prev_image_fix, out_config, f"prev_image_fix_crop_{i}_{j}.png")

            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)
            # save2Collect(interpol_image, out_config, f"interpol_prevcrop_{i}_{j}.png")

            if upscale_do and progress:
                progress(((i + 1) / num_outpainting_steps), desc="upscaling interpol")

            all_frames.append(
                do_upscaleImg(interpol_image, upscale_do, upscaler_name, upscale_by)
                if upscale_do
                else interpol_image
            )

        if upscale_do and progress:
            progress(((i + 1) / num_outpainting_steps), desc="upscaling current")

        all_frames.append(
            do_upscaleImg(current_image, upscale_do, upscaler_name, upscale_by)
            if upscale_do
            else current_image
        )

    # frames2Collect(all_frames, out_config)

    write_video(
        out_config["video_filename"],
        all_frames,
        video_frame_rate,
        video_zoom_mode,
        int(video_start_frame_dupe_amount),
        int(video_last_frame_dupe_amount),
    )

    return (
        out_config["video_filename"],
        processed.images,
        processed.js(),
        plaintext_to_html(processed.info),
        plaintext_to_html(""),
    )
