import math, time, os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from modules.ui import plaintext_to_html
import modules.shared as shared
from modules.paths_internal import script_path
from .helpers import (
    fix_env_Path_ffprobe,
    closest_upper_divisible_by_eight,
    load_model_from_setting,
    do_upscaleImg,
)
from .sd_helpers import renderImg2Img, renderTxt2Img
from .image import shrink_and_paste_on_blank
from .video import write_video

def crop_fethear_ellipse(image, feather_margin=30, width_offset=0, height_offset=0):
    # Create a blank mask image with the same size as the original image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Calculate the ellipse's bounding box
    ellipse_box = (
        width_offset,
        height_offset,
        image.width - width_offset,
        image.height - height_offset,
    )

    # Draw the ellipse on the mask
    draw.ellipse(ellipse_box, fill=255)

    # Apply the mask to the original image
    result = Image.new("RGBA", image.size)
    result.paste(image, mask=mask)

    # Crop the resulting image to the ellipse's bounding box
    cropped_image = result.crop(ellipse_box)

    # Create a new mask image with a black background (0)
    mask = Image.new("L", cropped_image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw an ellipse on the mask image
    draw.ellipse(
        (
            0 + feather_margin,
            0 + feather_margin,
            cropped_image.width - feather_margin,
            cropped_image.height - feather_margin,
        ),
        fill=255,
        outline=0,
    )

    # Apply a Gaussian blur to the mask image
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_margin / 2))
    cropped_image.putalpha(mask)
    res = Image.new(cropped_image.mode, (image.width, image.height))
    paste_pos = (
        int((res.width - cropped_image.width) / 2),
        int((res.height - cropped_image.height) / 2),
    )
    res.paste(cropped_image, paste_pos)

    return res


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
    overmask,
    frame_correction=True,  # TODO: add frame_Correction in UI
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
        current_image = shrink_and_paste_on_blank(
            current_image, mask_width, mask_height
        )

        mask_image = np.array(current_image.resize((current_image.width-overmask, current_image.height-overmask)))[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")
        # create mask (black image with white mask_width width edges)

        if custom_exit_image and ((i + 1) == outpaint_steps):
            current_image = custom_exit_image.resize(
                (width, height), resample=Image.LANCZOS
            )
            main_frames.append(current_image.convert("RGB"))
            # print("using Custom Exit Image")
            save2Collect(current_image, out_config, f"exit_img.png")
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
                save2Collect(processed.images[0], out_config, f"outpain_step_{i}.png")
            seed = newseed
            # TODO: seed behavior

        if frame_correction and inpainting_mask_blur > 0:
            corrected_frame = crop_inner_image(
                main_frames[i + 1], mask_width, mask_height
            )

            enhanced_img = crop_fethear_ellipse(
                main_frames[i],
                30,
                inpainting_mask_blur / 3 // 2,
                inpainting_mask_blur / 3 // 2,
            )
            save2Collect(main_frames[i], out_config, f"main_frame_{i}")
            save2Collect(enhanced_img, out_config, f"main_frame_enhanced_{i}")
            corrected_frame.paste(enhanced_img, mask=enhanced_img)
            main_frames[i] = corrected_frame
        # else :TEST
        # current_image.paste(prev_image, mask=prev_image)
    return main_frames, processed


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
    inpainting_mask_blur,
    inpainting_fill_mode,
    zoom_speed,
    seed,
    outputsizeW,
    outputsizeH,
    batchcount,
    sampler,
    upscale_do,
    upscaler_name,
    upscale_by,
    overmask,
    inpainting_denoising_strength=1,
    inpainting_full_res=0,
    inpainting_padding=0,
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
            inpainting_mask_blur,
            inpainting_fill_mode,
            zoom_speed,
            seed,
            outputsizeW,
            outputsizeH,
            sampler,
            upscale_do,
            upscaler_name,
            upscale_by,
            overmask,
            inpainting_denoising_strength,
            inpainting_full_res,
            inpainting_padding,
            progress,
        )
    return result


def prepare_output_path():
    isCollect = shared.opts.data.get("infzoom_collectAllResources", False)
    output_path = shared.opts.data.get("infzoom_outpath", "outputs")

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
    inpainting_mask_blur,
    inpainting_fill_mode,
    zoom_speed,
    seed,
    outputsizeW,
    outputsizeH,
    sampler,
    upscale_do,
    upscaler_name,
    upscale_by,
    overmask,
    inpainting_denoising_strength,
    inpainting_full_res,
    inpainting_padding,
    progress,
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
        save2Collect(current_image, out_config, f"init_custom.png")

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
            save2Collect(current_image, out_config, f"init_txt2img.png")
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
    main_frames, processed = outpaint_steps(
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
        overmask
    )
    all_frames.append(
        do_upscaleImg(main_frames[0], upscale_do, upscaler_name, upscale_by)
        if upscale_do
        else main_frames[0]
    )
    
    for i in range(len(main_frames) - 1):
        # interpolation steps between 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            current_image = main_frames[i + 1]
            interpol_image = current_image
            save2Collect(interpol_image, out_config, f"interpol_img_{i}_{j}].png")

            interpol_width = math.ceil(
                (
                    1
                    - (1 - 2 * mask_width / width)
                    ** (1 - (j + 1) / num_interpol_frames)
                )
                * width
                / 2
            )

            interpol_height = math.ceil(
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

            interpol_image = interpol_image.resize((width, height))
            save2Collect(interpol_image, out_config, f"interpol_resize_{i}_{j}.png")

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = math.ceil(
                (1 - (width - 2 * mask_width) / (width - 2 * interpol_width))
                / 2
                * width
            )

            interpol_height2 = math.ceil(
                (1 - (height - 2 * mask_height) / (height - 2 * interpol_height))
                / 2
                * height
            )

            prev_image_fix_crop = shrink_and_paste_on_blank(
                main_frames[i], interpol_width2, interpol_height2
            )

            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)
            save2Collect(interpol_image, out_config, f"interpol_prevcrop_{i}_{j}.png")

            if upscale_do and progress:
                progress(((i + 1) / num_outpainting_steps), desc="upscaling interpol")

            all_frames.append(interpol_image)

        all_frames.append(current_image)

    frames2Collect(all_frames, out_config)

    write_video(
        out_config["video_filename"],
        all_frames,
        video_frame_rate,
        video_zoom_mode,
        int(video_start_frame_dupe_amount),
        int(video_last_frame_dupe_amount),
    )
    print("Video saved in: " + os.path.join(script_path, out_config["video_filename"]))
    return (
        out_config["video_filename"],
        main_frames,
        processed.js(),
        plaintext_to_html(processed.info),
        plaintext_to_html(""),
    )
