import math, time, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
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
    upscalerinterpol_name,
    upscale_by,
    exitgamma,
    maskwidth_slider,
    maskheight_slider,
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
            upscalerinterpol_name,
            upscale_by,
            exitgamma,
            maskwidth_slider,
            maskheight_slider,
            progress
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
    upscalerinterpol_name,
    upscale_by,
    exitgamma,
    maskwidth_slider,
    maskheight_slider,
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
    prompt_images = {}

    for x in prompts_array:
        try:
            key = int(x[0])
            value = str(x[1])
            file_loc = str(x[2])
            prompts[key] = value
            prompt_images[key] = file_loc
        except ValueError:
            pass
    assert len(prompts_array) > 0, "prompts is empty"
    print(str(len(prompts)) + " prompts found")
    print(str(len(prompt_images)) + " prompts Images found")

    width = closest_upper_divisible_by_eight(outputsizeW)
    height = closest_upper_divisible_by_eight(outputsizeH)

    current_image = Image.new(mode="RGBA", size=(width, height))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    current_image = current_image.convert("RGB")
    current_seed = seed
    extra_frames = 0

    if custom_init_image:
        current_image = custom_init_image.resize(
            (width, height), resample=Image.LANCZOS
        )
        print("using Custom Initial Image")
    else:
        if prompt_images[min(k for k in prompt_images.keys() if k >= 0)] == "":
            load_model_from_setting(
                "infzoom_txt2img_model", progress, "Loading Model for txt2img: "
            )

            processed, current_seed = renderTxt2Img(
                prompts[min(k for k in prompts.keys() if k >= 0)],
                negative_prompt,
                sampler,
                num_inference_steps,
                guidance_scale,
                current_seed,
                width,
                height,
            )
            current_image = processed.images[0]
        else:
            current_image = Image.open(prompt_images[min(k for k in prompt_images.keys() if k >= 0)]).resize(
                (width, height), resample=Image.LANCZOS
            )

#    if custom_exit_image and ((i + 1) == num_outpainting_steps):
#        mask_width = 4  # fade out whole interpol
#        mask_height =4  # 
#        mask_width  = width*(20//30)  # fade out whole interpol
#        mask_height = height*(20//30)  # 
#    else:
    mask_width = math.trunc(width / 4)  # was initially 512px => 128px
    mask_height = math.trunc(height / 4)  # was initially 512px => 128px

    num_interpol_frames = round(video_frame_rate * zoom_speed)

    all_frames = []

    if upscale_do and progress:
        progress(0, desc="upscaling inital image")

    all_frames.append(
        do_upscaleImg(current_image, upscale_do, upscaler_name, upscale_by)
        if upscale_do
        else current_image
    )

    load_model_from_setting("infzoom_inpainting_model", progress, "Loading Model for inpainting/img2img: " )

    if custom_exit_image:
        extra_frames += 2

    # setup filesystem paths
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

    for i in range(num_outpainting_steps + extra_frames):
        print_out = (
            "Outpaint step: "
            + str(i + 1)
            + " / "
            + str(num_outpainting_steps + extra_frames)
            + " Seed: "
            + str(current_seed)
        )
        print(print_out)
        if progress:
            progress(((i + 1) / num_outpainting_steps), desc=print_out)

        if custom_exit_image and ((i + 1) == num_outpainting_steps):
            mask_width=round(width*maskwidth_slider)
            mask_height=round(height*maskheight_slider)
            
            # 30 fps@ maskw 0.25 => 30
            # normalize to default speed of 30 fps for 0.25 mask factor
            num_interpol_frames = round(num_interpol_frames * (1 + (max(maskheight_slider,maskwidth_slider)/0.5) * exitgamma))

        prev_image_fix = current_image
        prev_image = shrink_and_paste_on_blank(current_image, mask_width, mask_height)
        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")

        # Custom and specified images work like keyframes
        if custom_exit_image and (i + 1) >= (num_outpainting_steps + extra_frames):
            current_image = custom_exit_image.resize(
                (width, height), resample=Image.LANCZOS
            )
            print("using Custom Exit Image")
        else:
            if prompt_images[max(k for k in prompt_images.keys() if k <= (i + 1))] == "":
                processed, current_seed = renderImg2Img(
                    prompts[max(k for k in prompts.keys() if k <= (i + 1))],
                    negative_prompt,
                    sampler,
                    num_inference_steps,
                    guidance_scale,
                    current_seed,
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
                current_image = processed.images[0]
                # only paste previous image when generating a new image
                current_image.paste(prev_image, mask=prev_image)
            else:
                current_image = Image.open(prompt_images[max(k for k in prompt_images.keys() if k <= (i + 1))]).resize(
                    (width, height), resample=Image.LANCZOS
                )

        

        # interpolation steps between 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_image

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

            interpol_image = interpol_image.resize((width, height))

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

            if custom_exit_image and ((i + 1) == num_outpainting_steps):
                opacity = 1 - ((j+1)/num_interpol_frames )
            else: opacity = 1

            prev_image_fix_crop = shrink_and_paste_on_blank(
                prev_image_fix, interpol_width2, interpol_height2,
                opacity=opacity
            )

            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

            # exit image: from now we see the last prompt on the exit image 
            if custom_exit_image and ((i + 1) == num_outpainting_steps):

                mask_img = Image.new("L", (width,height), 0)
                in_center_x = interpol_image.width/2
                in_center_y = interpol_image.height/2
                
                # Draw a circular brush on the mask image with 64px diameter and 8px softness
                draw = ImageDraw.Draw(mask_img)
                brush_size = 64
                brush_softness = 8
                brush = Image.new("L", (brush_size, brush_size), 255)
                draw_brush = ImageDraw.Draw(brush)
                draw_brush.ellipse((brush_softness, brush_softness, brush_size-brush_softness, brush_size-brush_softness), fill=255, outline=None)
                brush = brush.filter(ImageFilter.GaussianBlur(radius=brush_softness))
                brush_width, brush_height = brush.size

                # Draw the rectangular frame on the mask image using the circular brush
                frame_width = width-2*interpol_width2
                frame_height = height-2*interpol_height2
                frame_left = in_center_x - (frame_width // 2)
                frame_top = in_center_y - (frame_height // 2)
                frame_right = frame_left + frame_width
                frame_bottom = frame_top + frame_height
                draw.ellipse((frame_left, frame_top, frame_left+brush_width, frame_top+brush_height), fill=255, outline=None)
                draw.ellipse((frame_right-brush_width, frame_top, frame_right, frame_top+brush_height), fill=255, outline=None)
                draw.ellipse((frame_left, frame_bottom-brush_height, frame_left+brush_width, frame_bottom), fill=255, outline=None)
                draw.ellipse((frame_right-brush_width, frame_bottom-brush_height, frame_right, frame_bottom), fill=255, outline=None)

                draw.rectangle((max(0,frame_left-brush_size/2), max(0,frame_top+brush_size/2), max(0,frame_right-brush_size/2), max(0,frame_bottom-brush_size/2)), fill=255)

                # inner rect, now we have a bordermask
                draw.rectangle((max(0,frame_left+brush_size/2), max(0,frame_top-brush_size/2), max(0,frame_right+brush_size/2), max(0,frame_bottom+brush_size/2)), fill=0)

                # Blur the mask image to soften the edges
                #mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=8))
                #mask_img = ImageOps.invert(mask_img)
                #mask_img.save(output_path+os.pathsep+"Mask"+str(int(time.time()))+".png")
                """processed, newseed = renderImg2Img(
                                prompts[max(k for k in prompts.keys() if k <= i)],
                                negative_prompt,
                                sampler,
                                num_inference_steps,
                                guidance_scale,
                                current_seed,
                                width,
                                height,
                                interpol_image,
                                mask_img,
                                inpainting_denoising_strength,
                                inpainting_mask_blur,

                                inpainting_fill_mode,
                                inpainting_full_res,
                                inpainting_padding,
                            )
                #interpol_image = processed.images[0]
                """

            if upscale_do and progress:
                progress(((i + 1) / num_outpainting_steps), desc="upscaling interpol")

            all_frames.append(
                do_upscaleImg(interpol_image, upscale_do, upscalerinterpol_name, upscale_by)
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
