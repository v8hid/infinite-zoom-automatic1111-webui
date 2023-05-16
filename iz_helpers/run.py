import math, time, os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from modules.ui import plaintext_to_html
import modules.shared as shared
from modules.processing import Processed, StableDiffusionProcessing
from modules.paths_internal import script_path
from .helpers import (
    fix_env_Path_ffprobe,
    closest_upper_divisible_by_eight,
    load_model_from_setting,
    do_upscaleImg,value_to_bool
)
from .sd_helpers import renderImg2Img, renderTxt2Img
from .image import shrink_and_paste_on_blank, open_image, apply_alpha_mask, draw_gradient_ellipse, resize_and_crop_image, crop_fethear_ellipse, crop_inner_image
from .video import write_video

def outpaint_steps(
    width,
    height,
    common_prompt_pre,
    common_prompt_suf,
    prompts,
    prompt_images,
    prompt_alpha_mask_images,
    prompt_image_is_keyframe,
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
    frame_correction=True,  # TODO: add frame_Correction in UI
    blend_gradient_size = 61,
):
    main_frames = [init_img.convert("RGBA")]
    prev_image = init_img.convert("RGBA")
    exit_img = custom_exit_image.convert("RGBA") if custom_exit_image else None

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

        # shrink image to mask size
        current_image = shrink_and_paste_on_blank(
            current_image, mask_width, mask_height
        )

        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image)        
        # create mask (black image with white mask_width width edges)

        #keyframes are not inpainted
        paste_previous_image = not prompt_image_is_keyframe[(i + 1)]
        print(f"paste_prev_image: {paste_previous_image} {i} {i + 1}")

        if custom_exit_image and ((i + 1) == outpaint_steps):
            current_image = resize_and_crop_image(custom_exit_image, width, height).convert("RGBA")
            exit_img = current_image
            print("using Custom Exit Image")
            save2Collect(current_image, out_config, f"exit_img.png")

            paste_previous_image = False
        else:
            if prompt_images[max(k for k in prompt_images.keys() if k <= (i + 1))] == "":
                pr = prompts[max(k for k in prompts.keys() if k <= i)]
                processed, seed = renderImg2Img(
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
                    main_frames.append(processed.images[0].convert("RGBA"))
                    save2Collect(processed.images[0], out_config, f"outpain_step_{i}.png")

                #paste_previous_image = True
            else:
                # use prerendered image, known as keyframe. Resize to target size
                print(f"image {i + 1} is a keyframe: {not paste_previous_image}")
                current_image = open_image(prompt_images[(i + 1)])
                current_image = resize_and_crop_image(current_image, width, height).convert("RGBA")

                # if keyframe is last frame, use it as exit image
                if (not paste_previous_image) and ((i + 1) == outpaint_steps):
                    exit_img = current_image
                    print("using keyframe as exit image")
                else:
                    main_frames.append(current_image)
                save2Collect(current_image, out_config, f"key_frame_{i + 1}.png")           

        #seed = newseed
        # TODO: seed behavior

        # paste previous image on top of current image            
        if frame_correction and (inpainting_mask_blur > 0):
            #if 0 <= (i + 1) < len(main_frames):
            if paste_previous_image and i > 0:
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
        else: #TEST
            # paste current image with alpha layer on previous image to merge : paste on i         
            if paste_previous_image and i > 0:
                # apply predefined or generated alpha mask to current image: 
                # current image must be redefined as most current image in frame stack
                # use previous image alpha mask if available
                if prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if k <= (i + 1))] != "":
                    current_image_amask = open_image(prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if  k <= (i + 1))])
                else:
                    current_image_gradient_ratio = (blend_gradient_size / 100) #max((min(current_image.width/current_image.height,current_image.height/current_image.width) * 0.925),0.1)
                    current_image_amask = draw_gradient_ellipse(main_frames[i + 1].width, main_frames[i + 1].height, current_image_gradient_ratio, 0.0, 2.5)
                current_image = apply_alpha_mask(main_frames[i + 1], current_image_amask)

                #handle previous image alpha layer
                #prev_image = (main_frames[i] if main_frames[i] else main_frames[0])
                ## apply available alpha mask of previous image (inverted)
                if prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if k <= (i))] != "":
                    prev_image_amask = open_image(prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if  k <= (i))])
                else:
                    prev_image_gradient_ratio = (blend_gradient_size / 100) #max((min(current_image.width/current_image.height,current_image.height/current_image.width) * 0.925),0.1)
                    prev_image_amask = draw_gradient_ellipse(prev_image.width, prev_image.height, prev_image_gradient_ratio, 0.0, 2.5)
                #prev_image = apply_alpha_mask(prev_image, prev_image_amask, invert = True)

                # merge previous image with current image
                corrected_frame = crop_inner_image(
                    current_image, mask_width, mask_height
                )
                prev = Image.new(prev_image.mode, (width, height), (255,255,255,255))
                prev.paste(apply_alpha_mask(main_frames[i], prev_image_amask))
                corrected_frame.paste(prev, mask=prev)
                
                main_frames[i] = corrected_frame
                save2Collect(corrected_frame, out_config, f"main_frame_gradient_{i + 0}")
            
    if exit_img is not None:
        main_frames.append(exit_img)

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
    blend_image,
    blend_mode,
    blend_gradient_size,
    blend_invert_do,
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
            blend_image,
            blend_mode,
            blend_gradient_size,
            blend_invert_do,
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
    blend_image,
    blend_mode,
    blend_gradient_size,
    blend_invert_do,
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
    prompt_images = {}
    prompt_alpha_mask_images = {}
    prompt_image_is_keyframe = {}

    for x in prompts_array:
        try:
            key = int(x[0])
            value = str(x[1])
            file_loc = str(x[2])
            alpha_mask_loc = str(x[3])
            is_keyframe = bool(x[4])
            prompts[key] = value
            prompt_images[key] = file_loc
            prompt_alpha_mask_images[key] = alpha_mask_loc
            prompt_image_is_keyframe[key] = value_to_bool(is_keyframe)
        except ValueError:
            pass

    assert len(prompts_array) > 0, "prompts is empty"
    print(str(len(prompts)) + " prompts found")
    print(str(len([value for value in prompt_images.values() if value != ""])) + " prompt Images found")
    print(str(len([value for value in prompt_alpha_mask_images.values() if value != ""])) + " prompt Alpha Masks found")

    width = closest_upper_divisible_by_eight(outputsizeW)
    height = closest_upper_divisible_by_eight(outputsizeH)

    current_image = Image.new(mode="RGBA", size=(width, height))
    #mask_image = np.array(current_image)[:, :, 3]
    #mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    #current_image = current_image.convert("RGB")
    current_seed = seed
    extra_frames = 0

    if custom_init_image:
        current_image = resize_and_crop_image(custom_init_image, width, height)
        print("using Custom Initial Image")
        save2Collect(current_image, out_config, f"init_custom.png")
        #processed = Processed(StableDiffusionProcessing(),images_list=[current_image], seed=current_seed, info="init_custom image")
    else:
        if prompt_images[min(k for k in prompt_images.keys() if k >= 0)] == "":
            load_model_from_setting(
                "infzoom_txt2img_model", progress, "Loading Model for txt2img: "
            )
            pr = prompts[min(k for k in prompts.keys() if k >= 0)]
            processed, current_seed = renderTxt2Img(
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
        else:
            print("using image 0 as Initial keyframe")
            current_image = open_image(prompt_images[min(k for k in prompt_images.keys() if k >= 0)])
            current_image = resize_and_crop_image(current_image, width, height)
            save2Collect(current_image, out_config, f"init_custom.png")
            #processed = Processed(StableDiffusionProcessing(),images_list=[current_image], seed=current_seed, info="prompt_0 image")

    mask_width = math.trunc(width / 4)  # was initially 512px => 128px
    mask_height = math.trunc(height / 4)  # was initially 512px => 128px

    num_interpol_frames = round(video_frame_rate * zoom_speed)

    all_frames = []

    if upscale_do and progress:
        progress(0, desc="upscaling inital image")

    load_model_from_setting(
        "infzoom_inpainting_model", progress, "Loading Model for inpainting/img2img: "
    )

    if custom_exit_image:
        extra_frames += 1

    main_frames, processed = outpaint_steps(
        width,
        height,
        common_prompt_pre,
        common_prompt_suf,
        prompts,
        prompt_images,
        prompt_alpha_mask_images,
        prompt_image_is_keyframe,
        negative_prompt,
        current_seed,
        sampler,
        num_inference_steps,
        guidance_scale,
        inpainting_denoising_strength,
        inpainting_mask_blur,
        inpainting_fill_mode,
        inpainting_full_res,
        inpainting_padding,
        current_image,
        num_outpainting_steps + extra_frames,
        out_config,
        mask_width,
        mask_height,
        custom_exit_image,
        False,
        blend_gradient_size,
    )

    #for k in range(len(main_frames)):
        #print(str(f"Frame {k} : {main_frames[k]}"))
        #resize_and_crop_image(main_frames[k], width, height)        

    all_frames.append(
        do_upscaleImg(main_frames[0], upscale_do, upscaler_name, upscale_by)
        if upscale_do
        else main_frames[0]
    )
    for i in range(len(main_frames) - 1):
        print(f"processing frame {i}")

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

            all_frames.append(
                do_upscaleImg(interpol_image, upscale_do, upscaler_name, upscale_by)
                if upscale_do
                else interpol_image
            )

        if upscale_do and progress:
            progress(((i + 1) / num_outpainting_steps), desc="upscaling current")

        all_frames.append(
            #do_upscaleImg(current_image, upscale_do, upscaler_name, upscale_by)
            #if upscale_do
            #else 
            current_image
        )

    frames2Collect(all_frames, out_config)

    write_video(
        out_config["video_filename"],
        all_frames,
        video_frame_rate,
        video_zoom_mode,
        int(video_start_frame_dupe_amount),
        int(video_last_frame_dupe_amount),
        num_interpol_frames,
        blend_invert_do,
        blend_image,
        blend_mode,
        blend_gradient_size,
    )
    print("Video saved in: " + os.path.join(script_path, out_config["video_filename"]))
    return (
        out_config["video_filename"],
        main_frames,
        processed.js(),
        plaintext_to_html(processed.info),
        plaintext_to_html(""),
    )
