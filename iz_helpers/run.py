import math, time, os
import numpy as np
from scipy.signal import savgol_filter
from typing import Callable
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageColor
from modules.ui import plaintext_to_html
import modules.shared as shared
from modules.processing import Processed, StableDiffusionProcessing
from modules.paths_internal import script_path
from .helpers import (
    fix_env_Path_ffprobe,
    closest_upper_divisible_by_eight,
    load_model_from_setting,
    do_upscaleImg,value_to_bool, find_ffmpeg_binary
)
from .sd_helpers import renderImg2Img, renderTxt2Img
from .image import shrink_and_paste_on_blank, open_image, apply_alpha_mask, draw_gradient_ellipse, resize_and_crop_image, crop_fethear_ellipse, crop_inner_image
from .video import write_video, add_audio_to_video, ContinuousVideoWriter
from .InfZoomConfig import InfZoomConfig

class InfZoomer:
    def __init__(self, config: InfZoomConfig) -> None:
        self.C = config
        self.prompts = {}
        self.prompt_images = {}
        self.prompt_alpha_mask_images = {}
        self.prompt_image_is_keyframe = {}
        self.main_frames = []
        self.out_config = {}

        for x in self.C.prompts_array:
            try:
                key = int(x[0])
                value = str(x[1])
                file_loc = str(x[2])
                alpha_mask_loc = str(x[3])
                is_keyframe = bool(x[4])
                self.prompts[key] = value
                self.prompt_images[key] = file_loc
                self.prompt_alpha_mask_images[key] = alpha_mask_loc
                self.prompt_image_is_keyframe[key] = value_to_bool(is_keyframe)
            except ValueError:
                pass

        assert len(self.C.prompts_array) > 0, "prompts is empty"

        fix_env_Path_ffprobe()
        self.out_config = self.prepare_output_path()

        self.current_seed = self.C.seed

        # knowing the mask_height and desired outputsize find a compromise due to align 8 contraint of diffuser
        self.width = closest_upper_divisible_by_eight(self.C.outputsizeW)
        self.height = closest_upper_divisible_by_eight(self.C.outputsizeH)

        if self.width > self.height:
            self.mask_width  = self.C.outpaint_amount_px 
            self.mask_height = math.trunc(self.C.outpaint_amount_px * self.height/self.width)  
        else:
            self.mask_height  = self.C.outpaint_amount_px  
            self.mask_width  = math.trunc(self.C.outpaint_amount_px * self.width/self.height)  

        # here we leave slightly the desired ratio since if size+2*mask_size % 8 != 0
        # distribute "aligning pixels" to the mask size equally. 
        # only consider mask_size since image size is alread 8-aligned
        self.mask_width -= self.mask_width % 4
        self.mask_height -= self.mask_height % 4

        assert 0 == (2*self.mask_width+self.width) % 8
        assert 0 == (2*self.mask_height+self.height) % 8

        print (f"Adapted sizes for diffusers to: {self.width}x{self.height}+mask:{self.mask_width}x{self.mask_height}. New ratio: {(self.width+self.mask_width)/(self.height+self.mask_height)} ")

        self.num_interpol_frames = round(self.C.video_frame_rate * self.C.zoom_speed) - 1 # keyframe not to be interpolated

        if (self.C.outpaintStrategy == "Corners"):
            self.fnOutpaintMainFrames = self.outpaint_steps_cornerStrategy
            self.fnInterpolateFrames = self.interpolateFramesOuterZoom
        elif (self.C.outpaintStrategy == "Center"):
           self.fnOutpaintMainFrames = self.outpaint_steps_v8hid
           self.fnInterpolateFrames = self.interpolateFramesSmallCenter
        else:
            raise ValueError("Unsupported outpaint strategy in Infinite Zoom")

        self.outerZoom = True    # scale from overscan to target viewport

    # object properties, different from user input config
    out_config = {}
    prompts = {}
    main_frames:Image = []

    outerZoom: bool
    mask_width: int
    mask_height: int
    current_seed: int
    contVW: ContinuousVideoWriter
    fnOutpaintMainFrames: Callable
    fnInterpolateFrames: Callable

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
    blend_gradient_size = 61
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
                    int(num_inference_steps),
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
                    # apply predefined or generated alpha mask to current image:
                    if prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if k <= (i + 1))] != "":
                        current_image_amask = open_image(prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if  k <= (i + 1))])
                    else:
                        current_image_gradient_ratio = (blend_gradient_size / 100)
                        current_image_amask = draw_gradient_ellipse(current_image.width, current_image.height, current_image_gradient_ratio, 0.0, 2.5)
                    current_image = apply_alpha_mask(current_image, current_image_amask)
                    main_frames.append(current_image)
                save2Collect(current_image, out_config, f"key_frame_{i + 1}.png")

        #seed = newseed
        # TODO: seed behavior

        # paste current image with alpha layer on previous image to merge : paste on i         
        if paste_previous_image and i > 0:
            # apply predefined or generated alpha mask to current image: 
            # current image must be redefined as most current image in frame stack
            # use previous image alpha mask if available
            if prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if k <= (i + 1))] != "":
                current_image_amask = open_image(prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if  k <= (i + 1))])
            else:
                current_image_gradient_ratio = (blend_gradient_size / 100)
                current_image_amask = draw_gradient_ellipse(main_frames[i + 1].width, main_frames[i + 1].height, current_image_gradient_ratio, 0.0, 2.5)
            current_image = apply_alpha_mask(main_frames[i + 1], current_image_amask)

            #handle previous image alpha layer
            #prev_image = (main_frames[i] if main_frames[i] else main_frames[0])
            ## apply available alpha mask of previous image (inverted)
            if prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if k <= (i))] != "":
                prev_image_amask = open_image(prompt_alpha_mask_images[max(k for k in prompt_alpha_mask_images.keys() if  k <= (i))])
            else:
                prev_image_gradient_ratio = (blend_gradient_size / 100)
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
    video_ffmpeg_opts,
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
    outpaintStrategy,
    outpaint_amount_px,
    blend_image,
    blend_mode,
    blend_gradient_size,
    blend_invert_do,
    blend_color,
    audio_filename=None,
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
            int(num_inference_steps),
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
            blend_color,
            inpainting_denoising_strength,
            inpainting_full_res,
            inpainting_padding,
            progress,
            audio_filename
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
    blend_color,
    inpainting_denoising_strength,
    inpainting_full_res,
    inpainting_padding,
    progress,
    audio_filename = None
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
        int(num_inference_steps),
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
        blend_gradient_size
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
        ImageColor.getcolor(blend_color, "RGBA"),
    )
    if audio_filename is not None:
        out_config["video_filename"] = add_audio_to_video(out_config["video_filename"], audio_filename, str.replace(out_config["video_filename"], ".mp4", "_audio.mp4"), find_ffmpeg_binary())

    print("Video saved in: " + os.path.join(script_path, out_config["video_filename"]))
    return (
        out_config["video_filename"],
        main_frames,
        processed.js(),
        plaintext_to_html(processed.info),
        plaintext_to_html(""),
    )

def create_mask_with_circles(original_image, border_width, border_height, overmask: int, radius=4):
    # Create a new image with border and draw a mask
    new_width = original_image.width + 2 * border_width
    new_height = original_image.height + 2 * border_height

    # Create new image, default is black
    mask = Image.new('RGB', (new_width, new_height), 'white')

    # Draw black rectangle
    draw = ImageDraw.Draw(mask)
    draw.rectangle([border_width+overmask, border_height+overmask, new_width - border_width-overmask, new_height - border_height-overmask], fill='black')

    # Coordinates for circles
    circle_coords = [
        (border_width, border_height),  # Top-left
        (new_width - border_width, border_height),  # Top-right
        (border_width, new_height - border_height),  # Bottom-left
        (new_width - border_width, new_height - border_height),  # Bottom-right
        (new_width // 2, border_height),  # Middle-top
        (new_width // 2, new_height - border_height),  # Middle-bottom
        (border_width, new_height // 2),  # Middle-left
        (new_width - border_width, new_height // 2)  # Middle-right
    ]

    # Draw circles
    for coord in circle_coords:
        draw.ellipse([coord[0] - radius, coord[1] - radius, coord[0] + radius, coord[1] + radius], fill='white')
    return mask





def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def cv2_crop_center(img, toSize: tuple[int,int]):
    y,x = img.shape[:2]
    startx = x//2-(toSize[0]//2)
    starty = y//2-(toSize[1]//2)
    return img[starty:starty+toSize[1],startx:startx+toSize[0]]