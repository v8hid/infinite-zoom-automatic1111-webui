import math, time, os
import numpy as np
from typing import Callable, Any
from PIL import Image, ImageFilter, ImageOps, ImageDraw

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
from .video import ContinuousVideoWriter
from .InfZoomConfig import InfZoomConfig
 
class InfZoomer:
    def __init__(self, config: InfZoomConfig) -> None:
        self.C = config
        self.prompts = {}
        self.main_frames = []
        self.out_config = {}

        for x in self.C.prompts_array:
            try:
                key = int(x[0])
                value = str(x[1])
                self.prompts[key] = value
            except ValueError:
                pass

        assert len(self.C.prompts_array) > 0, "prompts is empty"

        fix_env_Path_ffprobe()
        self.out_config = self.prepare_output_path()

        self.width = closest_upper_divisible_by_eight(self.C.outputsizeW)
        self.height = closest_upper_divisible_by_eight(self.C.outputsizeH)

        self.current_seed = self.C.seed

        self.mask_width  = self.width  * 1//4   # was initially 512px => 128px
        self.mask_height = self.height * 1//4   # was initially 512px => 128px
        self.num_interpol_frames = round(self.C.video_frame_rate * self.C.zoom_speed)

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

    def create_zoom(self):
        for i in range(self.C.batchcount):
            print(f"Batch {i+1}/{self.C.batchcount}")
            result = self.create_zoom_single()
        return result

    def create_zoom_single(self):

        self.main_frames.append(self.prepareInitImage())

        load_model_from_setting("infzoom_inpainting_model", self.C.progress, "Loading Model for inpainting/img2img: ")

        processed = self.fnOutpaintMainFrames()

        if (self.C.upscale_do): 
            self.doUpscaling()

        if self.C.video_zoom_mode:
            self.main_frames = self.main_frames[::-1]

        if not self.outerZoom:
            self.contVW = ContinuousVideoWriter(self.out_config["video_filename"], self.main_frames[0],self.C.video_frame_rate,int(self.C.video_start_frame_dupe_amount))
        
        self.fnInterpolateFrames(self) # changes main_frame and writes to video

        self.contVW.finish(self.main_frames[-1],int(self.C.video_last_frame_dupe_amount))

        print("Video saved in: " + os.path.join(script_path, self.out_config["video_filename"]))

        return (
            self.out_config["video_filename"],
            self.main_frames,
            processed.js(),
            plaintext_to_html(processed.info),
            plaintext_to_html(""),
        )

    def doUpscaling(self):
        for idx,mf in enumerate(self.main_frames):
            print (f"\033[KInfZoom: Upscaling mainframe: {idx}   \r")
            self.main_frames[idx]=do_upscaleImg(mf, self.C.upscale_do, self.C.upscaler_name, self.C.upscale_by)

        self.width  = self.main_frames[0].width
        self.height = self.main_frames[0].height
        self.mask_width = math.trunc(self.mask_width*self.C.upscale_by)
        self.mask_height = math.trunc(self.mask_height *self.C.upscale_by)

    def prepareInitImage(self) -> Image:
        if self.C.custom_init_image:
            current_image = Image.new(mode="RGBA", size=(self.width, self.height))
            current_image = current_image.convert("RGB")
            current_image = self.C.custom_init_image.resize(
                (self.width, self.height), resample=Image.LANCZOS
            )
            self.save2Collect(current_image, f"init_custom.png")
        else:
            load_model_from_setting("infzoom_txt2img_model", self.C.progress, "Loading Model for txt2img: ")

            processed, newseed = self.renderFirstFrame()

            if len(processed.images) > 0:
                current_image = processed.images[0]
                self.save2Collect(current_image, f"init_txt2img.png")
            self.current_seed = newseed
        return current_image

    def renderFirstFrame(self):
        pr = self.getInitialPrompt()

        return renderTxt2Img(
                f"{self.C.common_prompt_pre}\n{pr}\n{self.C.common_prompt_suf}".strip(),
                self.C.negative_prompt,
                self.C.sampler,
                self.C.num_inference_steps,
                self.C.guidance_scale,
                self.current_seed,
                self.width,
                self.height
        )

    def getInitialPrompt(self):
        return self.prompts[min(k for k in self.prompts.keys() if k >= 0)]
    
    def outpaint_steps_cornerStrategy(self):
        currentImage = self.main_frames[-1]

        original_width, original_height = currentImage.size

        new_width = original_width + self.mask_width*2
        new_height = original_height + self.mask_height*2
        left = right = int(self.mask_width)
        top = bottom = int(self.mask_height)

        corners = [
            (0, 0),  
            (new_width - 512, 0),  
            (0, new_height - 512),  
            (new_width - 512, new_height - 512),  
        ]
        masked_images = []
        
        for idx, corner in enumerate(corners):
            white = Image.new("1", (new_width,new_height), 1)
            draw = ImageDraw.Draw(white)
            draw.rectangle([corner[0], corner[1], corner[0]+512, corner[1]+512], fill=0)
            masked_images.append(white)

        outpaint_steps=self.C.num_outpainting_steps
        for i in range(outpaint_steps):
            print (f"Outpaint step: {str(i + 1)}/{str(outpaint_steps)} Seed: {str(self.current_seed)}")
            currentImage = self.main_frames[-1]

            if self.C.custom_exit_image and ((i + 1) == outpaint_steps):
                currentImage = self.C.custom_exit_image.resize(
                    (self.C.width, self.C.height), resample=Image.LANCZOS
                )
                
                if not self.outerZoom:
                    self.main_frames.append(currentImage.convert("RGB"))

                self.save2Collect(currentImage, self.out_config, f"exit_img.png")
            else:
                expanded_image = ImageOps.expand(currentImage, (left, top, right, bottom), fill=(0, 0, 0))
                pr = self.prompts[max(k for k in self.prompts.keys() if k <= i)]
                
                # outpaint 4 corners loop
                for idx,cornermask in enumerate(masked_images):
                    processed, newseed = renderImg2Img(
                        f"{self.C.common_prompt_pre}\n{pr}\n{self.C.common_prompt_suf}".strip(),
                        self.C.negative_prompt,
                        self.C.sampler,
                        self.C.num_inference_steps,
                        self.C.guidance_scale,
                        self.current_seed,
                        512,  #outpaintsizeW
                        512,  #outpaintsizeH
                        expanded_image,
                        cornermask,
                        1, #inpainting_denoising_strength,
                        0, # inpainting_mask_blur,
                        2, ## noise? fillmode
                        True,  # only masked, not full, keep size of expandedimage!
                        0 #inpainting_padding,
                    )
                    expanded_image = processed.images[0]
                #
                
                if len(processed.images) > 0:
                    zoomed_img = expanded_image.resize((self.width,self.height), Image.Resampling.LANCZOS)

                    if self.outerZoom:
                        self.main_frames[-1] = expanded_image # replace small image
                        processed.images[0]=expanded_image    # display overscaned image in gallery
                        self.save2Collect(processed.images[0], f"outpaint_step_{i}.png")
                        
                        if (i < outpaint_steps-1):
                            self.main_frames.append(zoomed_img)   # prepare next frame with former content

                    else:
                        zoomed_img = expanded_image.resize((self.width,self.height), Image.Resampling.LANCZOS)
                        self.main_frames.append(zoomed_img)
                        processed.images[0]=self.main_frames[-1]
                        self.save2Collect(processed.images[0], f"outpaint_step_{i}.png")

        return processed
    

    def outpaint_steps_v8hid(self):

        for i in range(self.C.num_outpainting_steps):
            print (f"Outpaint step: {str(i + 1)} / {str(self.C.num_outpainting_steps)} Seed: {str(self.current_seed)}")
        
            current_image = self.main_frames[-1]
            current_image = shrink_and_paste_on_blank(
                current_image, self.mask_width, self.mask_height
            )

            mask_image = np.array(current_image)[:, :, 3]
            mask_image = Image.fromarray(255 - mask_image).convert("RGB")

            if self.C.custom_exit_image and ((i + 1) == self.C.num_outpainting_steps):
                current_image = self.C.custom_exit_image.resize(
                    (self.width, self.height), resample=Image.LANCZOS
                )
                self.main_frames.append(current_image.convert("RGB"))
                # print("using Custom Exit Image")
                self.save2Collect(current_image, f"exit_img.png")
            else:
                pr = self.prompts[max(k for k in self.prompts.keys() if k <= i)]
                processed, newseed = renderImg2Img(
                    f"{self.C.common_prompt_pre}\n{pr}\n{self.C.common_prompt_suf}".strip(),
                    self.C.negative_prompt,
                    self.C.sampler,
                    self.C.num_inference_steps,
                    self.C.guidance_scale,
                    self.current_seed,
                    self.width,
                    self.height,
                    current_image,
                    mask_image,
                    self.C.inpainting_denoising_strength,
                    self.C.inpainting_mask_blur,
                    self.C.inpainting_fill_mode,
                    self.C.inpainting_full_res,
                    self.C.inpainting_padding,
                )

                if len(processed.images) > 0:
                    self.main_frames.append(processed.images[0].convert("RGB"))
                    self.save2Collect(processed.images[0], f"outpain_step_{i}.png")
                seed = newseed
                # TODO: seed behavior

        return processed


    def outpaint_steps_smallCenter(self):

        # one mask for all steps and outpaints
        mask_image = np.array(shrink_and_paste_on_blank(
            self.main_frames[0], self.mask_width+self.C.overmask, self.mask_height+self.C.overmask)
        )[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")

#       if (self.C.inpainting_fill_mode == 1): # ORIGINAL

        outpaint_steps=self.C.num_outpainting_steps

        for i in range(outpaint_steps):
            print (f"Outpaint step: {str(i + 1)}/{str(outpaint_steps)} Seed: {str(self.current_seed)}")
            currentImage = self.main_frames[-1]

            if self.C.custom_exit_image and ((i + 1) == outpaint_steps):
                currentImage = self.C.custom_exit_image.resize(
                    (self.C.width, self.C.height), resample=Image.LANCZOS
                )
                self.main_frames.append(currentImage.convert("RGB"))
                # print("using Custom Exit Image")
                self.save2Collect(currentImage, self.out_config, f"exit_img.png")
            else:
                smaller_image = shrink_and_paste_on_blank(currentImage,self.mask_width,self.mask_height)
                pr = self.prompts[max(k for k in self.prompts.keys() if k <= i)]
                
                processed, newseed = renderImg2Img(
                    f"{self.C.common_prompt_pre}\n{pr}\n{self.C.common_prompt_suf}".strip(),
                    self.C.negative_prompt,
                    self.C.sampler,
                    self.C.num_inference_steps,
                    self.C.guidance_scale,
                    self.current_seed,
                    512,  #outpaintsizeW
                    512,  #outpaintsizeH
                    smaller_image,
                    mask_image,
                    1, #inpainting_denoising_strength,
                    self.C.inpainting_mask_blur,
                    self.C.inpainting_fill_mode,
                    True,  # only masked, not full, keep size of expandedimage!
                    0 #inpainting_padding,
                )
                
                if len(processed.images) > 0:
                    self.main_frames.append(processed.images[0])
                    self.save2Collect(processed.images[0], f"outpaint_step_{i}.png")
                seed = newseed
                # TODO: seed behavior
        return processed


    def calculate_interpolation_steps(self, original_size, target_size, steps):
        width, height = original_size
        target_width, target_height = target_size

        if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0 or steps <= 0:
            return None

        width_step = math.floor((width - target_width) / steps)
        height_step = math.floor((height - target_height) / steps)

        scaling_steps = [(width - i * width_step, height - i * height_step) for i in range(1,steps)]

        return scaling_steps

   
    def interpolateFramesOuterZoom(cls,self):

        if 0 == self.C.video_zoom_mode:
            current_image = self.main_frames[0]
        elif 1 == self.C.video_zoom_mode:
            current_image = self.main_frames[-1]
        else:
            raise ValueError("unsupported Zoom mode in INfZoom")

        self.contVW = ContinuousVideoWriter(self.out_config["video_filename"], 
                                            cls.cropCenterTo(current_image,(self.width,self.height)),
                                            self.C.video_frame_rate,int(self.C.video_start_frame_dupe_amount))

        outzoomSize = (self.width+self.mask_width*2, self.height+self.mask_height*2)
        target_size = (self.width, self.height)
        scaling_steps = self.calculate_interpolation_steps(outzoomSize, target_size, self.num_interpol_frames)

        print(f"{scaling_steps}, length: {len(scaling_steps)}")
        for x,y in scaling_steps:
            print (f"ratios: {x/y}")
        
        for i in range(len(self.main_frames)):
            if 0 == self.C.video_zoom_mode:
                current_image = self.main_frames[0+i]
            else:
                current_image = self.main_frames[-1-i]

            self.contVW.append([
                cls.cropCenterTo(current_image,(self.width, self.height))
            ])

            # interpolation steps between 2 inpainted images (=sequential zoom and crop)
            for j in range(self.num_interpol_frames - 1):

                print (f"\033[KInfZoom: Interpolate frame: main/inter: {i}/{j}   \r")
                #todo: howto zoomIn when writing each frame; self.main_frames are inverted, howto interpolate?
                scaled_image = current_image.resize(scaling_steps[j])                    
                cropped_image = cls.cropCenterTo(scaled_image,(self.width, self.height))

                self.contVW.append([cropped_image])


    def interpolateFramesSmallCenter(self):
        for i in range(len(self.main_frames) - 1):
            # interpolation steps between 2 inpainted images (=sequential zoom and crop)
            for j in range(self.num_interpol_frames - 1):

                print (f"\033[KInfZoom: Interpolate frame: main/inter: {i}/{j}   \r")
                #todo: howto zoomIn when writing each frame; self.main_frames are inverted, howto interpolate?
                if self.C.video_zoom_mode:
                    current_image = self.main_frames[i + 1]
                else:
                    current_image = self.main_frames[i + 1]
                    
                interpol_image = current_image
                self.save2Collect(interpol_image, f"interpol_img_{i}_{j}].png")

                interpol_width = math.ceil(
                    ( 1 - (1 - 2 * self.mask_width / self.width) **(1 - (j + 1) / self.num_interpol_frames) ) 
                    * self.width / 2
                )

                interpol_height = math.ceil(
                    ( 1 - (1 - 2 * self.mask_height / self.height) ** (1 - (j + 1) / self.num_interpol_frames) )
                    * self.height/2
                )

                interpol_image = interpol_image.crop(
                    (
                        interpol_width,
                        interpol_height,
                        self.width - interpol_width,
                        self.height - interpol_height,
                    )
                )

                interpol_image = interpol_image.resize((self.width, self.height))
                self.save2Collect(interpol_image, f"interpol_resize_{i}_{j}.png")

                # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
                interpol_width2 = math.ceil(
                    (1 - (self.width - 2 * self.mask_width) / (self.width - 2 * interpol_width))
                    / 2 * self.width
                )

                interpol_height2 = math.ceil(
                    (1 - (self.height - 2 * self.mask_height) / (self.height - 2 * interpol_height))
                    / 2 * self.height
                )

                prev_image_fix_crop = shrink_and_paste_on_blank(
                    self.main_frames[i], interpol_width2, interpol_height2
                )

                interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)
                self.save2Collect(interpol_image, f"interpol_prevcrop_{i}_{j}.png")

                self.contVW.append([interpol_image])

            self.contVW.append([current_image])


    def prepare_output_path(self):
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


    def save2Collect(self, img, name):
        if self.out_config["isCollect"]:
            img.save(f'{self.out_config["save_path"]}/{name}.png')


    def frame2Collect(self,all_frames):
        self.save2Collect(all_frames[-1], self.out_config, f"frame_{len(all_frames)}")


    def frames2Collect(self, all_frames):
        for i, f in enumerate(all_frames):
            self.save2Collect(f, self.out_config, f"frame_{i}")


    def crop_inner_image(self, outpainted_img, width_offset, height_offset):
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

    @staticmethod
    def cropCenterTo(im: Image, toSize: tuple[int,int]):
        width, height = im.size
        left = (width - toSize[0])//2
        top = (height - toSize[1])//2
        right = (width + toSize[0])//2
        bottom = (height + toSize[1])//2
        return im.crop((left, top, right, bottom))
