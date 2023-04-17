# Infinite Zoom extension for  [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/).

<p align="center">     
    <a href="https://discord.gg/v2nHqSrWdW">
        <img src="https://img.shields.io/discord/1095469311830806630?color=blue&label=discord&logo=discord&logoColor=white" alt="build status">
    </a>
</p>

This is an extension for the AUTOMATIC1111's webui that allows users to create infinite zoom effect videos using stable diffusion outpainting method. 


## How to install?
<details>
  <summary> Click to expand </summary>
  
1. Open [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/).

2. Go to the `Extensions tab` > `Install from URL`.

3. Enter `https://github.com/v8hid/infinite-zoom-automatic1111-webui.git` for the URL and leave the second field empty and wait for it to be installed.
> <img width="587" alt="Screenshot 2023-04-12 at 10 17 32 PM" src="https://user-images.githubusercontent.com/62482657/231554653-16484c48-582e-489d-8191-bafc4cccbd3b.png">

4. Go to the Installed tab and press Apply, wait for installation, and restart.

> <img width="616" alt="Screenshot 2023-04-12 at 10 17 57 PM" src="https://user-images.githubusercontent.com/62482657/231554793-4a54ae94-51d2-408e-9908-2eed73cde9c0.png">

5. Wait for the Stable Diffusion WebUI to restart and now you can try the Infinite Zoom extension.

</details>

## How to use?

<details>
  <summary> Click to expand </summary>
  
 1. Click on the Infinite Zoom tab <img width="1431" alt="Screenshot 2023-04-12 at 10 14 50 PM" src="https://user-images.githubusercontent.com/62482657/231571341-92767f0d-af36-4b94-8ba9-c40a63c209ba.png">
 
 2. Modify the parameters as you wish and click Generate video, the video will appear as soon as it generates
 
 </details>
 
**To learn more about the parameters, please refer to our [WIKI](https://github.com/v8hid/infinite-zoom-automatic1111-webui/wiki).**
 ## Effective Friendly Tips for Optimal Outcomes
 
<details>
  <summary> Click to expand </summary>
  
* You're only as good as your model, so level up with an <ins>Inpainting model</ins> for killer results.

* Heads up: Setting <ins>Mask Blur</ins> parameter above 0 will give you results that look like they've been hit by the ugly stick.

* Just between us - don't forget to uncheck <ins> Apply color correction to img2img results to match original colors</ins> in the Stable Diffusion tab of the WebUI settings. You don't want your results looking like a bad Instagram filter.

</details>

## Examples

<details>
  <summary> Click to expand </summary>



https://user-images.githubusercontent.com/62482657/232369614-e112d17a-db12-47b2-9795-5be4037fa9fe.mp4


https://user-images.githubusercontent.com/62482657/231573289-2db85c57-540d-4c7d-859f-3c3ddfcd2c8a.mp4


https://user-images.githubusercontent.com/62482657/231574588-3196beda-7237-407f-bc76-eae10599b5eb.mp4


https://user-images.githubusercontent.com/62482657/231574839-9d3aab52-7a87-4658-88d0-46b8dd7f4b60.mp4

 </details>

## How it works?
<details>
  <summary> Click to expand </summary>
  
To start, let's break down the workflow of the extension into three main steps:

- **Step 1: Choose an image to start with**
The program either generates an initial image using the first prompt you provide or you can upload your own image in the `custom initial image` field. This initial image will be the basis for the outpainting process.

- **Step 2: Generate outpaint steps**
Once you have your initial image, the program will start generating outpaint steps. The number of outpaint steps is determined by the `Total Outpaint Steps` input. In each outpaint step, the program makes the initial image smaller in the center of the canvas and generates a new image in the empty space that is created. This process is repeated for each outpaint step until the desired number is reached.

- **Step 3: Create a gradual zoom effect**
After all outpaint steps have been generated, the program creates an interpolation between each outpaint step to create a gradual zoom effect. The number of frames created between each outpaint step is determined by the `Zoom Speed` parameter and the `Frames per second` parameter.

Number of frames for each outpaint step = `Zoom Speed` $\times$ `Frames per second`

Length of each outpaint step in second = `Number of frames` $\div$ `Frames per second` 

 </details>
 
## Google Colab version
It works on free colab plan

<a target="_blank" href="https://colab.research.google.com/github/v8hid/infinite-zoom-stable-diffusion/blob/main/infinite_zoom_gradio.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://github.com/v8hid/infinite-zoom-stable-diffusion">
  <img src="https://img.shields.io/static/v1?label=github&message=repository&color=blue&style=flat&logo=github&logoColor=white" alt="Open In Colab"/>
</a>

## Contributing

Contributions are welcome! Please follow these guidelines:

  1. Fork the repository.
  2. Make your changes and commit them.
  3. Submit a pull request to the main repository.
