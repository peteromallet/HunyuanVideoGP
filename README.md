<!-- ## **HunyuanVideo** -->

[中文阅读](./README_zh.md)


# HunyuanVideoGP: Large Video Generation for the GPU Poor
<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2412.03603"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv:HunyuanVideo&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp; &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-PromptRewrite"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-PromptRewrite&message=HuggingFace&color=yellow"></a> &ensp; &ensp;

</div>
<p align="center">


## News
* 02/25/2025: Version 5.0: **Out Of this World Release by DeepBeepMeep that lands only in HunyuanVideo GP: VRAM laws have been broken as VRAM consumption has been divided by 3 and 20%-50% faster at no quality loss !**

*You can now generate videos that lasts up to 10s of 1280x720 and 16s of 848x480 with 24 GB of VRAM with Loras and no quantization !!!*

Welcome to low VRAM GPUs owners as from now on you can generate multiseconds videos.

Many thanks to RIFLEx (https://github.com/thu-ml/RIFLEx) and their very good released timing, for their positional embeddign breakthrough that allows generating videos longer than up to 10s that doesn't look like still life.

Please note that although there will be still sufficient VRAM left, generating video longer than 10s with Hunyuan current models is useless as the videos starts to get redundant

If you have already installed HunyuanVideoGP, you will need to run *pip install -r requirements.txt*. Upgrading to python 2.6.0 and the corresponding attention libaries is a plus for performance.

* 02/11/2025: Version 4.0 Quality of life features: fast abort video generation, detect automatically attention modes not supported, you can now change video engine parameters without having to restart the app
* 02/11/2025: Version 3.5 optimized lora support (reduced VRAM requirements and faster). You can now generate 1280x720 97 frames with Loras in 3 minutes only in the fastest mode
* 02/10/2025: Version 3.4 New --fast and --fastest switches to automatically get the best performance
* 02/10/2025: Version 3.3 Prefill automatically optimal parameters for Fast Hunyuan
* 02/07/2025: Version 3.2 Added support for Xformers attention and reduce VRAM requirements for sdpa attention
* 01/21/2025: Version 3.1 Ability to define a Loras directory and turn on / off any Lora when running the application
* 01/11/2025: Version 3.0 Multiple prompts / multiple generations per prompt, new progression bar, support for pretrained Loras
* 01/06/2025: Version 2.1 Integrated Tea Cache (https://github.com/ali-vilab/TeaCache) for even faster generations
* 01/04/2025: Version 2.0 Full leverage of mmgp 3.0 (faster and even lower RAM requirements ! + support for compilation on Linux and WSL)
* 12/22/2024: Version 1.0 First release

## Features
*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly on any GPU.*

This version has the following improvements over the original Hunyuan Video model:
- Reduce greatly the RAM requirements and VRAM requirements
- Much faster thanks to compilation and fast loading / unloading
- 5 profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM)
- Autodownloading of the needed model files
- Improved gradio interface with progression bar and more options
- Multiples prompts / multiple generations per prompt
- Support multiple pretrained Loras with 32 GB of RAM or less
- Switch easily between Hunyuan and Fast Hunyuan models and quantized / non quantized models
- Much simpler installation



This fork by DeepBeepMeep is an integration of the mmpg module on the gradio_server.py.

It is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with changing only a few lines of code in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp

You will find the original Hunyuan Video repository here: https://github.com/Tencent/HunyuanVideo
 


## Installation Guide for Linux and Windows

We provide an `environment.yml` file for setting up a Conda environment.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

This app has been tested on Python 3.10 / Pytorch 2.5.1 - 2.6.0  / Cuda 12.4.\
(Pytorch compilation will not properly work for long video on Pytorch 2.5.1)

```shell
# 1 - conda. Prepare and activate a conda environment
conda env create -f environment.yml
conda activate HunyuanVideo

# OR

# 1 - venv. Alternatively create a python 3.10 venv and then do the following
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124  # if pytorch 2.5.1
#or 
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124  # if pytorch 2.6.0


# 2. Install pip dependencies
python -m pip install -r requirements.txt

# 3.1 optional Flash attention support (easy to install on Linux but much harder on Windows)
python -m pip install flash-attn==2.7.2.post1

# 3.2 optional Sage attention support (30% faster, easy to install on Linux but much harder on Windows)
python -m pip install sageattention==1.0.6 

# or for Sage Attention 2 (40% faster, sorry only manual compilation for the moment)
git pull https://github.com/thu-ml/SageAttention
cd sageattention 
pip install -e .

# 3.3 optional Xformers attention support (same speed as sdpa attention but lower VRAM requirements, easy to install on Linux but much harder on Windows)
python -m pip install xformers==0.0.29

```

Note that *Flash attention* and *Sage attention* are quite complex to install on Windows but offers a better memory management (and consequently longer videos) than the default *sdpa attention*.
Likewise *Pytorch Compilation* will work on Windows only if you manage to install Triton. It is quite a complex process (see below for links).

### Ready to use python wheels for Windows users
I provide here links to simplify the installation for Windows users with Python 3.10 / Pytorch 2.51 / Cuda 12.4. As I am not hosting these files I won't be able to provide support neither guarantee they do what they should do.
- Triton attention (needed for *pytorch compilation* and *Sage attention*)
```
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/triton-3.1.0-cp310-cp310-win_amd64.whl # triton for pytorch 2.6.0
#or 
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl # triton for pytorch 2.6.0
```
- Xformers attention
```
pip install https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp310-cp310-win_amd64.whl
```

- Sage attention
```
pip install https://github.com/sdbds/SageAttention-for-windows/releases/download/2.0.1/sageattention-2.0.1+cu124torch2.5.1-cp310-cp310-win_amd64.whl # for pytorch 2.5.1
# Or
pip install https://github.com/deepbeepmeep/SageAttention/raw/refs/heads/main/releases/sageattention-2.1.0-cp310-cp310-win_amd64.whl # for pytorch 2.6.0 (experimental, if it works, otherwise you you will need to install and compile manually, see above) 
 
```

## Run the application

### Run a Gradio Server on port 7860 (recommended)
```bash
python gradio_server.py
```

Every lora stored in the subfoler 'loras' will be automatically loaded. You will be then able to activate / desactive any of them when running the application

You can also pre activate some loras and specify their corresponding multipliers when loading the app:
```bash
python gradio_server.py --lora-weight lora.safetensors --lora-multiplier 1
```

For each activated Lora, you may specify one float number that corresponds to its weight, alternatively you may specify a list of floats separated by a "," that gives the evolution of this Lora over the steps. For instance let's assume there are 30 denoising steps and the multiplier is *0.9,0.8,0.7* then for the steps ranges 0-9, 10-19 and 20-29 the Lora multiplier will be respectively 0.9, 0.8 and 0.7.


You can find prebuilt Loras on https://civitai.com/ or build them with tools such kohya or onetrainer.

### Give me Speed !
If you are a speed addict and are ready to accept some tradeoff on the quality I have added two switches:
- Fast Hunyuan Video enabled by default + Sage Attention + Teacache (an advanced acceleration algorithm x2 the speed for a small quality cost)
```bash
python gradio_server.py --fast
```

- Fast Hunyuan Video enabled by default + Sage Attention + Teacache (an advanced acceleration algorithm x2 the speed for a small quality cost) + Compilation  
```bash
python gradio_server.py --fastest
```
Please note that the first sampling step of the first video generation will take two minutes to perform the compilation. Consecutive generations will be very fast unless you trigger a new compilation by changing the resolution, duration of the video or add / remove loras.

For these two switches to work you will need to install Triton and Sage attention.

As you can change the prompt without causing a recompilation, theses switches work quite well with th *Multiple prompts* and / or *Multiple Generations* options.

With the *--fastest* switch activated **a 1280x720 97 frames video takes with a Lora takes less than 4 minutes to be generated** !


If you are looking for a good tradeoff between speed and quality I suggest you use the official HunyuanVideo model with Sage attention and pytorch compilation. You may as well turn on Teacache which will degrade less the video quality given there are more processing steps. 
```bash
python gradio_server.py --attention sage --compile
```

### Command line parameters for Gradio Server
--profile no : default (4) : no of profile between 1 and 5\
--quantize-transformer bool: (default True) : enable / disable on the fly transformer quantization\
--lora-dir path : Path of directory that contains Loras in diffusers / safetensor format\
--lora-weight path1 path2 ... : list of Loras Path preselected Loras\
--lora-multiplier float mult1 mult2 ... : list of relative weights for each preselected Lora. The corresponding Lora file must be in the diffusers format.\
--verbose level : default (1) : level of information between 0 and 2\
--server-port portno : default (7860) : Gradio port no\
--server-name name : default (0.0.0.0) : Gradio server name\
--open-browser : open automatically Browser when launching Gradio Server\
--fast : start the app by loading Fast Hunyuan Video generator (faster but lower quality) + sage attention + teacache x2 
--compile : turn on pytorch compilation\
--fastest : shortcut for --fast + --compile\
--attention mode: force attention mode among, sdpa, flash, sage and xformers\
--vae-mode: 0-5, defalt(0) : VAE tiling to be used for latents decoding

### Profiles (for power users only)
You can choose between 5 profiles, these will try to leverage the most your hardware, but have little impact for HunyuanVideo GP:
- HighRAM_HighVRAM  (1):  the fastest well suited for a RTX 3090 / RTX 4090 but consumes much more VRAM, adapted for fast shorter video
- HighRAM_LowVRAM  (2): a bit slower, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos
- LowRAM_HighVRAM  (3): adapted for RTX 3090 / RTX 4090 with limited RAM  but at the cost of VRAM (shorter videos)
- LowRAM_LowVRAM  (4): if you have little VRAM or want to generate longer videos 
- VerylowRAM_LowVRAM  (5): at least 24 GB of RAM and 10 GB of VRAM : if you don't have much it won't be fast but maybe it will work

Profile 2 (High RAM) and 4 (Low RAM)are the most recommended profiles since they are versatile (support for long videos for a slight performance cost).\
However, a safe approach is to start from profile 5 (default profile) and then go down progressively to profile 4 and then to profile 2 as long as the app remains responsive or doesn't trigger any out of memory error.

### Other Models for the GPU Poor
- Hunyuan3D-2GP: https://github.com/deepbeepmeep/Hunyuan3D-2GP :\
A great image to 3D and text to 3D tool by the Tencent team. Thanks to mmgp it can run with less than 6 GB of VRAM

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP :\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP :\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).

- OminiControlGP: https://github.com/deepbeepmeep/OminiControlGP :\
A Flux derived application very powerful that can be used to transfer an object of your choice in a prompted scene. With mmgp you can run it with only 6 GB of VRAM.

- YuE GP: https://github.com/deepbeepmeep/YuEGP :\
A great song generator (instruments + singer's voice) based on prompted Lyrics and a genre description. Thanks to mmgp you can run it with less than 10 GB of VRAM without waiting forever.




### Run through the command line
```bash
cd HunyuanVideo

python sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --save-path ./results
```

Please note currently that profile and the models used need to be mentioned inside the *sample_video.py* file.

### More Configurations for Gradio Server and Command line

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--prompt`       |   None    |   The text prompt for video generation    |
|     `--video-size`     | 720 1280  |      The size of the generated video      |
|    `--video-length`    |    129    |     The length of the generated video     |
|    `--infer-steps`     |    50     |     The number of steps for sampling      |
| `--embedded-cfg-scale` |    6.0    |    Embeded  Classifier free guidance scale       |
|     `--flow-shift`     |    7.0    | Shift factor for flow matching schedulers |
|     `--flow-reverse`   |    False  | If reverse, learning/sampling from t=1 -> t=0 |
|        `--seed`        |     None  |   The random seed for generating video, if None, we init a random seed    |
|  `--use-cpu-offload`   |   False   |    Use CPU offload for the model load to save more memory, necessary for high-res video generation    |
|     `--save-path`      | ./results |     Path to save the generated video      |

## **Abstract**
We present HunyuanVideo, a novel open-source video foundation model that exhibits performance in video generation that is comparable to, if not superior to, leading closed-source models. In order to train HunyuanVideo model, we adopt several key technologies for model learning, including data curation, image-video joint model training, and an efficient infrastructure designed to facilitate large-scale model training and inference. Additionally, through an effective strategy for scaling model architecture and dataset, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. 

We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion diversity, text-video alignment, and generation stability. According to professional human evaluation results, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and 3 top-performing Chinese video generative models. By releasing the code and weights of the foundation model and its applications, we aim to bridge the gap between closed-source and open-source video foundation models. This initiative will empower everyone in the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. 



## 🔗 BibTeX
If you find [HunyuanVideo](https://arxiv.org/abs/2412.03603) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv preprint arXiv:2412.03603},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03603}, 
}
```



## 🧩 Projects that use HunyuanVideo

If you develop/use HunyuanVideo in your projects, welcome to let us know.

- ComfyUI (with support for F8 Inference and Video2Video Generation): [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) by [Kijai](https://github.com/kijai)



## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
Additionally, we also thank the Tencent Hunyuan Multimodal team for their help with the text encoder. 

## Star History
<a href="https://star-history.com/#Tencent/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
 </picture>
</a>
