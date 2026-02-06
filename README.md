# 3D-RE-GEN

<p align="left">
  <strong>
    3D Reconstruction of Indoor Scenes with a Generative Framework
  </strong>
</p>


<img width="2504" height="843" alt="teaser" src="https://github.com/user-attachments/assets/bc8057b2-488d-4bcc-ab05-73331066cfac" />



<p align="center">
    <span> 🌐  <a href="https://3dregen.jdihlmann.com/"> Project Page </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 📄  <a href="https://arxiv.org/abs/2512.17459"> Paper (Arxiv) </a> </span>&nbsp;&nbsp;&nbsp;
  <!-- <span>  📦  <a href="https://drive.google.com/drive/folders/1znN_KllBKllIY_1PLZUHbnfHsB6KNifR?usp=sharing"> Materials </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  ✍🏻
     <a href="https://github.com/cgtuebingen/MatSpray?tab=readme-ov-file#citation"> Citation </a> </span>&nbsp;&nbsp;&nbsp; -->
</p>

# About
We propose single-image 3D scene reconstruction for producing complete, editable scenes from a single photograph. Our method reconstructs individual objects and the surrounding background as textured 3D assets, enabling coherent scene assembly from minimal input. We combine instance segmentation, context-aware generative inpainting, 2D-to-3D asset creation, and constrained optimization to recover physically plausible geometry, materials, and lighting. The resulting scenes preserve correct spatial relationships, lighting consistency, and material fidelity, making them suitable for production-ready workflows.

# Getting Started

See [INSTALLATION.md](INSTALLATION.md) for setup instructions.

Quick start:
```bash
git clone --recursive https://github.com/cgtuebingen/3D-RE-GEN.git
cd 3D-RE-GEN
cd segmentor && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && cd ..
mamba run -p ./venv_py310 python run.py -p 1 2 3 4 5 6 7 8 9
```

<!-- # Code
Paper is currently under review, we will realse the code shortly (~ end of january 2026) in a cleaned version - Star the Repo to not miss it. Up until then if you have any questions regarding the project or need material to compare against fast, feel free to contact us.  -->



# Citation
You can find our paper on [arXiv](https://arxiv.org/), please consider citing, if you find this work useful:

```
@inproceeding{sautter20253dregen,
author ={Sautter, Tobias and Dihlmann, Jan-Niklas and Lensch, Hendrik P.A.},
title ={3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework},
booktitle ={arXiv preprint},
year ={2025}
}
