# Overview

This is an unofficial and incomplete implementation of **Line-based PatchMatch MVS (LPMVS)**, presented as a module of [ Strand-accurate multi-view hair capture. CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nam_Strand-Accurate_Multi-View_Hair_Capture_CVPR_2019_paper.pdf)

This code cannot reproduce the results in the paper.

# Difference

## CPU implementation

This code is so slow because it is a CPU implementation from scratch.

The original authors may modify the CUDA implementation of the official implementation of the following paper.

S. Galliani, K. Lasinger and K. Schindler, [Massively Parallel Multiview Stereopsis by Surface Normal Diffusion](https://prs.igp.ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/galliani-lasinger-iccv15.pdf), ICCV 2015

- https://github.com/kysucix/gipuma

- https://github.com/kysucix/fusibile

# Next steps

## Better 2D line feature extraction

The current implementation is not as good as the paper.

## Depth initialization range

The original authors didn't report the details of random depth value initialization **range**. Accurate depth value initialization (e.g., traditional MVS such as COLMAP or OpenMVS) may improve performance.

# Build

- `git submodule update --init --recursive`
  - To pull dependencies registered as git submodule.
- Use CMake with `CMakeLists.txt`.

# Reference

[Nam, G., Wu, C., Kim, M. H., & Sheikh, Y. (2019). Strand-accurate multi-view hair capture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 155-164).](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nam_Strand-Accurate_Multi-View_Hair_Capture_CVPR_2019_paper.pdf)
