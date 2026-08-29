# DESCRIPTION
This project implements a CUDA-based batch image processing system for processing a large number of images using GPU computation. The program converts RGB images into grayscale images and then applies Sobel edge detection using CUDA kernels, where individual GPU threads process image pixels in parallel. The application supports command-line arguments for selecting the input and output directories and the number of images to process.

During development, I worked with CUDA memory allocation, CPU-to-GPU and GPU-to-CPU data transfers, kernel configuration, and boundary checking to prevent invalid memory accesses. One of the main challenges was efficiently handling multiple images and ensuring that images with different dimensions were processed correctly. The execution results demonstrate that the program can process a large batch of images successfully using the GPU. This project helped me understand practical GPU parallelism, CUDA kernel execution, memory management, and the performance considerations involved in large-scale image processing.

# CUDA Batch Image Processing Using GPU Kernels

## Project Overview

This project implements batch image processing using CUDA GPU kernels. The program processes a large collection of images and performs RGB-to-grayscale conversion followed by Sobel edge detection.

The main objective is to demonstrate how image-processing operations that are performed independently for each pixel can be parallelized using thousands of GPU threads.

## Processing Pipeline

Each input image follows the following pipeline:

1. Load the RGB image.
2. Copy image data from CPU memory to GPU memory.
3. Execute the RGB-to-grayscale CUDA kernel.
4. Execute the Sobel edge-detection CUDA kernel.
5. Copy the processed image from GPU memory back to CPU memory.
6. Save the resulting image to the output directory.

## CUDA Kernels

### RGB-to-Grayscale Kernel

The grayscale kernel assigns one CUDA thread to each pixel. Each thread reads the red, green, and blue values and calculates the corresponding grayscale intensity.

### Sobel Edge Detection Kernel

The Sobel kernel assigns one CUDA thread to each pixel and examines neighboring pixels to calculate horizontal and vertical image gradients. The gradients are combined to detect edges.

Boundary checks are included to prevent out-of-bounds memory accesses.

## Dataset

The program is designed to process hundreds of input images in a single execution. The dataset contains multiple RGB images stored in the input directory.

The command-line interface allows the number of images and input/output directories to be changed without modifying the source code.

## GPU Usage

CUDA kernels are used for the computationally intensive image-processing operations. Image data is transferred between host and device memory using CUDA memory-management functions.

## Performance Analysis

The execution output records the number of images processed, total processing time, and average processing time per image. These measurements can be used to evaluate the performance of GPU-based batch image processing.

## Challenges

One challenge was handling images with different dimensions and ensuring that CUDA threads did not access memory outside the image boundaries. Boundary checks were therefore added to both kernels.

Another challenge was managing CPU-to-GPU and GPU-to-CPU memory transfers efficiently.

## Lessons Learned

This project demonstrated how image-processing operations can be divided into independent pixel-level tasks and executed in parallel using CUDA. It also showed that GPU performance depends not only on kernel execution but also on memory-transfer overhead and kernel configuration.

The project provided practical experience with CUDA kernels, thread and block organization, device memory management, and batch processing of large amounts of image data.

# STRUCTURE
```
cuda-image-processing/
│
├── README.md
├── Makefile
├── run.sh
│
├── src/
│   ├── main.cu
│   ├── grayscale.cu
│   ├── grayscale.cuh
│   ├── sobel.cu
│   └── sobel.cuh
│
├── data/
│   └── input/
│
├── output/
│   └── processed/
│
├── results/
│   ├── execution_log.txt
│   └── performance.png
│
└── screenshots/
    ├── gpu_execution.png
    ├── input_images.png
    └── output_images.png
```
# PROGRAM:
## CUDA computation
```
__global__ void rgbToGray(const unsigned char* input,
                          unsigned char* output,
                          int width,
                          int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    int rgbIndex = index * 3;

    unsigned char r = input[rgbIndex];
    unsigned char g = input[rgbIndex + 1];
    unsigned char b = input[rgbIndex + 2];

    output[index] = 0.299f * r +
                    0.587f * g +
                    0.114f * b;
}
```
## Sobel edge detection
```
__global__ void sobelFilter(const unsigned char* input,
                            unsigned char* output,
                            int width,
                            int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (x == 0 || y == 0 ||
        x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int gx =
        -input[(y - 1) * width + (x - 1)] +
         input[(y - 1) * width + (x + 1)] +
        -2 * input[y * width + (x - 1)] +
         2 * input[y * width + (x + 1)] +
        -input[(y + 1) * width + (x - 1)] +
         input[(y + 1) * width + (x + 1)];

    int gy =
        -input[(y - 1) * width + (x - 1)] -
         2 * input[(y - 1) * width + x] -
         input[(y - 1) * width + (x + 1)] +
         input[(y + 1) * width + (x - 1)] +
         2 * input[(y + 1) * width + x] +
         input[(y + 1) * width + (x + 1)];

    int magnitude = abs(gx) + abs(gy);

    output[y * width + x] =
        magnitude > 255 ? 255 : magnitude;
}
```
# OUTPUT:
```
========================================================
        CUDA BATCH IMAGE PROCESSING
========================================================

CUDA Initialization Successful

GPU Device Information
----------------------
GPU Name          : Tesla T4
Compute Capability: 7.5
Global Memory     : 15109 MB

Input Directory   : data/input
Output Directory  : output/processed
Images Found      : 250
Block Size        : 16 x 16

--------------------------------------------------------
Starting Image Processing
--------------------------------------------------------

[1/250]   image_001.jpg     -> Grayscale -> Sobel
[2/250]   image_002.jpg     -> Grayscale -> Sobel
[3/250]   image_003.jpg     -> Grayscale -> Sobel
[4/250]   image_004.jpg     -> Grayscale -> Sobel
[5/250]   image_005.jpg     -> Grayscale -> Sobel
...
[50/250]  image_050.jpg     -> Grayscale -> Sobel
...
[100/250] image_100.jpg     -> Grayscale -> Sobel
...
[150/250] image_150.jpg     -> Grayscale -> Sobel
...
[200/250] image_200.jpg     -> Grayscale -> Sobel
...
[250/250] image_250.jpg     -> Grayscale -> Sobel

--------------------------------------------------------
PROCESSING COMPLETED
--------------------------------------------------------

Total Images        : 250
Successfully Processed: 250
Failed              : 0

GPU Kernel Time     : 2.84 seconds
Total Execution Time: 5.21 seconds
Average/Image       : 20.84 ms

Output Directory:
output/processed/

All processed images saved successfully.

========================================================
                 EXECUTION SUCCESSFUL
========================================================
```
