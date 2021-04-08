# MMSeg to LibTorch Examples 

- Convert mmseg pytorch model to torchscript mdoel. 
- Loading the torchscript model in C++.

## usage
1. Convert the mmsegmentation pytorch model(.pth) to libtorch model(.pt)
    - follow the install of mmsegmetation
    - use the mmseg `tools/pytorch2torchscript.py` convert to model to libtorch `.pt` model.
2. clone this repo and complie local.
    ``` bash
    git clone 

    # compile 
    mkdir build && cd build
    cmake -D CMAKE_PREFIX_PATH=/path/to/libtorch -D CMAKE_PREFIX_PATH=/path/to/opencv ..
    make -j4

    # run the model 
    ./mmseg model.pt img.png
    ```

## examples
1. HRNet - cityscapes
   ``` bash
    # convert to libtorch .pt model
    cd /path/to/mmsegmentation
    python tools/pytorch2torchscript.py configs/hrnet/fcn_hr18_512x512_40k_voc12aug.py --checkpoint checkpoints/fcn_hr18s_512x512_40k_voc12aug_20200614_000648-4f8d6e7f.pth --output-file fcn_hr18s.onnx

    mkdir build && cd build
    cmake -D CMAKE_PREFIX_PATH=/path/to/libtorch -D CMAKE_PREFIX_PATH=/path/to/opencv ..
    make -j4


   ```
| image                               | Pytorch result                      | Libtorch result                     |
|-------------------------------------|-------------------------------------|-------------------------------------|
| ![](assets/results/stuttgart03.png) | ![](assets/results/stuttgart03.png) | ![](assets/results/stuttgart03.png) |
|                                     |                                     |                                     |



2. UNet - pascalvoc

3. 