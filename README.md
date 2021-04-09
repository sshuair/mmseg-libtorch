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
    ./mmseg model.pt input_img.png output_result.png
    ```

## examples
1. PSPNet - cityscapes
   ``` bash
    # convert to libtorch .pt model
    cd /path/to/mmsegmentation
    python tools/pytorch2torchscript.py configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
    --checkpoint checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
    --output-file checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pt \
    --shape 512 1024
    
    # inference use libtorch
    cd /path/to/mmseg-libtorch
    mkdir build && cd build
    cmake -D CMAKE_PREFIX_PATH=/path/to/libtorch -D CMAKE_PREFIX_PATH=/path/to/opencv ..
    make -j4

   ./mmseg checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pt \
    assets/cityscapes/munster_000059_000019_leftImg8bit_512x1024.png \
    munster_000059_000019_leftImg8bit_512x1024_result_pytorch.png.png
   ```
   
   | image                               | PyTorch result                      | LibTorch result                     |
   |-------------------------------------|-------------------------------------|-------------------------------------|
   | ![](assets/cityscapes/munster_000059_000019_leftImg8bit_512x1024.png) | ![](assets/cityscapes/munster_000059_000019_leftImg8bit_512x1024_result_pytorch.png) | ![](assets/cityscapes/munster_000059_000019_leftImg8bit_512x1024_result_libtorch.png) |
   |                                     |                                     |                                     |



2. UNet - pascalvoc

3. FCN - ADE
   ``` bash
    # convert to libtorch .pt model
    cd /path/to/mmsegmentation
   python tools/pytorch2torchscript.py configs/fcn/fcn_r101-d8_512x512_160k_ade20k.py \
   --checkpoint checkpoints/fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pth \
   --output-file checkpoints/fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pt \
   --shape 512 771
    
    # inference use libtorch
    cd /path/to/mmseg-libtorch
    mkdir build && cd build
    cmake -D CMAKE_PREFIX_PATH=/path/to/libtorch -D CMAKE_PREFIX_PATH=/path/to/opencv ..
    make -j4

   ./mmseg checkpoints/fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pt \
    assets/ADE/ADE_val_00000143.jp \
    ADE_val_00000143_result_pytorch.png
   ```
   
   | image                               | PyTorch result                      | LibTorch result                     |
   |-------------------------------------|-------------------------------------|-------------------------------------|
   | ![](assets/ADE/ADE_val_00000143.jpg) | ![](assets/ADE/ADE_val_00000143_result_pytorch.png) | ![](assets/ADE/ADE_val_00000143_result_libtorch.png) |
   |                                     |                                     |                                     |
