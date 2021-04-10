import argparse
import itertools

import numpy as np
import torch
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from PIL import Image


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def ndarray2tensor(img:np.ndarray):
    img = img.transpose((2,0,1))

    input_tensor = torch.from_numpy(img)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor=input_tensor.type(torch.float)
    mean=torch.tensor([123.675, 116.28, 103.53])
    std=torch.tensor([58.395, 57.12, 57.375])
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    input_tensor.sub_(mean).div_(std)

    return input_tensor

def tensor2img(pred, palette):
    pred = pred.argmax(1).squeeze()
    palette = get_palette(palette)
    palette = list(itertools.chain.from_iterable(palette))
    out_img = Image.fromarray(pred.numpy().astype(np.uint8)).convert('L')
    out_img.putpalette(palette)
    return out_img
    

def parse_args():
    parser = argparse.ArgumentParser("predict result use mmseg pytorch version")
    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint of pytorch(.pth)')
    parser.add_argument('--imgfp', help='the input image')
    parser.add_argument('--output', help='the output png result of predict')
    parser.add_argument('--palette', help='the palette of dataset, voc, cityscapes, ade')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # init model
    model = init_segmentor(args.config, args.checkpoint, device='cpu')
    model = _convert_batchnorm(model)
    model.eval()

    # prepare the input tensor
    img = np.array(Image.open(args.imgfp))
    input_tensor = ndarray2tensor(img)

    # predict
    result = model.forward_dummy(input_tensor)
    out_img = tensor2img(result, args.palette)
    out_img.save(args.output)
