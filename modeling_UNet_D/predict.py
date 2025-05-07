import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.nn as nn

from utils.data_loading import BasicDataset
from UNet_D import UNet_D
import segmentation_models_pytorch as smp

from utils.utils import plot_img_and_mask
from utils.dice_score import dice_coeff,dice

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='.../checkpoints/checkpoint.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    src_test_dir='.../data/kvasir_split/test/images/'
    src_test_gt='.../data/kvasir_split/test/masks/'
    in_files = '.../data/kvasir_split/cju0u82z3cuma0835wlxrnrjv.jpg'
    out_files = '.../data/kvasir_split/predictions/'

    net = UNet_D(num_classes=1, BatchNorm=nn.BatchNorm2d)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    
    root,dirs,files = next(os.walk(src_test_dir))

    i=0
    dice_array=[]
    for filename in files:
        logging.info(f'Predicting image {filename} ...')
        print(filename)
        img = Image.open(src_test_dir+filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        gt=np.asarray(Image.open(src_test_gt+filename))
        
        out_filename = out_files+filename
        result = mask_to_image(mask, mask_values)
        print('unique mask = ', np.unique(mask))
        print('shape mask = ', np.shape(mask))
        
        [sh0,sh1]=np.shape(gt)
        new_gt=np.zeros((sh0,sh1))
        [r,c]=np.where(gt>=246)
        new_gt[r,c]=1
        
        print('unique gt = ', np.unique(new_gt))
        print('gt shape = ', np.shape(new_gt))
        
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')
        
        #----compute dice_score------------------------------------------------
        mydice=dice(mask, new_gt)
        print('np unique new_gt = ',np.unique(new_gt))
        print('dice = ',dice)
        dice_array.append(mydice)

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
        i=i+1
print('Final average dice over test set = ', np.mean(np.asarray(dice_array)))
