import argparse
import datetime
import json
import numpy as np
import os
import time
from scipy import stats
from pathlib import Path
import torch
import torchvision.transforms as transforms
import timm
from torchvision import datasets
import util.misc as misc
import models_mae_shared
from TTT_main import load_combined_model
from engine_pretrain import accuracy
from einops import repeat
import tqdm
import os.path
from torch.utils.data import Dataset
from PIL import Image

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, targets, transform=None):
        self.tensors = tensors
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]
        y = self.targets[index]
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # Remove singleton dimensions
        x = np.squeeze(x)
        # If shape is (3, H, W), transpose to (H, W, 3)
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.transpose(1, 2, 0)
        # Convert to uint8 if needed
        if x.dtype != np.uint8:
            x = (x * 255).clip(0, 255).astype(np.uint8)
        if self.transform:
            x = self.transform(Image.fromarray(x))        
        
        return x, y

    def __len__(self):
        return len(self.tensors)
    
def get_args_parser():
    parser = argparse.ArgumentParser('MAE testing.', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--classifier_depth', type=int, metavar='N', default=0,
                        help='number of blocks in the classifier')
    parser.add_argument('--resume_model', default='vit-B-ft-US.pt', help='resume from checkpoint')
    parser.add_argument('--resume_finetune', default='vit-B-ft-US.pt', help='resume from checkpoint')
    
    # parser.add_argument('--resume_model', default='', required=True, help='resume from checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    # For working with the original main_test_time_training.py:
    parser.add_argument('--load_optimizer', action='store_true')
    parser.set_defaults(load_optimizer=False)
    parser.add_argument('--load_loss_scalar', action='store_true')
    parser.set_defaults(load_loss_scalar=False)
    parser.add_argument('--predict_rotations', action='store_true',
                        help='Predict rotations.')
    parser.set_defaults(predict_rotations=False)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    # parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--head_type', default='vit_head',
                        help='Head type - linear or vit_head')
    parser.add_argument('--num_workers', default=10, type=int)
    
    return parser


def main(args):
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    
    # dataset_val = datasets.ImageFolder(args.data_path, transform=transform_val)
    classes = 2

    if not 'USIplateOK' in locals():
        exec(open('USdataRead.py').read(), globals())
    
    global USIplateNG, USIplateNGlabel, USIpipeNG, USIpipeNGlabel
    USIplateNGlabel = USIplateNGlabel.flatten().long()
    USIpipeNGlabel = USIpipeNGlabel.flatten().long()
    
    # USIlab = USIpipeNGlabel
    
    ds_val = CustomTensorDataset(USIplateNG, USIplateNGlabel, transform=transform_val)    
    
    ds_val = CustomTensorDataset(USIpipeNG, USIpipeNGlabel, transform=transform_val)
    
    # Set num_workers=0 to avoid multiprocessing issues
    dataset_val = torch.utils.data.DataLoader(ds_val, 32, shuffle=False, num_workers=0)
    
    print(f'Using dataset {args.data_path} with {len(dataset_val)}') 
    model, _, _ = load_combined_model(args, classes)
    _ = model.to(args.device)
    all_acc = []
    all_losses = []
    # model.eval()
    data_len = len(dataset_val)
    # for index in range(data_len):
    #     # Get the samples:
    #     current_idx = index
    #     samples, labels = dataset_val[current_idx]
    #     samples = samples.to(args.device, non_blocking=True).unsqueeze(0)
    #     labels = torch.LongTensor([labels]).to(args.device, non_blocking=True)
    #     with torch.no_grad():
    #         loss_dict, _, _, pred = model(samples, target=labels, mask_ratio=0)
    #         acc1 = (stats.mode(pred.argmax(axis=1).detach().cpu().numpy()).mode[0] == labels[0].cpu().detach().numpy()) * 100.
    #     all_acc.append(acc1)
    #     all_losses.append(float(loss_dict['classification'].detach().cpu().numpy()))
    #     if index % 4000 == 1:
    #         print(np.mean(all_acc[-classes:]))
    #         print(np.mean(all_losses[-classes:]))
            
    for index, (samples, labels) in enumerate(dataset_val):
        samples = samples.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        # loss_dict, _, _, pred = model(samples, target=labels, mask_ratio=0)
        with torch.no_grad():
            loss_dict, _, _, pred = model(samples, target=labels, mask_ratio=0)
            preds = pred.argmax(axis=1).detach().cpu().numpy()
            acc1 = (labels.cpu().detach() == preds).sum().item()/len(labels)
            # acc1 = (stats.mode(pred.argmax(axis=1).detach().cpu().numpy()).mode[0] == labels[0].cpu().detach().numpy()) * 100.
        all_acc.append(acc1)
        all_losses.append(float(loss_dict['classification'].detach().cpu().numpy()))
        # if index % 4000 == 1:
        # print(np.mean(all_acc[-classes:]))
        # print(np.mean(all_losses[-classes:]))
        print(f"Mean acc: {np.mean(all_acc[-classes:])}, Mean loss: {np.mean(all_losses[-classes:])}")
            
    print('Saving to', os.path.join(args.output_dir, 'accuracy.txt'))
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        f.write(f'{np.mean(all_acc)} {np.mean(all_losses)}\n')
    with open(os.path.join(args.output_dir, 'accuracy.npy'), 'wb') as f:
        np.save(f, np.array(all_acc))
        
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
