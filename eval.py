import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path
import argparse
import cv2

from dataset import RS21BD, get_test_augmentation
from model import Net, MODEL_NAME
from unet import UNet
from pspnet import PSPNet

try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(checkpoint_dir, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config_path = checkpoint_dir / "train_config.json"
    with open(train_config_path, 'r') as f:
        cfg = json.load(f)
    fix_seed(cfg['seed'])
    out_dir.mkdir(exist_ok=True, parents=True)
    # Define dataset
    test_images_dir = Path(cfg['dataset']['root']) / cfg['dataset']['test_images_dir']
    te_ds = RS21BD(test_images_dir, augmentation=get_test_augmentation(cfg['img_size']), classes=cfg['classes'])

    # Define dataloader
    te_dl = DataLoader(te_ds, batch_size=1, shuffle=False)

    # Define model
    net = MODEL_NAME[cfg['model_name'].lower()](**cfg['model'])
    model = Net(net, device)
    
    checkpoint_path = checkpoint_dir / "best-model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)

    model.eval()
    
    for batch_idx, batch in enumerate(tqdm(te_dl)):
        pred, rgb_fname = model.generating_step(batch, batch_idx)
        pred = (pred > 0).long().squeeze().cpu().numpy().round()
        pred = (pred*255).astype('uint8')
        pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_CUBIC)
        # save prediction with the original image size
        cv2.imwrite(str(out_dir / (rgb_fname + '.png')), pred)

        del batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '-d', help="Path to the directory where the checkpoint of the model is stored.", type=str, required=True)
    parser.add_argument('-out_dir', '-o', help="Path to the director of the segmentation image", type=str, required=True)
    args = parser.parse_args()

    ## example
    # args.dir = "checkpoints/tmp"
    ##

    main(Path(args.dir), Path(args.out_dir))