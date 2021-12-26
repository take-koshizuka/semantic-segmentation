import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import EarlyStopping
from tqdm import tqdm
from pathlib import Path
import argparse

from dataset import RS21BD, get_train_augmentation, get_val_augmentation
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

def main(train_config_path, checkpoint_dir, resume_path=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(train_config_path, 'r') as f:
        cfg = json.load(f)
    fix_seed(cfg['seed'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Define dataset
    train_images_dir = Path(cfg['dataset']['root']) / cfg['dataset']['train_images_dir']
    train_masks_dir = Path(cfg['dataset']['root']) / cfg['dataset']['train_masks_dir']
    tr_ds = RS21BD(train_images_dir, train_masks_dir, 
                augmentation=get_train_augmentation(cfg['img_size']), classes=cfg['classes'])

    val_images_dir = Path(cfg['dataset']['root']) / cfg['dataset']['val_images_dir']
    val_masks_dir = Path(cfg['dataset']['root']) / cfg['dataset']['val_masks_dir']
    va_ds = RS21BD(val_images_dir, val_masks_dir, 
                augmentation=get_val_augmentation(cfg['img_size']), classes=cfg['classes'])
        
    # Define dataloader
    tr_dl = DataLoader(tr_ds, batch_size=cfg['dataset']['batch_size'], shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=cfg['dataset']['batch_size'], drop_last=False)

    # Define model
    net = MODEL_NAME[cfg['model_name'].lower()](**cfg['model'])
    model = Net(net, device)
    model.to(device)

    # Define optimizer and scheduler (optional)
    optimizer = optim.Adam( model.parameters(), lr=cfg['optim']['lr'])
    # scheduler = 

    early_stopping = EarlyStopping('avg_loss', 'min') 
    init_epochs = 1
    max_epochs = cfg['epochs']

    if AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if not resume_path == "":
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_model(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        init_epochs = checkpoint['epochs']
        model.to(device)
        
    writer = SummaryWriter()
    records = {}
    
    for i in tqdm(range(init_epochs, max_epochs + 1)):
        # training phase
        outputs = []
        for batch_idx, train_batch in enumerate(tqdm(tr_dl, leave=False)):
            optimizer.zero_grad()
            out = model.training_step(train_batch, batch_idx)
            outputs.append(out)
            if AMP:
                with amp.scale_loss(out['loss'], optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            else:
                out['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            # scheduler.step()
            del train_batch

        training_result = model.training_epoch_end(outputs)
        writer.add_scalar('data/training_loss', training_result['log']['avg_loss'])
        writer.add_scalar('data/training_acc', training_result['log']['train_Accuracy'])

        # validation phase
        outputs = []
        for batch_idx, eval_batch in enumerate(tqdm(va_dl, leave=False)):
            out = model.validation_step(eval_batch, batch_idx)
            outputs.append(out)
            del eval_batch

        val_result = model.validation_epoch_end(outputs)
        writer.add_scalar('data/val_loss', val_result['log']['avg_loss'])
        writer.add_scalar('data/val_acc', val_result['log']['val_Accuracy'])
        
        records[f'epoch {i}'] = val_result['log']

        with open(str(checkpoint_dir / f"records_elapse.json"), "w") as f:
            json.dump(records, f, indent=4)
        
        # early_stopping
        if early_stopping.judge(val_result):
            early_stopping.update(val_result)
            state_dict = model.state_dict(optimizer)
            state_dict['epochs'] = i
            early_stopping.best_state = state_dict

        if i % cfg['checkpoint_period'] == 0:
            state_dict = model.state_dict(optimizer)
            state_dict['epochs'] = i
            torch.save(state_dict, str(checkpoint_dir / f"model-{i}.pt"))
    
    best_state = early_stopping.best_state

    torch.save(best_state, str(checkpoint_dir / "best-model.pt"))
    with open(str(checkpoint_dir / "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    
    with open(str(checkpoint_dir / f"records-{init_epochs}-{max_epochs}.json"), "w") as f:
        json.dump(records, f, indent=4)
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for training.", type=str, required=True)
    parser.add_argument('-dir', '-d', help="Path to the directory where the checkpoint of the model is stored.", type=str, required=True)
    parser.add_argument('-resume', '-r', help="Path to the checkpoint of the model you want to resume training.", type=str, default="")

    args = parser.parse_args()

    ## example
    # args.config = "config/train.json"
    # args.dir = "checkpoints/tmp"
    # args.resume = "checkpoints/tmp/best-model.pt"
    ##

    main(args.config, Path(args.dir), args.resume)