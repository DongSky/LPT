"""
Unofficial code for VPT(Visual Prompt Tuning) paper of arxiv 2203.12119

A toy Tuning process that demostrates the code

the code is based on timm

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from PromptModels.GetPromptModel import build_promptmodel
import argparse

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from timm.scheduler import CosineLRScheduler

from datasets import create_datasets
from utils import *
from timm.models.vision_transformer import vit_base_patch16_224_in21k

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='flowers102')
parser.add_argument('--split', type=str, default='1000')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--prompt_length', type=int, default=10)
parser.add_argument('--name', type=str, default='vitb_full_ft_check')
parser.add_argument('--base_model', type=str, default='vit_base_patch16_224_in21k')

def setup_seed(seed):  # setting up the random seed
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    setup_seed(42)
    save_path = os.path.join('./save', args.name)
    ensure_path(save_path)
    set_log_path(save_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lr = args.base_lr * args.batch_size / 256
    # labels = torch.ones(batch_size).long()  # long ones
    norm_params = {'mean': [0.5, 0.5, 0.5],
                       'std': [0.5, 0.5, 0.5]}
    normalize = transforms.Normalize(**norm_params)
    train_transforms = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    val_transforms = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            #transforms.CenterCrop((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset, val_dataset, num_classes = create_datasets(args.data_path, train_transforms, val_transforms, args.dataset, args.split)
    log(f"train dataset: {len(train_dataset)} samples")
    log(f"val dataset: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False)

    # model = build_promptmodel(num_classes=num_classes, img_size=args.image_size, base_model=args.base_model, model_idx='ViT', patch_size=16,
    #                         Prompt_Token_num=args.prompt_length, VPT_type="Deep")  # VPT_type = "Shallow"
    model = vit_base_patch16_224_in21k(pretrained=True)
    model.head = nn.Linear(768, num_classes)
    # test for updating
    # prompt_state_dict = model.obtain_prompt()
    # model.load_prompt(prompt_state_dict)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = CosineLRScheduler(optimizer, warmup_lr_init=1e-4, t_initial=args.epochs, cycle_decay=0.1, warmup_t=args.warmup_epochs)
    criterion = nn.CrossEntropyLoss()

    # preds = model(data)  # (1, class_number)
    # print('before Tuning model output：', preds)

    # check backwarding tokens
    for param in model.parameters():
        if param.requires_grad:
            print(param.shape)
    max_va = -1
    for epoch in range(args.epochs):
        print('epoch:',epoch)
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: Averager() for k in aves_keys}
        iter_num = 0
        model.train()#Freeze()
        for imgs, targets in tqdm(train_loader, desc='train', leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            acc = compute_acc(outputs, targets)
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            iter_num += 1
        # print()
        model.eval()
        for imgs, targets in tqdm(val_loader, desc='val', leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            acc = compute_acc(outputs, targets)
            aves['vl'].add(loss.item())
            aves['va'].add(acc)
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch, aves['tl'].v, aves['ta'].v)
        log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'].v, aves['va'].v)
        log(log_str)
        # preds = model(data)  # (1, class_number)
        print('After Tuning model output: ', aves['va'].v)
        save_obj = {
            'config': vars(args),
            'state_dict': model.state_dict(),
            'val_acc': aves['va'].v,
        }
        if epoch <= args.epochs:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
            torch.save(save_obj, os.path.join(
                save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'].v > max_va:
                max_va = aves['va'].v
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))
        scheduler.step(epoch)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
