import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from PromptModels_pool_eval.GetPromptModel import build_promptmodel
import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from timm.scheduler import CosineLRScheduler

from datasets import create_datasets
from sampler import ClassPrioritySampler, ClassAwareSampler
from utils import *

from loss import ASLSingleLabel, EQLv2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--split', type=str, default='1000')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--base_lr', type=float, default=0.02)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--warmup_epochs', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--prompt_length', type=int, default=10)
parser.add_argument('--name', type=str, default='vpt_deep')
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lr = args.base_lr * args.batch_size / 256
    # batch_size=2
    # edge_size=384
    # data = torch.randn(batch_size, 3, args.image_size, args.image_size)
    # labels = torch.ones(batch_size).long()  # long ones
    # different from DeiT, the original ViT-X are trained with Inception norm params, i.e., all 0.5
    norm_params = {'mean': [0.5, 0.5, 0.5],
                       'std': [0.5, 0.5, 0.5]}
    #norm_params = {'mean': [0.485, 0.456, 0.406],
    #                   'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**norm_params)
    train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    val_transforms = transforms.Compose([
            #transforms.Resize((args.image_size, args.image_size)),
            transforms.Resize((args.image_size * 8 // 7, args.image_size * 8 // 7)),
            transforms.CenterCrop((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset, val_dataset, num_classes = create_datasets(args.data_path, train_transforms, val_transforms, args.dataset, args.split)
    log(f"train dataset: {len(train_dataset)} samples")
    log(f"val dataset: {len(val_dataset)} samples")
    train_cls_num_list = np.array(list(train_dataset.label_freq.values()))
    test_cls_num_list = np.array(list(val_dataset.label_freq.values()))
    many_shot = train_cls_num_list > 100
    medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
    few_shot = train_cls_num_list < 20
    # tran_sampler = ClassPrioritySampler(train_dataset, manual_only=True)
    train_sampler = ClassAwareSampler(train_dataset, num_samples_cls=4)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False)

    model = build_promptmodel(num_classes=num_classes, img_size=args.image_size, base_model=args.base_model, model_idx='ViT', patch_size=16,
                            Prompt_Token_num=args.prompt_length, VPT_type="Deep")  # VPT_type = "Shallow"
    # test for updating
    ckpt = torch.load('LPT_places.pth', 'cpu')['state_dict']
    print(ckpt.keys())
    if list(ckpt.keys())[-1].startswith('module'):
       ckpt_new = {}
       for key in ckpt.keys():
           ckpt_new[key[7:]] = ckpt[key]
       ckpt = ckpt_new
    model.load_state_dict(ckpt)
    #fix all params
    for m in model.parameters():
        m.requires_grad = False
    # model.prompt_learner.head = nn.Linear(768, num_classes)
    # for m in model.prompt_learner.head.parameters():
    #     m.requires_grad = True
    model = model.to(device)
    model = torch.nn.parallel.DataParallel(model)
    # criterion = nn.CrossEntropyLoss()

    # preds = model(data)  # (1, class_number)
    # print('before Tuning model output：', preds)

    # check backwarding tokens
    for param in model.parameters():
        if param.requires_grad:
            print(param.shape)
    max_va = -1
    #fix all params in eval mode
    model.eval()
    total_outputs = []
    total_feats = []
    total_labels = []
    total_prompts = []
    for imgs, targets in tqdm(val_loader, desc='val', leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs, _, topk, feat = model(imgs)
            _, preds = outputs.detach().cpu().topk(1, 1, True, True)
            preds = preds.squeeze(-1)
            total_outputs.append(preds)
            total_labels.append(targets.detach().cpu())
            total_prompts.append(topk.detach().cpu())
            total_feats.append(feat.detach().cpu())
    total_outputs = torch.cat(total_outputs, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    total_prompts = torch.cat(total_prompts, dim=0)
    total_feats = torch.cat(total_feats, dim=0)
    unique, cnt = total_prompts.unique(return_counts=True)
    # per-class evaluation
    shot_cnt_stats = {
        'total': [0, train_cls_num_list.max(), 0, 0, 0.],
        'many': [100, train_cls_num_list.max(), 0, 0, 0.],
        'medium': [20, 100, 0, 0, 0.],
        'few': [0, 20, 0, 0, 0.],
    }
    for l in torch.unique(total_labels):
        class_correct = torch.sum((total_outputs[total_labels == l] == total_labels[total_labels == l])).item()
        test_class_count = len(total_labels[total_labels == l])
        for stat_name in shot_cnt_stats:
            stat_info = shot_cnt_stats[stat_name]
            if train_cls_num_list[l] > stat_info[0] and train_cls_num_list[l] <= stat_info[1]:
                stat_info[2] += class_correct
                stat_info[3] += test_class_count
    for stat_name in shot_cnt_stats:
        shot_cnt_stats[stat_name][-1] = shot_cnt_stats[stat_name][2] / shot_cnt_stats[stat_name][3] * 100.0 if shot_cnt_stats[stat_name][3] != 0 else 0.
    per_cls_eval_str = 'epoch {}, overall: {:.5f}%, many-shot: {:.5f}%, medium-shot: {:.5f}%, few-shot: {:.5f}%'.format(1, shot_cnt_stats['total'][-1], shot_cnt_stats['many'][-1], shot_cnt_stats['medium'][-1], shot_cnt_stats['few'][-1])
    log(per_cls_eval_str)
        
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
