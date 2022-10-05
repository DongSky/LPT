"""
Unofficial code for VPT(Visual Prompt Tuning) paper of arxiv 2203.12119

A toy Tuning process that demostrates the code

the code is based on timm

"""
# from msilib.schema import Class
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from PromptModels.GetPromptModel import build_promptmodel
from PromptModels_pool.GetPromptModel import build_promptmodel as build_promptmodel_pool
import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from timm.scheduler import CosineLRScheduler

from datasets import create_datasets
from sampler import ClassPrioritySampler, ClassAwareSampler, BalancedDatasetSampler, CBEffectNumSampler
from utils import *

from loss import ASLSingleLabel, EQLv2
from cb_loss import AGCL

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
parser.add_argument('--tau', type=float, default=1.0, help='logit adjustment factor')

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
    # batch_size=2
    # edge_size=384
    # data = torch.randn(batch_size, 3, args.image_size, args.image_size)
    # labels = torch.ones(batch_size).long()  # long ones
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
            transforms.Resize((args.image_size * 8 // 7, args.image_size * 8 // 7)),
            transforms.CenterCrop((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,  
        ])

    train_dataset, val_dataset, num_classes = create_datasets(args.data_path, train_transforms, val_transforms, args.dataset, args.split)
    log(f"train dataset: {len(train_dataset)} samples")
    log(f"val dataset: {len(val_dataset)} samples")
    
    label_freq_array = np.array(list(train_dataset.label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    train_cls_num_list = np.array(list(train_dataset.label_freq.values()))
    test_cls_num_list = np.array(list(val_dataset.label_freq.values()))
    many_shot = train_cls_num_list > 100
    medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
    few_shot = train_cls_num_list < 20
    adjustments = np.log(label_freq_array ** args.tau + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    criterion = AGCL(cls_num_list=list(train_dataset.label_freq.values()), m=0.1, s=20, weight=None, train_cls=False, noise_mul=0.5, gamma=4.)
    criterion_ibs = AGCL(cls_num_list=list(train_dataset.label_freq.values()), m=0.1, s=20, weight=None, train_cls=False, noise_mul=0.5, gamma=4., gamma_pos=0.5, gamma_neg=8.0)
    # criterion = ASLSingleLabel().to(device)#EQLv2(num_classes=num_classes).to(device)
    # tran_sampler = ClassPrioritySampler(train_dataset, manual_only=True)
    train_sampler = CBEffectNumSampler(train_dataset)
    train_sampler = ClassAwareSampler(train_dataset, num_samples_cls=4)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=False, drop_last=True)
    train_loader_ibs = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), num_workers=4, shuffle=False)
    extractor = build_promptmodel(num_classes=num_classes, img_size=args.image_size, base_model=args.base_model, model_idx='ViT', patch_size=16,
                            Prompt_Token_num=args.prompt_length, VPT_type="Deep")
    ckpt = torch.load('phase1.pth', 'cpu')['state_dict']
    if list(ckpt.keys())[0].startswith('module'):
       ckpt_new = {}
       for key in ckpt.keys():
           ckpt_new[key[7:]] = ckpt[key]
       ckpt = ckpt_new
    extractor.load_state_dict(ckpt)
    extractor.prompt_learner.head = nn.Identity()
    model = build_promptmodel_pool(num_classes=num_classes, img_size=args.image_size, base_model=args.base_model, model_idx='ViT', patch_size=16,
                            Prompt_Token_num=args.prompt_length, VPT_type="Deep")  # VPT_type = "Shallow"
    # test for updating
    #prompt_state_dict = model.obtain_prompt()
    #model.load_prompt(prompt_state_dict)
    extractor = extractor.to(device)
    model.load_state_dict(ckpt, strict=False)
    for param in extractor.parameters():
        param.requires_grad_(False)
    extractor.eval()
    model = model.to(device)
    model.Freeze()
    model.prompt_learner.Prompt_Tokens.requires_grad_(False)
    model = torch.nn.parallel.DataParallel(model)
    optimizer = optim.SGD(model.module.prompt_learner.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = CosineLRScheduler(optimizer, warmup_lr_init=1e-6, t_initial=args.epochs, cycle_decay=0.1, warmup_t=args.warmup_epochs)
    # criterion = nn.CrossEntropyLoss()

    # preds = model(data)  # (1, class_number)
    # print('before Tuning model output：', preds)

    # check backwarding tokens
    for param in model.parameters():
        if param.requires_grad:
            print(param.shape)
    max_va = -1
    #ibs_loss_weight = torch.range(start=0, end=0.4, step=41)[::-1]
    for epoch in range(args.epochs):
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: Averager() for k in aves_keys}
        iter_num = 0
        model.train()
        # model.Freeze()
        cnt = 0
        for imgs, targets in tqdm(train_loader, desc='train', leave=False):
            imgs_ibs, targets_ibs = next(iter(train_loader_ibs))
            if cnt > len(train_dataset) // args.batch_size:
                break
            cnt += 1
            imgs = imgs.to(device)
            targets = targets.to(device)
            imgs_ibs, targets_ibs = imgs_ibs.to(device), targets_ibs.to(device)
            optimizer.zero_grad()
            outputs, reduced_sim = model(imgs)
            outputs_ibs, reduced_sim_ibs = model(imgs_ibs)
            loss = criterion(outputs, targets) - 0.5 * reduced_sim + max(0.0, (0.5 * (args.epochs - epoch) / args.epochs)) * (criterion_ibs(outputs_ibs, targets_ibs) - 0.5 * reduced_sim_ibs)
            loss.backward()
            optimizer.step()
            acc = compute_acc(outputs, targets)
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            iter_num += 1
        # print()
        model.eval()
        total_outputs = []
        total_labels = []
        for imgs, targets in tqdm(val_loader, desc='val', leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs, reduced_sim = model(imgs)
                _, preds = outputs.detach().cpu().topk(1, 1, True, True)
                preds = preds.squeeze(-1)
                total_outputs.append(preds)
                total_labels.append(targets.detach().cpu())
                loss = criterion(outputs, targets)
            acc = compute_acc(outputs, targets)
            aves['vl'].add(loss.item())
            aves['va'].add(acc)
        total_outputs = torch.cat(total_outputs, dim=0)
        total_labels = torch.cat(total_labels, dim=0)
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
        per_cls_eval_str = 'epoch {}, overall: {:.5f}%, many-shot: {:.5f}%, medium-shot: {:.5f}%, few-shot: {:.5f}%'.format(epoch, shot_cnt_stats['total'][-1], shot_cnt_stats['many'][-1], shot_cnt_stats['medium'][-1], shot_cnt_stats['few'][-1])
        log(per_cls_eval_str)
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch, aves['tl'].v, aves['ta'].v)
        log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'].v, aves['va'].v)
        log(log_str)
        # preds = model(data)  # (1, class_number)
        print('After Tuning model output：', aves['va'].v)
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
        scheduler.step(epoch+1)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

