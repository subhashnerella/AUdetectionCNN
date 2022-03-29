import argparse
import datetime, time
import json
from matplotlib.pyplot import plot
import pandas as pd
import util.misc as utils
from util.mcManager import mcManager

import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
from pathlib import Path
import config
import torchvision
from dataset.dataset import BP4D_Dataset
from engine import train_epoch, eval_epoch, test_epoch
import dataset.transform as T
from models.model import build
import neptune.new as neptune
def arg_parser():
    parser = argparse.ArgumentParser('CNN: AU detector', add_help=False)

    parser.add_argument('--aus',default=config.BP4D_AU, type= list)
    parser.add_argument('--df',default = './dataset/BP4D_comprehensive.csv', type=str)
    parser.add_argument('--datasetpath',default='./dataset/',type=str)
    parser.add_argument('--split',default='1', type= str)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=7, type=int)

    #backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--fpn',action='store_true', help = 'use a Feature Pyramid Network')
 

    #ROI extraction
    parser.add_argument('--roi_size', default='14', type=int,help = 'output size for the roi_align function')
    parser.add_argument('--crop_dim', default='48', type=int,help = 'Region of interest for each AU center on the input image')
    
    #log
    parser.add_argument('--resume',action= 'store_true')
    parser.add_argument('--eval',default=True)
    parser.add_argument('--output_dir',default='./output')

    #multigpu
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9876', help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--local_world_size", type=int, default=2)
    return parser



def main(args):
    #python -m torch.distributed.launch --nproc_per_node=2 train.py --fpn |& tee log.txt

   
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.weights = config.get_weights(args.df)
    model, criterion = build(args)
    model.cuda(args.gpu)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
                    {"params": [p for n, p in model_without_ddp.named_parameters()]}
                  ]   
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    mcmanager = mcManager()
    train_transform = T.Compose((T.FaceAlign(mcManager=mcmanager),
                                T.ToTensor(),
                                T.Normalize(),
                                T.RandomHorizontalFlip(),
                                ))
    eval_transform = T.Compose((T.FaceAlign(mcManager=mcmanager),T.ToTensor(),T.Normalize()))

    dataset_train = BP4D_Dataset(args.datasetpath+'id_train_'+args.split+'.csv',mcManager=mcmanager,transform=train_transform)
    dataset_val = BP4D_Dataset(args.datasetpath+'id_val_'+args.split+'.csv',mcManager=mcmanager,transform = eval_transform)
    dataset_test = BP4D_Dataset(args.datasetpath+'id_test_'+args.split+'.csv',mcManager=mcmanager,transform = eval_transform)

    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    sampler_test = DistributedSampler(dataset_test, shuffle=False)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)


    train_loader = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    val_loader =  DataLoader(dataset_val, args.batch_size, sampler=sampler_val,drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset_test,args.batch_size, sampler=sampler_test,drop_last=False, num_workers=args.num_workers)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    output_dir = Path(args.output_dir)
    

    # def plot_stats(stats,epoch,split='train'):
    #     for k,v in stats.items():
    #         if isinstance(v, dict):
    #             plot_stats(v,epoch,)
    #             writer.add_scalars(split+'_'+k, v, epoch + 1)
    #         else:
    #             writer.add_scalar(split+'_'+k, v, epoch + 1)
     #neptune
    if utils.is_main_process():
        run = neptune.init(api_token= config.NEPTUNE_API_TOKEN,project='subhashnerella/AUdetection')

    def plot_stats(stats,split='train'):
        for k,v in stats.items():
            if isinstance(v, dict):
                plot_stats(v,split = split+'/'+str(k))
            elif 'test' in split:
                run[split+'/'+str(k)] = v
            else:
                run[split+'/'+str(k)].log(v)


    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats,train_metrics = train_epoch(model,criterion,train_loader,optimizer,device,epoch)

        lr_scheduler.step()
        if args.output_dir and  utils.is_main_process():
            plot_stats(train_stats)
            left,right,total = train_metrics
            plot_stats(left,'train_left')
            plot_stats(right,'train_right')
            plot_stats(total,'train_total')
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        val_stats,metrics = eval_epoch(model, criterion, val_loader,device)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            plot_stats(val_stats,'val')
            left,right,total = metrics
            plot_stats(left,'val_left')
            plot_stats(right,'val_right')
            plot_stats(total,'val_total')

            with (output_dir / "run_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
        
    _,metrics=eval_epoch(model,criterion,test_loader,device,header='test')
    if utils.is_main_process():
        left,right,total = metrics
        plot_stats(left,'val_left')
        plot_stats(right,'val_right')
        plot_stats(total,'val_total')
        run.stop()
    
    return

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args)