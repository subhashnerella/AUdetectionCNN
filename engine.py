import torch
import os
import sys
from typing import Iterable
import util.misc as utils
import numpy as np
from sklearn.metrics import classification_report
from collections import defaultdict
import config
import json
import pdb 
import pandas as pd
from models.evaluate import Evaluator

def train_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer:torch.optim.Optimizer, 
                device:torch.device, epoch:int):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    eval = Evaluator()
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    au_loss_keys= list(map(lambda x : x+'_loss',config.BP4D_AU))
    for samples, data in metric_logger.log_every(data_loader, print_freq, header):
        samples.to(device)
        data = {k: v.to(device) for k, v in data.items()}    
        aus = data['aus'];landmarks=data['landmarks']  
        out = model(samples,landmarks)
        loss = criterion(out,aus)
        optimizer.zero_grad()
        c_loss = loss.mean()
        c_loss.backward()
        optimizer.step()
        metric_logger.update(loss=c_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        au_loss_dict = dict(zip(au_loss_keys,loss.mean(dim=0)))
        metric_logger.update(**au_loss_dict)
        eval.update(data['id'],aus,out)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    eval.synchronize_between_processes()
    metrics = eval.evaluate()
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},metrics



@torch.no_grad()
def eval_epoch(model,criterion,data_loader,device,header='val'):
    model.eval() 
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 10
    eval=Evaluator()
    au_loss_keys= list(map(lambda x : x+'_loss',config.BP4D_AU))
    for samples, data in metric_logger.log_every(data_loader,print_freq,header):
        samples.to(device)
        data = {k: v.to(device) for k, v in data.items()}
        aus = data['aus'];landmarks=data['landmarks']  
        out = model(samples,landmarks)
        loss = criterion(out,aus)
        c_loss = loss.mean()
        metric_logger.update(loss=c_loss)

        au_loss_dict = dict(zip(au_loss_keys,loss.mean(dim=0)))
        metric_logger.update(**au_loss_dict)
        eval.update(data['id'],aus,out)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    eval.synchronize_between_processes()
    metrics = eval.evaluate(save=header=='test')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},metrics     


    
@torch.no_grad()
def test_epoch(model,criterion,data_loader,device,header='test'):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    print_freq = 10
    all_gt= [];all_pred=[]

    for samples, data in metric_logger.log_every(data_loader,print_freq,header):

        samples.to(device)
        data = {k: v.to(device)  for k, v in data.items()} 
        landmarks = data['landmarks']
        left,right = model(samples,landmarks)
        #gt
        labels = data['aus'].cpu().detach().numpy()
        ids = data['id'].cpu().detach().numpy()
        all_gt.extend(np.concatenate((ids[:,None],labels),axis=1))
        #pred
        b,n_au =labels.shape
        left = left.view(b,-1,n_au);right = right.view(b,-1,n_au);
        out = torch.cat((left,right),dim=1).sigmoid()
        out  = out.cpu().detach().numpy()
        out[out>=0.5]=1;out[out<0.5]=0
        out = np.logical_or.reduce(out,axis=1)
        all_pred.extend(np.concatenate((ids[:,None],out),axis=1))
    
    all_gt = np.vstack(utils.all_gather(all_gt))
    all_pred = np.vstack(utils.all_gather(all_pred))
    
    #pdb.set_trace()
    df = pd.DataFrame(columns=['image_id']+config.BP4D_AU,data=all_gt)
    df.to_csv('output/trues_.csv',index=False) 
    df_pred = pd.DataFrame(columns =['image_id']+config.BP4D_AU,data=all_pred)
    df_pred.to_csv('output/preds_.csv',index=False) 


    all_gt = np.transpose(all_gt)
    all_pred = np.transpose(all_pred)



    report = defaultdict(dict)
    
    for true,pred,AU in zip(all_gt[1:,:],all_pred[1:,:],config.BP4D_AU):
        reports = classification_report(true, pred,output_dict=True,labels=[0,1])
        try:
            precision = reports['1']['precision']
            accuracy = reports['accuracy']#['micro avg']['f1-score']
            recall = reports['1']['recall']
            specificity = reports['0']['recall']
            f1score = reports['1']['f1-score']
            support = reports['1']['support']

            report["f1"][AU] = f1score
            report["precision"][AU] = precision
            report["acc"][AU] = accuracy
            report["recall"][AU] = recall
            report["specificity"][AU] = specificity
            report['support'][AU] = support
        except:
            pass
    #pdb.set_trace()
    print(report)
    with open("./output/BP4D_test_.json", "w") as file_obj:
        json.dump(report, file_obj)
    return report

        
        




    
