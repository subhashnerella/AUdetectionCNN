import numpy as np
from sklearn.metrics import classification_report
from util.misc import all_gather
from collections import defaultdict
import config
import json

class Evaluator():
    def __init__(self,split = 'train'):
        #todo
        self.split = 'train'
        self.img_ids = []
        self.preds_left = []
        self.preds_right = []
        self.labels = []



    def update(self,ids,gt,pred):
        self.img_ids.extend(ids.cpu().detach().numpy())
        # process ground truth labels
        labels = gt.cpu().detach().numpy()
        self.labels.extend(labels)
        b,n_au = labels.shape
        #process model predictions
        left, right = pred #bXnc, n_au
        left = left.view(b,-1,n_au);right = right.view(b,-1,n_au) #b,nc,n_au
        left,right = left.cpu().detach().numpy(),right.cpu().detach().numpy()
        left[left>=0.5]=1;left[left<0.5]=0
        right[right>=0.5]=1;right[right<0.5]=0
        left = np.logical_or.reduce(left,axis=1) #b,n_au
        right = np.logical_or.reduce(right,axis=1) #b,n_au
        self.preds_left.extend(left)
        self.preds_right.extend(right)


    def synchronize_between_processes(self):

        self.img_ids = all_gather(self.img_ids)
        self.labels = np.vstack(all_gather(self.labels))
        self.preds_left = np.vstack(all_gather(self.preds_left))
        self.preds_right = np.vstack(all_gather(self.preds_right))


    def evaluate(self,save=False):

        #evaluate left
        left = _eval(self.preds_left,self.labels)
        right = _eval(self.preds_right,self.labels)
        total_preds = np.logical_or(self.preds_left,self.preds_right)
        report = _eval(total_preds,self.labels)
        if save:
            _savereport(left,'left_report.json')
            _savereport(right,'right_report.json')
            _savereport(report,'report.json')
        return left,right,report


def _savereport(report,filename):
    filename = './output/'+filename
    with open(filename, "w") as file_obj:
        json.dump(report, file_obj)

def _eval(preds,label):
    report = defaultdict(dict)
    for true,pred,AU in zip(label,preds,config.BP4D_AU):
        reports = classification_report(true, pred,output_dict=True,labels=[0,1],zero_division=1)
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
    return report

