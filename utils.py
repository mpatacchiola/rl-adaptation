import torch
from datetime import datetime
import os

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class Logger():
    def __init__(self, header, file_path, file_name=None, id="", verbose=True):
        if not os.path.exists(file_path): 
            os.makedirs(file_path)
            
        if(file_name is not None):
            self.file_name=file_name
        else:
            now = datetime.now()
            self.file_name = file_path + "/" + now.strftime("%H%M%S_%d%m%Y")
            
        if(id!=""): self.file_name += "_" + str(id) # Adding the ID string
        self.file_name +=  ".csv" # Adding the extension
        
        self.buffer = list()
        
        f = open(self.file_name, "a")
        f.write(header + "\n")
        f.close()
        if(verbose): print("[INFO] Log file created:", self.file_name)
    
    def append(self, *args):
        text = ",".join(list(args))
        self.buffer.append(text + "\n")           

    def write(self):
        f = open(self.file_name, "a")
        for line in self.buffer: f.write(line)
        f.close()
        self.buffer.clear()
    
