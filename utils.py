import random
import numpy as np
import torch
import os 
import shutil
import glob 
import sys
def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False

class AverageMeter (object):
    def __init__(self):
        self.reset ()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  
     
def copy_sourcefile(output_dir, src_dir = 'src' ):    
    source_dir = os.path.join(output_dir, src_dir)

    os.makedirs(source_dir, exist_ok=True)
    org_files1 = os.path.join('./', '*.py' )
    org_files2 = os.path.join('./', '*.sh' )
    org_files3 = os.path.join('./', '*.ipynb' )
    org_files4 = os.path.join('./', '*.txt' )
    org_files5 = os.path.join('./', '*.json' )    
    files =[]
    files = glob.glob(org_files1 )
    files += glob.glob(org_files2  )
    files += glob.glob(org_files3  )
    files += glob.glob(org_files4  ) 
    files += glob.glob(org_files5  )     

    # print("COPY source to output/source dir ", files)
    tgt_files = os.path.join( source_dir, '.' )
    for i, file in enumerate(files):
        shutil.copy(file, tgt_files)