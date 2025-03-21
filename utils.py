import os
import logging
import subprocess
from pathlib import Path
from math import exp
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn

# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

def gaussian(window_size, sigma) :
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel) :
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim_map(img1, img2, window, window_size, channel) :
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

def calc_ssim(img1, img2, window_size = 11, size_average = True) :
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    # img1 = img1.type(torch.float32)
    # img2 = img2.type(torch.float32)
    ssim_map = _ssim_map(img1, img2, window, window_size, channel)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def log_to_gray(img):
    grayImgLog2exp = torch.exp(img)
    grayImgLog2exp =  (torch.round(grayImgLog2exp))
    return grayImgLog2exp
    
def calc_psnr(img1, img2):    
    return 10. * torch.log10(1. / torch.mean((img1 - img2)**2))

# ENL: Equivalent number of looks
# To be computed in the homogeneous region only    
def calc_enl(img1):       
    img1 = img1.double()
    mean_est = torch.nanmean(img1)        
    var_est = torch.var(img1)      
    enl = torch.square(mean_est) / (var_est+0.00001)    
    return enl
    
def calc_mob(imgN, imgD):
    imgN = imgN.double()
    imgD = imgD.double()
    numtr = torch.sub(imgN,imgD)
    term = torch.div(numtr,imgN)
    mob = torch.mean(term)
    return mob
    
# COV: Coefficient of variation
# To be computed in the homogeneous region     
def calc_cov(imgD):
    # assert not torch.isnan(imgN).any()
    # assert not torch.isnan(imgD).any()
    imgD = imgD.double()
    std_patch = torch.std(imgD)
    mean_val = torch.nanmean(imgD)
    cov_val = std_patch/(mean_val+0.0000001)    
    return cov_val
    
# MoR: Mean of ratio image
# To be computed on the entire image     
def calc_mor(imgN, imgD):
    # assert not torch.isnan(imgN).any()
    # assert not torch.isnan(imgD).any()    
    imgN = imgN.double()
    imgD = imgD.double()
    ratio_img = imgN/(imgD+0.0000001)
    mean_val = torch.nanmean(ratio_img)
    var_val = torch.var(ratio_img)    
    return mean_val,var_val
    
# MoI: Mean of Image
# To be computed in the homogeneous region only     
def calc_moi(imgN, imgD):
    # assert not torch.isnan(imgN).any()
    # assert not torch.isnan(imgD).any()   
    imgN = imgN.double()
    imgD = imgD.double()
    numtr = torch.nanmean(imgN)
    den = torch.nanmean(imgD)
    term = torch.div(numtr,den+0.0000001)    
    return term
    
def calc_epdroa(imgN,imgD):
    print("Shape of noisy heterogeneous patch:\t",imgN.shape)
    img_sizehh,img_sizehw = imgN.shape
    epdroa_hdN = torch.empty((img_sizehh,img_sizehw-1))
    epdroa_hdD = torch.empty((img_sizehh,img_sizehw-1))
    epdroa_vdN = torch.empty((img_sizehh-1,img_sizehw))
    epdroa_vdD = torch.empty((img_sizehh-1,img_sizehw))
    eps = 0.0000001
    for i in range(imgN.shape[0]):
        for j in range(imgN.shape[1]-1):
            epdroa_hdN[i,j] = (imgN[i,j]/(imgN[i,j+1]+eps))
            epdroa_hdD[i,j] = (imgD[i,j]/(imgD[i,j+1]+eps))            
    epd_roa_hd = torch.sum(torch.abs(epdroa_hdD))/torch.sum(torch.abs(epdroa_hdN))
    
    for i in range(imgN.shape[0]-1):
        for j in range(imgN.shape[1]):
            epdroa_vdN[i,j] = (imgN[i,j]/(imgN[i+1,j]+eps))
            epdroa_vdD[i,j] = (imgD[i,j]/(imgD[i+1,j]+eps))
    epd_roa_vd = torch.sum(torch.abs(epdroa_vdD))/torch.sum(torch.abs(epdroa_vdN))        
    return epd_roa_hd,epd_roa_vd
    
def calc_dg(imgC,imgN, imgD):
    # assert not torch.isnan(imgN).any()
    # assert not torch.isnan(imgD).any()
    imgC = imgC.double()
    imgN = imgN.double()
    imgD = imgD.double()
    numtr = torch.mean((imgC - imgN)**2)
    den = torch.mean((imgC - imgD)**2)
    term = 10. * torch.log10(torch.div(numtr,den+0.0000001))    
    return term

class AverageMeter(object):
    def __init__(self):
        self.reset()

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

def set_seed(seed) :
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# https://github.com/ultralytics/yolov5/blob/develop/utils/torch_utils.py

try :
    import thop  # for FLOPS computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)

def git_describe() :
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    if Path('.git').exists() :
        return subprocess.check_output('git describe --tags --long --always', shell=True).decode('utf-8')[:-1]
    else:
        return ''

def select_device(project_name, device = "", batch_size = None) :
    # device = 'cpu' or '0' or '0,1,2,3'1
    s = f'{project_name} {git_describe()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu :
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device :  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda :
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else :
        s += 'CPU\n'

    logger.info(s)  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)