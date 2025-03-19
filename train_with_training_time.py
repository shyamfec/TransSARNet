import argparse
import copy
from os import listdir, getcwd, mkdir
from os.path import join
import wandb
from datasets import DataFromFolder
from TransSARNet import Transformer
from utils import *
from export_image import concat_image
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from time import time
import pandas as pd
from einops import rearrange
import math

from torch.cuda.amp import autocast, GradScaler

def time_conversion(in_time):
    temp = in_time
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    print('%d:%d:%d' %(hours,minutes,seconds))
    return hours,minutes,seconds
    
def main() :
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type = str, required = True)
    parser.add_argument("--model-name", type = str, default = "TransSARNet")
    # parser.add_argument("--weights-dir", type = str, required = True)
    parser.add_argument("--noisy-train-dir", type = str, required = True)
    parser.add_argument("--clean-train-dir", type = str, required = True)
    parser.add_argument("--noisy-valid-dir", type = str, required = True)
    parser.add_argument("--clean-valid-dir", type = str, required = True)
    parser.add_argument("--input-shape", type = int, default = 64)
    parser.add_argument("--scale", type = int, default = 2)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--epochs", type = int, default = 30)
    parser.add_argument("--seed", type = int, default = 123)
    parser.add_argument("--num-gpu", type = int, default = 1)
    parser.add_argument("--device", default = "", help = "cuda device, i.e. 0 or 0,1,2,3 or cpu")
    args = parser.parse_args()

    # Get Current Namespace
    print(args)

    # Initialize Weights & Biases Library
    wandb.init(
    config = args,
    resume = "never",
    project = args.project
    )

    # Initialize Project Name
    wandb.run.name = args.model_name

    # Assign Device
    set_logging()
    device = select_device(args.model_name, args.device)

    # Set Seed
    set_seed(args.seed)

    # Create Model Instance
    model = Transformer().to(device)
    
    # Set Seed
    set_seed(args.seed)

    # Load Dataset
    train_dataset = DataFromFolder(
                        args.noisy_train_dir,
                        args.clean_train_dir,
                        "train",
                        args.seed
                        )
    valid_dataset = DataFromFolder(
                        args.noisy_valid_dir,
                        args.clean_valid_dir,
                        "valid",
                        args.seed
                        )

    # Create Pytorch DataLoader Instance
    train_dataloader = DataLoader(
                            train_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = 4 * args.num_gpu,
                            pin_memory = True,
                            drop_last = True)
    valid_dataloader = DataLoader(
                            valid_dataset,
                            batch_size = args.batch_size,
                            shuffle = False,
                            num_workers = 4 * args.num_gpu,
                            pin_memory = True,
                            drop_last = True
                            )

    # Get Parameters of Current Model
    #print(summary(model, (1, args.input_shape, args.input_shape), batch_size = args.batch_size))

    # Create Optimizer Instance
    optimizer = optim.Adam(model.parameters(),
                            lr = 0.0001, weight_decay = 0.00001
                            )
                      

    # # Let wandb Watch Training Process
    wandb.watch(model)

    # # Create Learning Rate Scheduler Instance    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau( 
                optimizer,
                mode='max',
                patience=3,
                factor=0.5,
                min_lr=0.000001, threshold = 1.0, threshold_mode = 'abs'
            )

    # Initialize Model for Saving
    best_model = copy.deepcopy(model.state_dict())

    # Initialize Loss Function
    loss_function = torch.nn.MSELoss(reduction='sum')
    
    scaler = GradScaler()

    # Initialize Variables
    best_epoch = 0
    best_psnr, best_ssim = 0.0, 0.0

    # Create Directory for Saving Weights
    if "best_model" not in listdir(getcwd()) :
        mkdir(join(getcwd(), "best_model"))
    if args.project not in listdir(join(getcwd(), "best_model")) :
        mkdir(join(getcwd(), "best_model", args.project))

    # Start time
    start_time = time()
    
    # Run Training
    for epoch in range(args.epochs) :
        # Get Current Learning Rate
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            
        # Create TQDM Bar Instance
        train_bar = tqdm(train_dataloader)        
        
        # Train Current Model
        model.train()       
        
        # Create List Instance for Saving Metrics
        avg_train_time_list, total_train_time_list = list(), list()

        # Create Metric Instance
        train_loss = AverageMeter()
        train_psnr, train_ssim = AverageMeter(), AverageMeter()
        noisy_image_psnr, noisy_image_ssim = AverageMeter(), AverageMeter()
        
        # with autocast():
            # Trian Data
        for data in train_bar :
            inputs, targets = data

            # Assign Device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get Prediction
            tmapfm, preds = model(inputs)
            
            despeckle_loss = loss_function(preds, targets)
            
            criterion = despeckle_loss           
                      
            # Update Loss
            train_loss.update(criterion.item(), len(inputs))

            # Update PSNR
            train_psnr.update(calc_psnr(preds, targets).item(), len(inputs))
            noisy_image_psnr.update(calc_psnr(inputs, targets).item(), len(inputs))

            # Update SSIM
            train_ssim.update(calc_ssim(preds, targets).item(), len(inputs))
            noisy_image_ssim.update(calc_ssim(inputs, targets).item(), len(inputs))

            # Set Gradient to Zero
            optimizer.zero_grad()
            
            criterion.backward()
            
            max_norm = 1.0
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Update Model Model
            optimizer.step()
            
            # Update TQDM Bar
            train_bar.set_description(desc=f"[{epoch}/{args.epochs - 1}] [Train] [Loss : {train_loss.avg:.4f}, PSNR(Noisy) : {noisy_image_psnr.avg:.4f}, SSIM(Noisy) : {noisy_image_ssim.avg:.4f}, PSNR(Denoised) : {train_psnr.avg:.4f}, SSIM(Denoised) : {train_ssim.avg:.4f}]")


        # Create TQDM Bar Instance
        valid_bar = tqdm(valid_dataloader)

        # Validate Model
        model.eval()

        # Initialize Variables
        valid_loss = AverageMeter()
        valid_psnr, valid_ssim = AverageMeter(), AverageMeter()
        noisy_image_psnr, noisy_image_ssim = AverageMeter(), AverageMeter()

        with torch.no_grad() :
            for data in valid_bar :
                # Assign Training Data
                inputs, targets = data

                # Assign Device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Get Prediction
                tmapfm, preds = model(inputs)                

                # Get Loss
                criterion = loss_function(preds, targets)                

                # Update Train Loss
                valid_loss.update(criterion.item(), len(inputs))

                # Update PSNR
                valid_psnr.update(calc_psnr(preds, targets).item(), len(inputs))
                noisy_image_psnr.update(calc_psnr(inputs, targets).item(), len(inputs))

                # Update SSIM
                valid_ssim.update(calc_ssim(preds, targets).item(), len(inputs))
                noisy_image_ssim.update(calc_ssim(inputs, targets).item(), len(inputs))

                # Update TQDM Bar
                valid_bar.set_description(desc=f"[{epoch}/{args.epochs - 1}] [Validation] [Loss : {valid_loss.avg:.4f}, PSNR(Noisy) : {noisy_image_psnr.avg:.4f}, SSIM(Noisy) : {noisy_image_ssim.avg:.4f}, PSNR(Denoised) : {valid_psnr.avg:.4f}, SSIM(Denoised) : {valid_ssim.avg:.4f}]")
        
        time_duration = time()- start_time
        
        avg_time_per_epoch = time_duration/args.epochs
        
        hourT, minT, secT = time_conversion(time_duration)
        hourA, minA, secA = time_conversion(avg_time_per_epoch)
        print("Total training time:\t", time_duration)
        print("Average time per epoch:\t", avg_time_per_epoch)
        
        total_train_time_list.append(time_duration)
        avg_train_time_list.append(avg_time_per_epoch)
        # Create List Instance for Saving Image
        sample_list = list()

        # Append Image
        for i in range(args.batch_size) :
            sample_image = concat_image(
                                    torch.clamp(inputs[i].cpu().squeeze(0), min = 0.0, max = 1.0),
                                    torch.clamp(preds[i].cpu().squeeze(0), min =0.0, max = 1.0),
                                    torch.clamp(targets[i].cpu().squeeze(0), min = 0.0, max = 1.0)
                                    )

            sample_list.append(wandb.Image(sample_image, caption = f"Sample {i + 1}"))

        # Update Log
        wandb.log({
        "Learning Rate" : current_lr,
        "Validation PSNR" : valid_psnr.avg,
        "Validation SSIM" : valid_ssim.avg,
        "Validation Loss" : valid_loss.avg,
        "Image Comparison" : sample_list})

        # Save New Values
        if valid_psnr.avg > best_psnr :
            # Save Best Model
            best_model = copy.deepcopy(model.state_dict())
            
            # Update Variables
            best_psnr = valid_psnr.avg
            best_epoch = epoch

            # Save Best Model
            torch.save(best_model, f"best_model/{args.project}/{args.model_name}_best.pth")

        if valid_ssim.avg > best_ssim :
            # Save Best Model
            best_model = copy.deepcopy(model.state_dict())
            
            # Update Variables
            best_ssim = valid_ssim.avg
            best_epoch = epoch

            # Save Best Model
            torch.save(best_model, f"best_model/{args.project}/{args.model_name}_best.pth")

        # Update Learning Rate Scheduler
        scheduler.step(valid_psnr.avg)
        # print(scheduler.get_lr())
        
    # Create Dictionary Instance
    d = {"Total train time" : total_train_time_list,
            "Average train time per epoch" : avg_train_time_list
            }

    # Create Pandas Dataframe Instance
    df = pd.DataFrame(data = d)

    # Save as CSV Format
    df.to_csv( f"best_model/{args.project}/{args.model_name}_computation_time.csv")
    # Print Training Result
    print(f"Best Epoch : {best_epoch}")
    print(f"Best PSNR : {best_psnr:.6f}")
    print(f"Best SSIM : {best_ssim:.6f}")

if __name__ == "__main__" :
    main()
