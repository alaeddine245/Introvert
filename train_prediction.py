import os
import torch
from torch import optim
from torch import nn
import time
import argparse
import numpy as np
import sqlite3
import sys
import pandas as pd
import datetime
from torch.utils.tensorboard import SummaryWriter
from model.introvert_utils import *
from model.introvert import Seq2Seq
from data_utils.TrajectoryPredictionDataset import TrajectoryPredictionDataset
from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# PARAMETERS
def parse_args():
    '''PARAMETERS'''
    # Create an ArgumentParser object with the description 'training'
    # Leave out stochastic_mode, avg_n_path_eval, bst_n_path_eval, path_mode, image_size,
    # image_dimension, mask_size, visual_features_size, hidden_size, in_size, 
    # vsn_module_out_size, embed_size
    parser = argparse.ArgumentParser('training')
    parser = argparse.ArgumentParser(description='Input parameters for the script')
    parser.add_argument('--dataset_name', type=str, default='university')
    parser.add_argument('--T_obs', type=int, default=8)
    parser.add_argument('--T_pred', type=int, default=12)
    parser.add_argument('--dropout_val', type=float, default=0.2)
    parser.add_argument('--regularization_factor', type=float, default=0.5)
    parser.add_argument('--regularization_mode', type=str, default="regular")
    parser.add_argument('--startpoint_mode', type=str, default="on")
    parser.add_argument('--enc_out', type=str, default="on")
    parser.add_argument('--biased_loss_mode', type=int, default=0)
    parser.add_argument('--table_out', type=str, default="results_delta")
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--from_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    args = parser.parse_args()
    args.T_total = args.T_obs + args.T_pred
    args.avg_n_path_eval = 20
    args.bst_n_path_eval = 20
    args.path_mode = 'top5'
    args.image_size = 256
    args.image_dimension = 3
    args.mask_size = 16
    args.visual_features_size = 128
    args.visual_embed_size = 64
    args.hidden_size = 64
    args.in_size = 2
    args.vsn_module_out_size = 256
    args.embed_size = 64
    args.teacher_forcing_ratio = 0.7
    args.stochastic_mode = 1
    args.table ="dataset_T_length_"+str(args.T_total)+"delta_coordinates"
    if args.dataset_name == 'eth' or args.dataset_name =='hotel':   # ETH dataset
        args.h = np.array([[0.0110482,0.000669589,-3.32953],[-0.0015966,0.0116324,-5.39514],[0.000111907,0.0000136174,0.542766]])
    else:                                       # UCY dataset
        args.h = np.array([[47.51,0,476],[0,41.9,117],[0,0,1]])

    args.chunk_size = args.batch_size * args.T_total
    return args

def train(model, optimizer, scheduler, criterion, criterion_vision, clip,train_loader, validation_loader, writer, args):
    # Create directories for logs and checkpoints
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('introvert')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    
    i               = None
    checked_frame   = 0

    print("Data Size ",args.data_size,"\tChunk Size ",args.chunk_size)
    counter =0
    teacher_forcing_ratio = args.teacher_forcing_ratio
    for j in range(args.epoch_num):
        model.train()
        epoch_loss=0
        if j%7 == 6:
            teacher_forcing_ratio = (teacher_forcing_ratio - 0.2) if teacher_forcing_ratio>=0.1 else 0.0

        # Update TeachForce ratio to gradually change during training
        # global teacher_forcing_ratio
        # teacher_forcing_ratio-= 1/epoch_num
        print("TEACHER FORCE RATIO\t",teacher_forcing_ratio)
        #print("Learning Rate\t", scheduler.get_last_lr())

        if(j>=args.from_epoch):
            optimizer.zero_grad()
            start_time = time.time()
            ADE = 0
            FDE = 0
            i   = 0
            for i,data in enumerate( train_loader):
                print("\n--------------- Batch %d/ %d ---------------"%(j,i))  
                # Forward
                obs, pred, visual_obs, frame_tensor              = data
                input_tensor, output_tensor                      = obs.float().squeeze().to('cuda', non_blocking=True), pred.float().squeeze().to('cuda', non_blocking=True)               #(obs.to(device), pred.to(device))
                visual_input_tensor                              = visual_obs.squeeze().to('cuda', non_blocking=True)   #(visual_obs.to(device), visual_pred.to(device))
                prediction, stochastic_prediction, encoder_hidden, decoder_hidden, visual_embedding, attn_rep,_,_,_ = model(input_tensor,  visual_input_tensor, output_tensor, args.batch_size,train_mode=1)
                

                calculated_prediction = prediction.cumsum(axis=1) 

                loss_line_regularizer                            = distance_from_line_regularizer(input_tensor,calculated_prediction)
                
                if args.biased_loss_mode:
                    weight  = torch.arange(1,2*args.T_pred+1,2).cuda().float()
                    weight  = torch.exp(weight / args.T_pred).repeat(prediction.size(0)).view(prediction.size(0),args.T_pred,1)
                    loss    = criterion( (calculated_prediction)*weight, torch.cumsum(output_tensor,dim=-2)*weight)
                else:
                    loss    = criterion( (calculated_prediction), torch.cumsum(output_tensor,dim=-2))       
                out_x           = output_tensor[:,:,0].cumsum(axis=1)
                out_y           = output_tensor[:,:,1].cumsum(axis=1)
                pred_x          = calculated_prediction[:,:,0]
                pred_y          = calculated_prediction[:,:,1]
                ADE             += ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0).mean(0)   
                # FDE             += ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0)[-1]
                # Backward Propagation

                total_loss      = loss.double() + torch.tensor(args.regularization_factor).to('cuda', non_blocking=True) * loss_line_regularizer.double()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                print("Total Loss\t{:.2f}".format(total_loss.item()))
                epoch_loss += total_loss.item()
                print("Time\t\t{:.2f} sec \n".format(time.time() - start_time))
                start_time = time.time()
                torch.cuda.empty_cache()
                writer.close()
                count_div=i
            
            # tensorboard log
            writer.add_scalar('ADE/train', ADE.item()/(count_div+1),     counter )
            # writer.add_scalar('FDE/train', FDE.item()/(count_div+1),     counter )
            # writer.add_scalar('LOSS/train', epoch_loss/(count_div+1)   , counter)
            counter += 1

        if scheduler.get_last_lr()[0]>0.001:
            scheduler.step()
        # validation(model, optimizer, criterion, criterion_vision, clip, validation_loader, j) 
        print("EPOCH ", j, "\tLOSS ",epoch_loss / (int(args.data_size/args.chunk_size)))
        writer.add_scalar('epoch_loss/train', epoch_loss/ (int(args.data_size/args.chunk_size)), j )
        torch.save( model.state_dict(), str(checkpoints_dir) + "/best_model.pth")
        print("-----------------------------------------------\n"+"-----------------------------------------------")
    return epoch_loss / (int(args.data_size/args.chunk_size))


def main(args):
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    image_folder_path       = 'data/'+args.dataset_name
    DB_PATH_train     = "data/"+args.dataset_name+"/pos_data_train.db"
    cnx_train         = sqlite3.connect(DB_PATH_train)
    df_id       = pd.read_sql_query("SELECT data_id FROM "+args.table, cnx_train)
    args.data_size   = df_id.data_id.max() * args.T_total
    # Make log folder for tensorboard
    SummaryWriter_path = str(ROOT_DIR) + "/log"
    writer = SummaryWriter(SummaryWriter_path,comment="ADE_FDE_Train")

    



    model                       = Seq2Seq(args)
    model                       = nn.DataParallel( model ).cuda()

    learning_step               = 40
    initial_learning_rate       = 0.01
    clip                        = 1
    # MSE loss
    criterion                   = nn.MSELoss(reduction='mean')#nn.NLLLoss()
    criterion_vision            = nn.MSELoss(reduction='sum')#nn.NLLLoss()
    # SGD optimizer
    optimizer                   = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.01)
    scheduler                   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_step, gamma=0.1)
    five_fold_cross_validation  = 0

    dataset_train = TrajectoryPredictionDataset(image_folder_path, DB_PATH_train, cnx_train, args)

    train_loader        = torch.utils.data.DataLoader(dataset_train,        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
    validation_loader   = None



    print("TRAIN")
    model.train()
    print("path mode\t",args.path_mode)
    loss = train(model, optimizer, scheduler, criterion, criterion_vision, clip, train_loader, validation_loader, writer, args)
    print("LOSS ",loss)
if __name__ == '__main__':
    args = parse_args()
    main(args)