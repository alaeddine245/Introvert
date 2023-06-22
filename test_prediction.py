import os
import torch
from torch import optim
from torch import nn
from model.introvert import Seq2Seq
from model.introvert_utils import *
import time
from data_utils.TrajectoryPredictionDataset import TrajectoryPredictionDataset
import argparse
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sqlite3
import numpy as np
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# PARAMETERS
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('testing')
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
    parser.add_argument('--model_path', type=str)
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
    args.biased_loss_mode        = 0 
    args.table_out   = "results_delta"
    args.table ="dataset_T_length_"+str(args.T_total)+"delta_coordinates"
    if args.dataset_name == 'eth' or args.dataset_name =='hotel':   # ETH dataset
        args.h = np.array([[0.0110482,0.000669589,-3.32953],[-0.0015966,0.0116324,-5.39514],[0.000111907,0.0000136174,0.542766]])
    else:                                       # UCY dataset
        args.h = np.array([[47.51,0,476],[0,41.9,117],[0,0,1]])

    args.chunk_size = args.batch_size * args.T_total
    return args

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make log folder for tensorboard
SummaryWriter_path = str(ROOT_DIR) + "/log"
writer = SummaryWriter(SummaryWriter_path,comment="ADE_FDE_Train")


def evaluate_eval(model, optimizer, criterion, criterion_vision, clip, test_loader, cnx2):
    global batch_size
    model.eval()
    i           = None
    ADEs        = 0
    FDEs        = 0
    epoch_loss  = 0
    list_x_obs          = ['x_obs_'+str(i)              for i in range(0,args.T_obs)]
    list_y_obs          = ['y_obs_'+str(i)              for i in range(0,args.T_obs)]
    list_c_context      = ['context_c_'+str(i)          for i in range(0,args.hidden_size)]
    list_h_context      = ['context_h_'+str(i)          for i in range(0,args.hidden_size)]
    list_x_pred         = ['x_pred_'+str(i)             for i in range(0,args.T_pred)]
    list_y_pred         = ['y_pred_'+str(i)             for i in range(0,args.T_pred)]
    list_x_stoch_pred_m = ['x_stoch_pred_m_'+str(i)     for i in range(0,args.T_pred)]
    list_y_stoch_pred_m = ['y_stoch_pred_m_'+str(i)     for i in range(0,args.T_pred)]
    list_x_stoch_pred_s = ['x_stoch_pred_s_'+str(i)     for i in range(0,args.T_pred)]
    list_y_stoch_pred_s = ['y_stoch_pred_s_'+str(i)     for i in range(0,args.T_pred)]
    list_x_out          = ['x_out_'+str(i)              for i in range(0,args.T_pred)]
    list_y_out          = ['y_out_'+str(i)              for i in range(0,args.T_pred)]
    list_vsn           = ['vsn_'+str(i)               for i in range(0,args.hidden_size)]
    # list_vsn_visual    = ['vsn_vis_'+str(i)           for i in range(0,T_obs*T_pred)]
    df_out              = pd.DataFrame(columns=list_x_obs + list_y_obs + list_x_out + list_y_out + list_x_pred + list_y_pred + list_x_stoch_pred_m + list_y_stoch_pred_m + list_x_stoch_pred_s + list_y_stoch_pred_s + list_c_context + list_h_context + list_vsn)# + list_vsn_visual)

    for i,data in enumerate(test_loader):
        start_time = time.time()
        # Forward
        obs, pred, visual_obs, frame_tensor                 = data
        input_tensor, output_tensor                         = obs.float().squeeze().to('cuda', non_blocking=True), pred.float().squeeze().to('cuda', non_blocking=True)               #(obs.to(device), pred.to(device))
        visual_input_tensor                                 = visual_obs.squeeze().cuda()   #(visual_obs.to(device), visual_pred.to(device))
        prediction, stochastic_prediction, encoder_hidden, decoder_hidden, visual_embedding, attn_rep, attn2, attn4, attn6 = model(input_tensor, visual_input_tensor, output_tensor, args.batch_size, train_mode=0)
    

        calculated_prediction =  prediction.cumsum(axis=1) 

        loss_line_regularizer                               = distance_from_line_regularizer(input_tensor,calculated_prediction)

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
        ADE             = ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0).mean(0)   
        FDE             = ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0)[-1]
        total_loss      = loss.double() + args.regularization_factor * loss_line_regularizer.double() 
        print("Total Loss\t{:.2f}".format(total_loss.item()))
        epoch_loss += total_loss.item()
        ADEs    += ADE.item()
        FDEs    += FDE.item()
        input_x_lin                 = input_tensor[:,:,0].view(-1, args.T_obs).cpu()
        input_y_lin                 = input_tensor[:,:,1].view(-1, args.T_obs).cpu()
        output_x_lin                = output_tensor[:,:,0].view(-1, args.T_pred).cpu()
        output_y_lin                = output_tensor[:,:,1].view(-1, args.T_pred).cpu()
        prediction_x_lin            = prediction[:,:,0].view(-1, args.T_pred).cpu()
        prediction_y_lin            = prediction[:,:,1].view(-1, args.T_pred).cpu()
        stoch_prediction_x_m        = stochastic_prediction[:,:,0].view(-1, args.T_pred).cpu()
        stoch_prediction_x_s        = stochastic_prediction[:,:,1].view(-1, args.T_pred).cpu()
        stoch_prediction_y_m        = stochastic_prediction[:,:,2].view(-1, args.T_pred).cpu()
        stoch_prediction_y_s        = stochastic_prediction[:,:,3].view(-1, args.T_pred).cpu()
        context_h_lin               = encoder_hidden[0].view(-1, args.hidden_size).cpu()
        context_c_lin               = encoder_hidden[1].view(-1, args.hidden_size).cpu()
        visual_embedding_weights    = visual_embedding.view(-1, args.hidden_size).cpu()

        whole_data                  = torch.cat((input_x_lin, input_y_lin, output_x_lin, output_y_lin, prediction_x_lin, prediction_y_lin, stoch_prediction_x_m, stoch_prediction_y_m, stoch_prediction_x_s, stoch_prediction_y_s, context_c_lin, context_h_lin, visual_embedding_weights), 1)
        temp                        = pd.DataFrame(whole_data.detach().cpu().numpy(), columns=list_x_obs + list_y_obs + list_x_out + list_y_out + list_x_pred + list_y_pred + list_x_stoch_pred_m + list_y_stoch_pred_m + list_x_stoch_pred_s + list_y_stoch_pred_s + list_c_context + list_h_context + list_vsn)
        df_out                      = df_out.append(temp)
        df_out.reset_index(drop=True,inplace=True)

        print("Time\t\t{:.2f} sec \n".format(time.time() - start_time))


    # ADE/FDE Report
    out_x  = df_out[['x_out_' +str(i) for i in range(0,args.T_pred)]].cumsum(axis=1)
    pred_x = df_out[['x_pred_'+str(i) for i in range(0,args.T_pred)]].cumsum(axis=1)
    out_y  = df_out[['y_out_' +str(i) for i in range(0,args.T_pred)]].cumsum(axis=1)
    pred_y = df_out[['y_pred_'+str(i) for i in range(0,args.T_pred)]].cumsum(axis=1)
    ADE = (out_x.sub(pred_x.values)**2).add((out_y.sub(pred_y.values)**2).values, axis=1)**(1/2)
    df_out['ADE'] = ADE.mean(axis=1)
    FDE = ADE.x_out_11
    df_out['FDE'] = FDE
    Mean_ADE = df_out.ADE.mean()
    Mean_FDE = df_out.FDE.mean()
    print("MEAN ADE/FDE\t",Mean_ADE,Mean_FDE)
    writer.add_scalar("Final_Test/ADE_"+args.path_mode, Mean_ADE, global_step=0)
    writer.add_scalar("Final_Test/FDE_"+args.path_mode, Mean_FDE, global_step=0)

    df_out.to_sql(args.table_out+'_'+args.path_mode, cnx2, if_exists="replace", index=False)
    writer.close()
    return ADEs, FDEs, int(args.data_size/args.chunk_size)

def main(args):
    DB_DIR      = str(ROOT_DIR) + '/database'
    os.makedirs( DB_DIR )
    DB_PATH2    = DB_DIR+'/db_one_ped_delta_coordinates_results.db'
    cnx2        = sqlite3.connect(DB_PATH2)
    image_folder_path       = 'data/'+args.dataset_name
    print("LOAD MODEL")
    # Change device to cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    learning_step               = 40
    initial_learning_rate       = 0.01
    clip                        = 1
    criterion                   = nn.MSELoss(reduction='mean')#nn.NLLLoss()
    criterion_vision            = nn.MSELoss(reduction='sum')#nn.NLLLoss()


    model               = Seq2Seq(args)
    model               = nn.DataParallel( model ).cuda()
    checkpoint          = torch.load(args.model_path,map_location=device)
    model.load_state_dict(checkpoint)

    optimizer                   = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.01)

    # ----------------------------------------------------------
    #TEST DATASET AND LOADER
    DB_PATH_val     = "data/"+args.dataset_name+"/pos_data_val.db"
    cnx_val         = sqlite3.connect(DB_PATH_val)
    df_id       = pd.read_sql_query("SELECT data_id FROM "+args.table, cnx_val)
    args.data_size   = df_id.data_id.max() * args.T_total

    dataset_val   = TrajectoryPredictionDataset(image_folder_path, DB_PATH_val, cnx_val, args)
    test_loader   = torch.utils.data.DataLoader(dataset_val,         batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    print("EVALUATE")
    model.eval()
    path_mode = 'bst'
    print("path mode\t",path_mode)
    evaluate_eval(model, optimizer, criterion, criterion_vision, clip, test_loader, cnx2)

    print("EVALUATE")
    model.eval()
    path_mode = 'top5'
    print("path mode\t",path_mode)
    evaluate_eval(model, optimizer, criterion, criterion_vision, clip, test_loader, cnx2)

if __name__ == '__main__':
    args = parse_args()
    main(args)