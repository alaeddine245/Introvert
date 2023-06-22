
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import random
import warnings
warnings.filterwarnings('ignore')
from .introvert_utils import *
import sys


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args
        #args: in_size, embed_size, hidden_size, dropout_val, batch_size
        torch.cuda.empty_cache()
        self.encoder        = EncoderRNN(args)
        #in_size, embed_size, hidden_size, dropout_val, batch_size=batch_size
        self.encoder.apply(init_weights)
        
        self.decoder        = DecoderRNN(args)
        #in_size, embed_size, hidden_size, dropout_val, batch_size=batch_size
        self.decoder.apply(init_weights)
        
        self.vsn_module    = Vision(args)
        #dropout_val, batch_size=batch_size
        self.vsn_module.apply(init_weights)
        
        if device.type=='cuda':
            self.encoder.cuda()
            self.decoder.cuda()
            self.vsn_module.cuda()

    def forward(self,input_tensor, visual_input_tensor, output_tensor, batch_size, train_mode):
        batch_size      = int(input_tensor.size(0))#/torch.cuda.device_count())
        encoder_hidden  = (self.encoder.initHidden(batch_size),self.encoder.initHidden(batch_size))
        encoder_outputs = torch.zeros(batch_size, self.args.T_obs, self.args.hidden_size).cuda()#.cpu()
        start_point     = (input_tensor[:,0,:]).to(device).clone().detach()
        
        if self.args.startpoint_mode=="on":
            input_tensor[:,0,:]    = 0
        
        for t in range(0,self.args.T_obs):
            encoder_output, encoder_hidden  = self.encoder(input_tensor[:,t,:], encoder_hidden)
            encoder_outputs[:,t,:]          = encoder_output.squeeze(1)
        
        # Visual extraction/attention       
        # Enc outputs
        if self.args.enc_out=="on" and self.args.startpoint_mode=="on":
            encoder_extract             = self.encoder.emb_out(encoder_outputs.view(batch_size,-1))
            condition                   = torch.cat([encoder_extract.view(batch_size,-1),start_point.view(batch_size,-1)],dim=-1)
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,condition)
            decoder_hidden              = [torch.cat([encoder_extract.view(batch_size,-1),   visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        elif self.args.enc_out=="on" and self.args.startpoint_mode=="off":  
            encoder_extract             = self.encoder.emb_out(encoder_outputs.view(batch_size,-1))
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,encoder_extract)
            decoder_hidden              = [torch.cat([encoder_extract.view(batch_size,-1),   visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        elif self.args.enc_out=="off" and self.args.startpoint_mode=="on":
            condition = torch.cat([encoder_hidden[0].view(batch_size,-1),start_point.view(batch_size,-1)],dim=-1)
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,condition)
            decoder_hidden              = [torch.cat([encoder_hidden[0].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        else:
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,encoder_hidden[0])
            decoder_hidden              = [torch.cat([encoder_hidden[0].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        
        visual_vsn_result   = visual_initial_vsn

        # Initial Decoder input is current state coordinates
        # Initial Decoder hidden state is last Encoder hidden state

        decoder_input                   = input_tensor[:,-1,:]


        a0 = encoder_hidden[0].view(batch_size,-1)
        a1 = visual_vsn_result.view(batch_size,-1)
        a2 = torch.einsum("bn,bm->bnm",a0,a1)
        #tens_a = torch.cat([a0,a1,a2],dim=-1)
        tens_a = torch.ones(batch_size, a0.size(1)+1, a1.size(1)+1, device="cuda")
        tens_a[:,1:,1:] = a2
        tens_a[:,0,1:]  = a1
        tens_a[:,1:,0]  = a0


        b0 = encoder_hidden[1].view(batch_size,-1)
        b1 = visual_vsn_result.view(batch_size,-1)
        b2 = torch.einsum("bn,bm->bnm",b0,b1)
        # tens_b = torch.cat([b0,b1,b2],dim=-1)
        tens_b = torch.ones(batch_size, b0.size(1)+1, b1.size(1)+1, device="cuda")
        tens_b[:,1:,1:] = b2
        tens_b[:,0,1:]  = b1
        tens_b[:,1:,0]  = b0

        tens_a_red = self.decoder.dim_red(tens_a)
        tens_b_red = self.decoder.dim_red(tens_b)
        
        decoder_hidden                  = [tens_a_red.unsqueeze(0),\
                                           tens_b_red.unsqueeze(0)]

        
        # Tensor to store decoder outputs
        outputs                         = torch.zeros(batch_size, self.args.T_pred , self.args.in_size).cuda()#.cpu() 
        stochastic_outputs              = torch.zeros(batch_size, self.args.T_pred , self.args.in_size*2).cuda()#.cpu()
        teacher_force                   = 1
        print('cuda:'+str(torch.cuda.current_device()))
        epsilonX                        = Normal(torch.zeros(batch_size,1),torch.ones(batch_size,1))
        epsilonY                        = Normal(torch.zeros(batch_size,1),torch.ones(batch_size,1))
        teacher_force                   = int(random.random() < self.args.teacher_forcing_ratio) if train_mode else 0
        print("Teacher Force\t",teacher_force)
        print("path mode\t",self.args.path_mode)
        for t in range(0, self.args.T_pred):
            stochastic_decoder_output, decoder_hidden   = self.decoder(decoder_input, decoder_hidden)
            # Reparameterization Trick :)
            decoder_output              = torch.zeros(batch_size,1,2).cuda()

            if self.args.stochastic_mode and self.args.path_mode=='single':
                decoder_output[:,:,0]  = stochastic_decoder_output[:,:,0] + epsilonX.sample().cuda() * stochastic_decoder_output[:,:,1]
                decoder_output[:,:,1]  = stochastic_decoder_output[:,:,2] + epsilonY.sample().cuda() * stochastic_decoder_output[:,:,3]
            elif self.args.stochastic_mode and self.args.path_mode=='avg':
                decoder_output[:,:,0]  = stochastic_decoder_output[:,:,0] + epsilonX.sample((self.args.avg_n_path_eval,1)).view(-1,self.args.avg_n_path_eval,1).mean(-2).cuda() * stochastic_decoder_output[:,:,1]
                decoder_output[:,:,1]  = stochastic_decoder_output[:,:,2] + epsilonY.sample((self.args.avg_n_path_eval,1)).view(-1,self.args.avg_n_path_eval,1).mean(-2).cuda() * stochastic_decoder_output[:,:,3]
            elif not(self.args.stochastic_mode):
                decoder_output[:,:,0]  = stochastic_decoder_output[:,:,0] 
                decoder_output[:,:,1]  = stochastic_decoder_output[:,:,2] 
            elif self.args.stochastic_mode and self.args.path_mode == "bst":
                epsilon_x               = torch.randn([batch_size,self.args.bst_n_path_eval,1], dtype=torch.float).cuda()
                epsilon_y               = torch.randn([batch_size,self.args.bst_n_path_eval,1], dtype=torch.float).cuda()
                multi_path_x            = stochastic_decoder_output[:,:,0].unsqueeze(1) + epsilon_x * stochastic_decoder_output[:,:,1].unsqueeze(1)
                multi_path_y            = stochastic_decoder_output[:,:,2].unsqueeze(1) + epsilon_y * stochastic_decoder_output[:,:,3].unsqueeze(1)
                ground_truth_x          = output_tensor[:,t,0].view(batch_size,1,1).cuda()
                ground_truth_y          = output_tensor[:,t,1].view(batch_size,1,1).cuda()
                diff_path_x             = multi_path_x - ground_truth_x
                diff_path_y             = multi_path_y - ground_truth_y
                diff_path               = (torch.sqrt( diff_path_x.pow(2) + diff_path_y.pow(2) )).sum(dim=-1)
                idx                     = torch.arange(batch_size,dtype=torch.long).cuda()
                min                     = torch.argmin(diff_path,dim=1).squeeze()
                decoder_output[:,:,0]   = multi_path_x[idx,min,:].view(batch_size,1)
                decoder_output[:,:,1]   = multi_path_y[idx,min,:].view(batch_size,1)
            elif self.args.stochastic_mode and self.args.path_mode == "top5":
                k = 5 #topk
                epsilon_x               = torch.randn([batch_size,self.args.bst_n_path_eval,1], dtype=torch.float).cuda()
                epsilon_y               = torch.randn([batch_size,self.args.bst_n_path_eval,1], dtype=torch.float).cuda()
                multi_path_x            = stochastic_decoder_output[:,:,0].unsqueeze(1) + epsilon_x * stochastic_decoder_output[:,:,1].unsqueeze(1)
                multi_path_y            = stochastic_decoder_output[:,:,2].unsqueeze(1) + epsilon_y * stochastic_decoder_output[:,:,3].unsqueeze(1)
                ground_truth_x          = output_tensor[:,t,0].view(batch_size,1,1).cuda()
                ground_truth_y          = output_tensor[:,t,1].view(batch_size,1,1).cuda()
                diff_path_x             = multi_path_x - ground_truth_x
                diff_path_y             = multi_path_y - ground_truth_y
                diff_path               = (torch.sqrt( diff_path_x.pow(2) + diff_path_y.pow(2) )).sum(dim=-1)
                idx                     = torch.arange(batch_size,dtype=torch.long).repeat(k).view(k,-1).transpose(0,1).cuda()
                min_val, min            = torch.topk(diff_path, k=k, dim=1,largest=False)
                decoder_output[:,:,0]   = multi_path_x[idx,min,:].mean(dim=-2).view(batch_size,1)
                decoder_output[:,:,1]   = multi_path_y[idx,min,:].mean(dim=-2).view(batch_size,1)

            # Log output
            outputs[:,t,:]                        = decoder_output.squeeze(1)
            stochastic_outputs[:,t,:]             = stochastic_decoder_output.squeeze(1)
            decoder_input                         = output_tensor[:,t,:] if teacher_force else decoder_output

        return outputs, stochastic_outputs, encoder_hidden, decoder_hidden, visual_vsn_result, attn_rep,attn2,attn4,attn6
            