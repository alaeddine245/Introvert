import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import math
# Load arguments from json file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)


"""## Regularizer loss"""

sum_sigma_distance  = torch.zeros(1)

def distance_from_line_regularizer(input_tensor,prediction, regularization_mode='weighted', T_pred=8):
    global sum_sigma_distance
    # Fit a line to observation points over batch 
    input_tensor    = input_tensor.double()
    prediction      = prediction.double()
    input_tensor    = input_tensor.cumsum(dim=1).double()
    X               = torch.ones_like(input_tensor).double().to('cuda', non_blocking=True)
    X[:,:,0]        = input_tensor[:,:,0]
    Y               = (input_tensor[:,:,1]).unsqueeze(-1).double()
    try:
        try:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().inverse()
        except:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().pinverse()
        XTY             = torch.matmul( X.transpose(-1,-2), Y)
        theta           = torch.matmul( XTX_1.double(), XTY.double())
        # Calculate real values of prediction instead of delta
        prediction[:,:,0] = prediction[:,:,0] + input_tensor[:,-1,0].unsqueeze(-1) 
        prediction[:,:,1] = prediction[:,:,1] + input_tensor[:,-1,1].unsqueeze(-1)
        
        # Calculate distance ( predicted_points , observation_fitted_line ) over batch
        theta0x0        = theta[:,0,:].double() * prediction[:,:,0].double()
        denominator     = torch.sqrt( theta[:,0,:].double() * theta[:,0,:].double() + 1 )
        nominator       = theta0x0 + theta[:,1,:] - prediction[:,:,1].double()
        distance        = nominator.abs() / denominator
        if regularization_mode =='weighted':
            weight              = torch.flip( torch.arange(1,T_pred+1).cuda().float(),[0])
            weight              = (weight / T_pred).repeat(distance.size(0)).view(-1,T_pred)
            weighted_distance   = weight * distance

        elif regularization_mode =='e_weighted':
            weight              = torch.flip( torch.arange(1,T_pred+1).cuda().float(),[0])
            weight              = (weight / T_pred).repeat(distance.size(0)).view(distance.size(0),T_pred)
            weight              = torch.exp(weight)
            weighted_distance   = weight*distance

        else:
            weighted_distance = distance
        sigma_distance  = torch.mean(weighted_distance,1)
        sum_sigma_distance  = torch.mean(sigma_distance)
        return sum_sigma_distance
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1).to('cuda', non_blocking=True) + 20
        return sum_sigma_distance



def angle_from_line_regularizer(input_tensor,prediction):
    global sum_sigma_distance
    input_tensor    = input_tensor.double()
    prediction      = prediction.double()

    # Calculate real values of observation instead of delta
    input_tensor    = input_tensor.cumsum(dim=1).double()

    # Calculate real values of prediction instead of delta
    prediction[:,:,0] = prediction[:,:,0] + input_tensor[:,-1,0].unsqueeze(-1) 
    prediction[:,:,1] = prediction[:,:,1] + input_tensor[:,-1,1].unsqueeze(-1)

    # Fit a line to observation points over batch 
    X               = torch.ones_like(input_tensor).double()
    X[:,:,0]        = input_tensor[:,:,0]
    Y               = (input_tensor[:,:,1]).unsqueeze(-1).double()
    try:
        try:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().inverse()
        except:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().pinverse()
        XTY             = torch.matmul( X.transpose(-1,-2), Y)
        theta           = torch.matmul( XTX_1.double(), XTY.double())
        theta_observation   = theta.double()
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1) + math.pi/4
        return sum_sigma_distance  

    # Fit a line to prediction points over batch 
    X               = torch.ones_like(prediction).double()
    X[:,:,0]        = prediction[:,:,0]
    Y               = (prediction[:,:,1]).unsqueeze(-1).double()
    try:
        XTX_1           = torch.matmul( X.transpose(-1,-2), X).inverse()
        XTY             = torch.matmul( X.transpose(-1,-2), Y)
        theta           = torch.matmul( XTX_1.double(), XTY.double())
        theta_prediction    = theta.double()
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1) + math.pi/4
        return sum_sigma_distance 

    try:
        # Find two vectors(directed lines)
        x_first             = input_tensor[:,0,0].unsqueeze(-1)
        x_last              = input_tensor[:,-1,0].unsqueeze(-1)
        y_first             = theta_observation[:,0,:].double() * x_first.double()  +  theta_observation[:,1,:].double()
        y_last              = theta_observation[:,0,:].double() * x_last.double()  +  theta_observation[:,1,:].double()
        vector_observation  = [x_last-x_first , y_last-y_first]

        x_first             = prediction[:,0,0].unsqueeze(-1)
        x_last              = prediction[:,-1,0].unsqueeze(-1)
        y_first             = theta_prediction[:,0,:].double() * x_first.double()  +  theta_prediction[:,1,:].double()
        y_last              = theta_prediction[:,0,:].double() * x_last.double()  +  theta_prediction[:,1,:].double()
        vector_prediction   = [x_last-x_first , y_last-y_first]

        # Find the angle between two vectors
        nominator                   = vector_observation[0]*vector_prediction[0] + vector_observation[1]*vector_prediction[1]
        denominator0                = torch.sqrt(vector_observation[0]*vector_observation[0] + vector_observation[1]*vector_observation[1])
        denominator1                = torch.sqrt(vector_prediction[0]*vector_prediction[0] + vector_prediction[1]*vector_prediction[1])
        denominator                 = denominator0 * denominator1
        cosine                      = nominator / (denominator+0.01)
        cosine[torch.isnan(cosine)] = -0.01
        angle                       = torch.acos(cosine)#*180/math.pi 
        
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1) + math.pi/4
        return sum_sigma_distance

    return torch.mean(angle)

"""## Masking by segmentation"""

def mask_pedestrian_segmentation(segmentation, image_size = 256, image_dimension=3, mask_size=16):

    # tensor to img
    seg_frame       = (segmentation.permute(0,1,3,4,2).view(-1,image_size,image_size,image_dimension))#.numpy()) #PERMUTE to correct RGB space
    #seg_frame       = img_as_ubyte(seg_frame)

    # pool img
    avg_pool        = nn.AvgPool2d((int(image_size/mask_size), int(image_size/mask_size)), stride=(int(image_size/mask_size), int(image_size/mask_size)))
    #maskp           = torch.tensor(seg_frame[:,:,:,0].to(device), dtype=torch.double).view(-1,image_size,image_size)
    if device.type=='cuda':
            avg_pool.cuda()
    pooled_mask     = avg_pool(seg_frame[:,:,:,0].to(device)).view(-1, segmentation.size(1), mask_size, mask_size)

    return pooled_mask


class EncoderRNN(nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        self.args = args
        # Configurations
        self.in_size                = self.args.in_size
        self.hidden_size            = self.args.hidden_size
        self.batch_size             = self.args.batch_size
        self.embed_size             = self.args.embed_size
        self.seq_length             = self.args.T_obs
        self.dropout_val            = self.args.dropout_val
        self.num_RRN_layers         = 1
        #self.vgg16_features_size    = vgg16_features_size
        self.visual_embed_size      = self.args.visual_embed_size

        #Architecture
        self.embedder_phi               = nn.Linear(self.in_size, self.embed_size)

        self.encoder                    = nn.LSTM(self.embed_size , self.hidden_size, self.num_RRN_layers, batch_first=True) # ezafe kon visual embedding_size ro
        self.dropout                    = nn.Dropout(self.args.dropout_val)
        self.embedder_out               = nn.Sequential(
                                                        nn.Linear(self.args.T_obs*self.args.hidden_size, self.args.hidden_size),
                                                        nn.ReLU(),
                                                        nn.Dropout(p=self.args.dropout_val),
                                                        nn.Linear(self.args.hidden_size, self.args.hidden_size),
                                                        nn.ReLU()
                                                        )
    
    def forward(self, input, hidden): 
        # Coordination Embedding
        embedding                   = self.embedder_phi(input.view(-1,2))
        embedding                   = F.relu(self.dropout(embedding))
        # RNN
        output, hidden              = self.encoder(embedding.unsqueeze(1), ( hidden[0],hidden[1] ) )
        return output, hidden


    def initHidden(self,batch_size):
        self.batch_size=batch_size
        return torch.zeros([self.num_RRN_layers, self.batch_size, self.hidden_size]).cuda()#, device = device)

    def emb_out(self,input):
        out= self.embedder_out(input)
        return out    
class DecoderRNN(nn.Module):
    def __init__(self, args):
        super(DecoderRNN, self).__init__()
        self.args = args
        # Configurations
        self.in_size                = self.args.in_size
        self.stochastic_out_size    = self.args.in_size * 2
        self.hidden_size            = self.args.hidden_size
        self.batch_size             = self.args.batch_size
        self.embed_size             = self.args.embed_size
        self.seq_length             = self.args.T_pred
        self.dropout_val            = self.args.dropout_val
        self.num_RRN_layers         = 1
        #self.vgg16_features_size    = vgg16_features_size
        self.visual_embed_size      = self.args.visual_embed_size
        #self.vgg16_features_size    = vgg16_features_size
        self.visual_embed_size      = self.args.visual_embed_size
        self.visual_size            = self.args.image_dimension * self.args.image_size * self.args.image_size

        #Architecture
        self.embedder_rho               = nn.Linear(self.in_size, self.embed_size)
        if self.args.startpoint_mode=="on":
            self.decoder                    = nn.LSTM(self.embed_size , self.hidden_size + self.hidden_size + self.args.in_size, self.num_RRN_layers, batch_first=True)
            self.fC_mu                      = nn.Sequential(
                                                            nn.Linear(self.hidden_size + self.hidden_size + self.args.in_size, int(self.hidden_size/2), bias=True),
                                                            nn.ReLU(),
                                                            nn.Dropout(p=self.args.dropout_val),
                                                            nn.Linear(int(self.hidden_size/2), self.stochastic_out_size, bias=True)
                                                            )
        else:
            self.decoder                    = nn.LSTM(self.embed_size , self.hidden_size + self.hidden_size , self.num_RRN_layers, batch_first=True)
            self.fC_mu                      = nn.Sequential(
                                                            nn.Linear(self.hidden_size + self.hidden_size , int(self.hidden_size/2), bias=True),
                                                            nn.ReLU(),
                                                            nn.Dropout(p=self.args.dropout_val),
                                                            nn.Linear(int(self.hidden_size/2), self.stochastic_out_size, bias=True)
                                                            )
        self.dropout                        = nn.Dropout(self.args.dropout_val)

        self.reducted_size = int((self.hidden_size-1)/3)+1
        if self.args.startpoint_mode =="on":
            self.reducted_size2 = int((self.hidden_size+self.args.in_size-1)/3)+1
            self.FC_dim_red                     = nn.Sequential(
                                                            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
                                                            nn.Flatten(start_dim=1, end_dim=-1),
                                                            nn.Linear(self.reducted_size*self.reducted_size2, 2*self.hidden_size+self.args.in_size, bias=True),
                                                            nn.ReLU()
                                                            )
        else:
            self.FC_dim_red                     = nn.Sequential(
                                                            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
                                                            nn.Flatten(start_dim=1, end_dim=-1),
                                                            nn.Linear(self.reducted_size*self.reducted_size, 2*self.hidden_size, bias=True),
                                                            nn.ReLU()
                                                            )

    def forward(self, input, hidden): 
        # Coordination Embedding
        embedding                       = self.embedder_rho(input.view(-1,2))
        embedding                       = F.relu(self.dropout(embedding))
        output, hidden                  = self.decoder(embedding.unsqueeze(1), ( hidden[0],hidden[1] ))
        prediction                      = self.fC_mu(output.squeeze(0)) 
        return prediction, hidden #prediction_v.view(-1, self.visual_embed_size).cpu(), visual_embedding_ground_truth.cpu(), hidden


    def dim_red(self, input):
        output = self.FC_dim_red(input)
        return output
  

class Vision(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Vision, self).__init__()
        k0 = s0 = 3
        p0 = 1
        self.CNN_0  = nn.Sequential(
                        nn.Conv3d(3, 16, kernel_size=k0, stride=[s0,s0,s0], padding=p0),
                        nn.ReLU(),
                        )
        # -----------
        m0 = 16
        n0 = 8#16
        k_a0 = s_a0 = 1
        h0 = w0 = int( (args.image_size - k0 + 2*p0)/s0 ) +1
        t0 = int( (args.T_obs - k0 + 2*p0)/s0 ) +1
        self.ATTN_0 = A2Net(16, m0, n0, t0, h0, w0, k_a0, s_a0)
        # -----------
        k1 = s1 = 3
        p1 = 1
        self.CNN_1  = nn.Sequential(
                        nn.Conv3d(16,16, kernel_size=k1, stride=[1,s1,s1], padding=p1),
                        nn.ReLU(),
                        nn.Conv3d(16,16, kernel_size=k1, stride=[1,s1,s1], padding=p1),
                        nn.ReLU()
                        )
        # -----------
        m1 = 16
        n1 = 8#32
        k_a1 = s_a1 = 1
        h1 = w1 = int( (h0 - k1 + 2*p1)/s1 ) +1
        h1 = w1 = int( (h1 - k1 + 2*p1)/s1 ) +1
        t1 = int( (t0 - k1 + 2*p1)/1 ) +1
        t1 = int( (t1 - k1 + 2*p1)/1 ) +1
        if args.startpoint_mode=="on":
            self.ATTN_1 = A2Net_Cond(16, m1, n1, t1, h1, w1, k_a1, s_a1, args.hidden_size+args.in_size)
        else:
            self.ATTN_1 = A2Net_Cond(16, m1, n1, t1, h1, w1, k_a1, s_a1, args.hidden_size)  
        # -----------
        k2 = 3
        s2 = 3
        p2 = 1
        self.CNN_2  = nn.Sequential(
                        nn.Conv3d(16, 16, kernel_size=k2, stride=[1,s2,s2], padding=p2),
                        nn.ReLU(),
                        )
        # -----------
        m2 = 16
        n2 = 8#8
        k_a2 = s_a2 = 1
        h2 = w2 = int( (h1 - k2 + 2*p2)/s2 ) +1
        t2 = int( (t1 - k2 + 2*p2)/1 ) +1
        if args.startpoint_mode=="on":
            self.ATTN_2 = A2Net_Cond(16, m2, n2, t2, h2, w2, k_a2, s_a2, args.hidden_size+args.in_size)  
        else:
            self.ATTN_2 = A2Net_Cond(16, m2, n2, t2, h2, w2, k_a2, s_a2, args.hidden_size)  
        # -----------
        # Global Average Pooling
        #self.GAP    = nn.AvgPool3d(kernel_size=[t2,h2,w2],stride=[t2,h2,w2])
        # -----------
        self.linear                             = nn.Sequential(
                                                    nn.Linear(16*h2*w2*t2 , args.hidden_size, bias=True),#+ hidden_size
                                                    nn.ReLU(),
                                                    nn.Dropout(p=args.dropout_val),
                                                    nn.Linear(args.hidden_size, args.hidden_size, bias=True),
                                                    nn.ReLU()
                                                    )    
        self.vsn_out_size = 16*h2*w2*t2#*16
        self.lastCNN_size = [h2, w2, t2]
        print("LAST conv size hwt\t", self.lastCNN_size)
        self.upsampling   = torch.nn.Upsample(size=(args.image_size,args.image_size,args.T_obs), mode='trilinear', align_corners=True)                 

    def forward(self, visual_input, condition):
         
        if self.args.startpoint_mode=="on":
            condition   = condition.view(-1, self.args.hidden_size + self.args.in_size)
        else:
            condition   = condition.view(-1, self.args.hidden_size)

        cc               = int(self.vsn_out_size/self.lastCNN_size[2]/self.lastCNN_size[0]/self.lastCNN_size[1])
        visual_input     = visual_input.view(-1,self.args.image_dimension,self.args.T_obs,self.args.image_size,self.args.image_size)
        cnn1             = self.CNN_0(visual_input)
        cnn2, attn2      = self.ATTN_0(cnn1)
        cnn3             = self.CNN_1(cnn2)
        cnn4, attn4      = self.ATTN_1(cnn3, condition)
        cnn5             = self.CNN_2(cnn4)
        cnn6, attn6      = self.ATTN_2(cnn5, condition)


        self.imgs        = cnn6.view(-1, cc, self.lastCNN_size[2], self.lastCNN_size[0], self.lastCNN_size[1]).requires_grad_(True)
        #cnn             = self.GAP(imgs)
        result           = self.linear(self.imgs.view(-1,self.vsn_out_size))
        # Calculate attn representations
        # weights         = self.linear[0].weight.sum(dim=0).view(self.vsn_out_size) #, dim=0)#weight[0]
        # attn_rep        = torch.einsum('bchwt,c->bhwt',imgs.detach(), weights.detach()).view(-1,1,self.lastCNN_size[0], self.lastCNN_size[1], self.lastCNN_size[2])
        # attn_rep_upsamp = self.upsampling(attn_rep).squeeze()


        return result, self.imgs, attn2,attn4,attn6#attn_rep_upsamp
    
    
    def forward(self, input, hidden): 
        # Coordination Embedding
        embedding                   = self.embedder_phi(input.view(-1,2))
        embedding                   = F.relu(self.dropout(embedding))
        # RNN
        output, hidden              = self.encoder(embedding.unsqueeze(1), ( hidden[0],hidden[1] ) )
        return output, hidden


    def initHidden(self,batch_size):
        self.batch_size=batch_size
        return torch.zeros([self.num_RRN_layers, self.batch_size, self.hidden_size]).cuda()#, device = device)

    def emb_out(self,input):
        out= self.embedder_out(input)
        return out


class A2Net(nn.Module):
    def __init__(self, in_channel, m, n, t, height, width, kernel_size, stride_size, batch_size=1):
        super(A2Net, self).__init__()
        self.m = m
        self.n = n
        self.width = width
        self.height = height
        self.t = t
        self.in_channel = in_channel
        self.Conv_Phi   = nn.Conv3d(in_channels=in_channel, out_channels=m, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Theta = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Rho   = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)
    
    def forward(self, input):
        input   = input.view(-1, self.in_channel, self.t, self.height, self.width)
        A       = self.Conv_Phi(input)
        B0      = self.Conv_Theta(input)
        # Softmax over thw dimension
        B       = F.softmax(B0.view(B0.size(0),B0.size(1),-1), dim=-1).view(B0.size()) 
        # 1st ATTN: Global Descriptors
        AB_T    = torch.einsum('bmthw, bnthw->bmn', A, B) 
        # 2nd ATTN: Attention Vectors
        # Softmax over n
        V       = F.softmax(self.Conv_Rho(input), dim=1)
        Z       = torch.einsum('bmn, bnthw->bmthw', AB_T, V)
        attn    = torch.einsum('bnthw, bnthw->bnhwt', B, V)
        return Z , attn#+input
class A2Net_Cond(nn.Module):
    def __init__(self, in_channel, m, n, t, height, width, kernel_size, stride_size, condition_size, batch_size=1):
        super(A2Net_Cond, self).__init__()
        self.m = m
        self.n = n
        self.width = width
        self.height = height
        self.t = t
        self.in_channel = in_channel
        self.Conv_Phi   = nn.Conv3d(in_channels=in_channel, out_channels=m, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Theta = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Rho   = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)

        self.FC_Cond    = nn.Sequential(
                                nn.Linear(condition_size, condition_size, bias=True),
                                nn.ReLU(),
                                nn.Linear( condition_size, height*width, bias=True),
                                nn.ReLU()
                                )
    
    def forward(self, input, condition):
        input       = input.view(-1, self.in_channel, self.t, self.height, self.width)
        condition   = condition.view(input.size(0), -1)

        A       = self.Conv_Phi(input)
        B0      = self.Conv_Theta(input)
        # Softmax over thw dimension
        B       = F.softmax(B0.view(B0.size(0),B0.size(1),-1), dim=-1).view(B0.size())
        # Produce vector C from the condition
        self.start_point= condition[:,(-2,-1)] # trajectory encoded + start point
        C               = F.softmax(self.FC_Cond(condition), dim=-1).view(-1, self.height, self.width)
        self.condition  = C
        # 1st ATTN: Global Descriptors
        BC      = torch.einsum('bnthw, bhw->bnthw', B, C) 
        AB_T    = torch.einsum('bmthw, bnthw->bmn', A, BC) 
        # AB_TC   = torch.einsum('bmn, bm->bmn', AB_T, C)
        # 2nd ATTN: Attention Vectors
        # Softmax over n
        V       = F.softmax(self.Conv_Rho(input), dim=1)
        Z       = torch.einsum('bmn, bnthw->bmthw', AB_T, V)
        attn    = torch.einsum('bnthw, bnthw->bnhwt', B, V)
        return Z , attn#+input

class Vision(nn.Module):
    def __init__(self, args):
        super(Vision, self).__init__()
        self.args = args
        k0 = s0 = 3
        p0 = 1
        self.CNN_0  = nn.Sequential(
                        nn.Conv3d(3, 16, kernel_size=k0, stride=[s0,s0,s0], padding=p0),
                        nn.ReLU(),
                        )
        # -----------
        m0 = 16
        n0 = 8#16
        k_a0 = s_a0 = 1
        h0 = w0 = int( (args.image_size - k0 + 2*p0)/s0 ) +1
        t0 = int( (args.T_obs - k0 + 2*p0)/s0 ) +1
        self.ATTN_0 = A2Net(16, m0, n0, t0, h0, w0, k_a0, s_a0)
        # -----------
        k1 = s1 = 3
        p1 = 1
        self.CNN_1  = nn.Sequential(
                        nn.Conv3d(16,16, kernel_size=k1, stride=[1,s1,s1], padding=p1),
                        nn.ReLU(),
                        nn.Conv3d(16,16, kernel_size=k1, stride=[1,s1,s1], padding=p1),
                        nn.ReLU()
                        )
        # -----------
        m1 = 16
        n1 = 8#32
        k_a1 = s_a1 = 1
        h1 = w1 = int( (h0 - k1 + 2*p1)/s1 ) +1
        h1 = w1 = int( (h1 - k1 + 2*p1)/s1 ) +1
        t1 = int( (t0 - k1 + 2*p1)/1 ) +1
        t1 = int( (t1 - k1 + 2*p1)/1 ) +1
        if args.startpoint_mode=="on":
            self.ATTN_1 = A2Net_Cond(16, m1, n1, t1, h1, w1, k_a1, s_a1, args.hidden_size+args.in_size)
        else:
            self.ATTN_1 = A2Net_Cond(16, m1, n1, t1, h1, w1, k_a1, s_a1, args.hidden_size)  
        # -----------
        k2 = 3
        s2 = 3
        p2 = 1
        self.CNN_2  = nn.Sequential(
                        nn.Conv3d(16, 16, kernel_size=k2, stride=[1,s2,s2], padding=p2),
                        nn.ReLU(),
                        )
        # -----------
        m2 = 16
        n2 = 8#8
        k_a2 = s_a2 = 1
        h2 = w2 = int( (h1 - k2 + 2*p2)/s2 ) +1
        t2 = int( (t1 - k2 + 2*p2)/1 ) +1
        if args.startpoint_mode=="on":
            self.ATTN_2 = A2Net_Cond(16, m2, n2, t2, h2, w2, k_a2, s_a2, args.hidden_size+args.in_size)  
        else:
            self.ATTN_2 = A2Net_Cond(16, m2, n2, t2, h2, w2, k_a2, s_a2, args.hidden_size)  
        # -----------
        # Global Average Pooling
        #self.GAP    = nn.AvgPool3d(kernel_size=[t2,h2,w2],stride=[t2,h2,w2])
        # -----------
        self.linear                             = nn.Sequential(
                                                    nn.Linear(16*h2*w2*t2 , args.hidden_size, bias=True),#+ hidden_size
                                                    nn.ReLU(),
                                                    nn.Dropout(p=self.args.dropout_val),
                                                    nn.Linear(args.hidden_size, args.hidden_size, bias=True),
                                                    nn.ReLU()
                                                    )    
        self.vsn_out_size = 16*h2*w2*t2#*16
        self.lastCNN_size = [h2, w2, t2]
        print("LAST conv size hwt\t", self.lastCNN_size)
        self.upsampling   = torch.nn.Upsample(size=(args.image_size,args.image_size,args.T_obs), mode='trilinear', align_corners=True)                 

    def forward(self, visual_input, condition):
         
        if self.args.startpoint_mode=="on":
            condition   = condition.view(-1, self.args.hidden_size + self.args.in_size)
        else:
            condition   = condition.view(-1, self.args.hidden_size)

        cc               = int(self.vsn_out_size/self.lastCNN_size[2]/self.lastCNN_size[0]/self.lastCNN_size[1])
        visual_input     = visual_input.view(-1,self.args.image_dimension,self.args.T_obs,self.args.image_size,self.args.image_size)
        cnn1             = self.CNN_0(visual_input)
        cnn2, attn2      = self.ATTN_0(cnn1)
        cnn3             = self.CNN_1(cnn2)
        cnn4, attn4      = self.ATTN_1(cnn3, condition)
        cnn5             = self.CNN_2(cnn4)
        cnn6, attn6      = self.ATTN_2(cnn5, condition)


        self.imgs        = cnn6.view(-1, cc, self.lastCNN_size[2], self.lastCNN_size[0], self.lastCNN_size[1]).requires_grad_(True)
        #cnn             = self.GAP(imgs)
        result           = self.linear(self.imgs.view(-1,self.vsn_out_size))
        # Calculate attn representations
        # weights         = self.linear[0].weight.sum(dim=0).view(self.vsn_out_size) #, dim=0)#weight[0]
        # attn_rep        = torch.einsum('bchwt,c->bhwt',imgs.detach(), weights.detach()).view(-1,1,self.lastCNN_size[0], self.lastCNN_size[1], self.lastCNN_size[2])
        # attn_rep_upsamp = self.upsampling(attn_rep).squeeze()


        return result, self.imgs, attn2,attn4,attn6#attn_rep_upsamp
    
  