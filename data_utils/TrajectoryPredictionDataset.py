import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# args: table, image_size, T_obs, T_pred, T_total, in_size 
class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, ROOT_DIR, DB_PATH,cnx, args): 
        self.args = args   
        self.pos_df    = pd.read_sql_query("SELECT * FROM "+str(args.table), cnx)
        self.root_dir  = ROOT_DIR+'/visual_data'
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((args.image_size,args.image_size)), \
                                                         torchvision.transforms.ToTensor(), \
                                                         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.visual_data = []
        # read sorted frames
        for img in sorted(os.listdir(self.root_dir)): 
            self.visual_data.append(self.transform( Image.open(os.path.join(self.root_dir)+"/"+img) ))
        self.visual_data = torch.stack(self.visual_data)
       
    
    def __len__(self):
        return self.pos_df.data_id.max()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        extracted_df     = self.pos_df[ self.pos_df["data_id"] == idx ]
        
        tensor           = torch.tensor(extracted_df[['pos_x_delta','pos_y_delta']].values).reshape(-1,self.args.T_total,self.args.in_size)
        obs, pred        = torch.split(tensor,[self.args.T_obs,self.args.T_pred],dim=1)
        
        start_frames     = (extracted_df.groupby('data_id').frame_num.min().values/10).astype('int')
        extracted_frames = []
        for i in start_frames:            
            extracted_frames.append(self.visual_data[i:i+self.args.T_obs])
        frames = torch.stack(extracted_frames)
        start_frames = torch.tensor(start_frames)
        return obs, pred, frames, start_frames
    
    