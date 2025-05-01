import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.tools import MoE


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.input_size = args.seq_len  
        self.hidden_size = args.AE_hidden_size
        self.code_size = args.AE_code_size
        self.num_experts = args.num_experts
        self.input_dim = self.code_size
        self.hidden_dim = self.hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.code_size),
            nn.ReLU()
        )
        #self.moe = MoE(self.num_experts, self.input_dim, self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.code_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size),
            nn.Sigmoid()  
        )
        
    def forward(self, x, batch_x_mark, y, batch_y_mark, mask):
        batch_size, tim_steps, num_features = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous().view(-1, tim_steps)
        

        code = self.encoder(x)
        #code, moe_weight = self.moe(code)
        out = self.decoder(code)
        
        out = out.view(batch_size, num_features, tim_steps)
        out = out.transpose(1, 2)
        
        #return out, moe_weight
        return out
