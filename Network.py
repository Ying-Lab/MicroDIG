import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class AE_model(nn.Module):
    def __init__(self,input_dimension):
        super(AE_model,self).__init__()
        self.input = input_dimension
        self.hidden1 = 64
        self.hidden2 = 32
        self.encoder=nn.Sequential(
            nn.Linear(self.input,self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1,self.hidden2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden2,self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1,self.input)
        )
        self.compute = nn.Sequential(
            nn.Linear(self.hidden2*2,self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2,1)
        )
    
    def forward(self,x):
        z = self.encoder(x)
        rec_output = self.decoder(z)
        if z.ndim==3:
            corr_output = self.compute(z.reshape(z.shape[0],1,-1))
        elif z.ndim==2:
            corr_output = self.compute(z.reshape(1,-1))
        return rec_output,corr_output
    
class AE_3_model(nn.Module):
    def __init__(self,input_dimension):
        super(AE_3_model,self).__init__()
        self.input = input_dimension
        self.hidden1 = 64
        self.hidden2 = 32
        self.encoder=nn.Sequential(
            nn.Linear(self.input,self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1,self.hidden2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden2,self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1,self.input)
        )
        self.compute = nn.Sequential(
            nn.Linear(self.hidden2*2,self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2,3)
        )
    
    def forward(self,x):
        z = self.encoder(x)
        rec_output = self.decoder(z)
        if z.ndim==3:
            corr_output = self.compute(z.reshape(z.shape[0],1,-1))
        elif z.ndim==2:
            corr_output = self.compute(z.reshape(1,-1))
        return rec_output,corr_output

#----------------------------------------------------------------------------------

class MLP_model(nn.Module):
    def __init__(self,input_dimension):
        super(MLP_model,self).__init__()
        self.input = input_dimension
        self.hidden1 = 64
        self.hidden2 = 32
        self.encoder=nn.Sequential(
            nn.Linear(self.input,self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1,self.hidden2)    
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden2*2,self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2,1)
        )
    
    def forward(self,x):
        z = self.encoder(x)
        if z.ndim==3:
            output = self.mlp(z.reshape(z.shape[0],1,-1))
        elif z.ndim==2:
            output = self.mlp(z.reshape(1,-1))
        return output
    
class CNN_model(nn.Module):
    def __init__(self,input_dimension):
        super(CNN_model,self).__init__()
        self.input = input_dimension
        self.hidden1 = 64
        self.hidden2 = 32
        self.encoder=nn.Sequential(
            nn.Linear(self.input,self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1,self.hidden2)    
        )
        self.conv1 = nn.Conv2d(1,3,kernel_size=2)
        self.maxpool1 = nn.AdaptiveAvgPool2d(output_size=(2,20))
        self.conv2 = nn.Conv2d(3,1,kernel_size=2)
        self.maxpool2 = nn.AdaptiveAvgPool2d(output_size=(1,10))
        self.linear = nn.Linear(10,1)

    
    def forward(self,x):
        z = self.encoder(x)
        z = z.unsqueeze(dim=1)
        z = self.conv1(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        out = self.linear(z)

        return out
    
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.constant_(m.bias, 0.0)

class VAE(nn.Module):
    def __init__(self,input_dimension,only_decoder=False):
        super(VAE,self).__init__()
        self.only_decoder = only_decoder
        self.hidden1 = 400
        self.hidden2 = 200
        # encoder
        self.fc1 = nn.Linear(input_dimension,self.hidden1)
        self.fc21 = nn.Linear(self.hidden1,self.hidden2)
        self.fc22 = nn.Linear(self.hidden1,self.hidden2)
        #decoder
        self.fc3 = nn.Linear(self.hidden2,self.hidden1)
        self.fc4 = nn.Linear(self.hidden1,input_dimension)

    def encoder(self,x):
        h1 = nn.functional.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu,logvar
    
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decoder(self, z):
        h3 = nn.functional.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self,x):
        if self.only_decoder:
            output=self.decoder(x)
            return output
        else:
            mu,logvar = self.encoder(x)
            z = self.reparameterize(mu,logvar)
            output = self.decoder(z)
            return output,z,mu,logvar

class GNN_model(nn.Module):
    def __init__(self,node_features,hidden_channels): 
        super(GNN_model,self).__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = x.to(torch.float32)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        output = self.lin(x)
        return output
