import torch.nn as nn
class LayerNormlization(nn.Module):
    def __init__(self, d_model):
        super(LayerNormlization,self).__init__()
        #create normalization layer:dim=d_model
        self.norm_layer=nn.LayerNorm(d_model)

    def forward(self,input,output):
        #input:(batch_size,seq_length,d_model)->from pre layer
        #output:(batch_size,seq_length,d_model)->from post layer 
        #skip connection and layer normalization 
        return self.norm_layer(input+output)
