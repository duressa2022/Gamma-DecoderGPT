import torch.nn as nn
class FeedForwardLayer(nn.Module):
    def __init__(self,d_model,d_ff):
        super(FeedForwardLayer,self).__init__()
        #create linear layer:(d_model,d_ff)
        self.f1=nn.Linear(in_features=d_model,out_features=d_ff)

        #create an activatiion layer:Relu
        self.a=nn.ReLU()

        #create linear layer:(d_ff,d_model)
        self.f2=nn.Linear(in_features=d_ff,out_features=d_model)

    def forward(self,input):
        #pass through f1
        output=self.f1(input)

        #pass through RELU
        output=self.a(output)

        #pass through f2
        return self.f2(output)

