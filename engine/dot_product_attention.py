import torch
import torch.nn as nn
import torch.nn.functional as f

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention,self).__init__()

    def forward(self,queries,keys,values,dk,mask=None):
        #queries:(batch_size,seq_length,dk)
        #keys:(batch_size,seq_length,dk)
        #values:(batch_size,seq_length,dv)
        #mask:(batch_size,seq_length,seq_length)

        #calculate attention score:(batch_size ,seq_length,seq_length)
        score=torch.matmul(queries,keys.transpose(-1,-2))/torch.sqrt(torch.tensor(dk,dtype=torch.float32))
    
        #padding or look ahead mask:(batch_size,seq_length,seq_length)
        if mask is not None:
            score=score+mask*(-1e9)
        
        #apply softmax for probs cal:(batch_size,seq_length,seq_length)
        score=f.softmax(score,dim=-1)

        #apply weighted values for seq:(batch_size,seq_length,dv)
        values=torch.matmul(score,values)

        return values
    
#test code 
# app=DotProductAttention()
# q=torch.rand(size=(2,4,4))
# k=torch.rand(size=(2,4,4))
# v=torch.rand(size=(2,4,5))
# m=torch.triu(torch.ones(size=(2,4,4)),diagonal=1)
# values=app(q,k,v,4,m)
# print(values)

        


