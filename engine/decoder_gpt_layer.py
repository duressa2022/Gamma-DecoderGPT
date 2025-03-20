from engine.decoder_sub_layer import DecoderSubLayer
from engine.position_embedding import PosEmbedding
import torch.nn as nn
import torch

class GPTDecoder(nn.Module):
    def __init__(self, dk,dv,h,d_model,dff,n,seq_length,number_vocabs,rate=0.2):
        super(GPTDecoder,self).__init__()
        #create a positional and embedding layer:
        self.pos_embedding=PosEmbedding(d_model,seq_length,number_vocabs)

        #create a dropout layer for decoder
        self.dropout=nn.Dropout(rate)

        #create a layer of sub layer:number n
        self.decoders=nn.ModuleList([DecoderSubLayer(dk,dv,h,d_model,dff,rate)
                                    for _ in range(n)])
        
    def forward(self,input,combined_mask=None,training=False):
        #input:(batch_size,seq_length,d_model)
        #en_output:(batch_size,seq_length,d_model)

        #through embedding: (batch_size,seq_length,d_model)
        output=self.pos_embedding(input)

        #pass thr output through dropout layer 
        output=self.dropout(output)

        #itrate through each decoders
        for decoder in self.decoders:
            #pass through decoder layer
            output=decoder(output,combined_mask,training)
        
        #return thr output:(batch_size,seq_length,d_model)
        return output 
    
    

#test code
# dk=10
# dv=5
# d_model=5
# h=2
# dff=10
# n=6
# input1=torch.rand(size=(2,2,d_model))
# input2=torch.rand(size=(2,2,d_model))
# input3=torch.rand(size=(2,2,d_model))

# app=GPTDecoder(dk,dv,h,d_model,dff,n)
# output=app(input1,input2)
# print(output)


