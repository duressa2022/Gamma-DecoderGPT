from engine.multi_head_attention import MultiHeadAttention
from engine.layer_normalization import LayerNormlization
from engine.feed_forward_layer import FeedForwardLayer
import torch.nn as nn
import torch

class DecoderSubLayer(nn.Module):
    def __init__(self, dk,dv,h,d_model,dff,rate=0.2):
        super(DecoderSubLayer,self).__init__()
        #define masked multihead attention layer
        self.masked_attention=MultiHeadAttention(dk,dv,h,d_model)

        #define first skip and normalization layer
        self.skip_norm1=LayerNormlization(d_model)

        #define first dropout layer
        self.dropout1=nn.Dropout(rate)

        #define feedforward layer 
        self.feed=FeedForwardLayer(d_model,dff)

        #define third skip and normalization layer
        self.skip_norm2=LayerNormlization(d_model)

        #define third dropout layer 
        self.dropout2=nn.Dropout(rate)
    
    def forward(self,input,combined_mask=None,training=False):
        #input:(batch_size,seq_length,d_model)
        #en_output: (batch_size,seq_length,d_model)

        #pass through masked attention layer
        masked=self.masked_attention(input,input,input,combined_mask)

        #pass through first dropout layer 
        masked=self.dropout1(masked) if training else masked

        #pass through skip-normalization layer
        output=self.skip_norm1(input,masked)

        #pass through feed forward layer
        feed=self.feed(output)

        #pass through dropout layer 
        feed=self.dropout2(feed) if training else feed

        #pass through thrid layer normalization layer 
        output=self.skip_norm2(output,feed)

        return output
#test code
# dk=10
# dv=5
# d_model=5
# h=2
# dff=10
# input1=torch.rand(size=(2,2,d_model))
# input2=torch.rand(size=(2,2,d_model))
# input3=torch.rand(size=(2,2,d_model))

# app=DecoderSubLayer(dk,dv,h,d_model,dff)
# output=app(input1,input2)
#print(output)

