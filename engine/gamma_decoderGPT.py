from engine.decoder_gpt_layer import GPTDecoder
import torch.nn as nn
import torch
class GammaGPT(nn.Module):
    def __init__(self,dk,dv,h,d_model,dff,n,seq_length,number_vocabs,device,rate=0.2):
        super(GammaGPT,self).__init__()
        self.device=device

        #define decoder layer:with params
        self.decoder=GPTDecoder(dk,dv,h,d_model,dff,n,seq_length,number_vocabs,rate)

        #define final connected layer:linear
        self.connected=nn.Linear(in_features=d_model,out_features=number_vocabs)

    def combined_mask(input):
        #input: (batch_size,seq_length)

        #padding mask:(batch_size,seq_len)
        padding_mask=(input==0).float()


        #ahead mask:(batch_size,seq_len)
        ahead_mask=torch.triu(torch.ones(size=(input.size(0),input.size(1))),diagonal=1)
        
        #combined mask:(batch_size ,seq_len)
        combined=torch.max(padding_mask,ahead_mask)

        #return combined mask|data 
        return combined
    
    def padding_mask(self, input):
        # input: (batch_size, seq_length)
        mask = (input == 0).float()
        return mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)

    def lookahead_mask(self, length):
        # Create causal mask
        mask = 1 - torch.tril(torch.ones((length, length), dtype=torch.float32))
        return mask.to(self.device)
        
    
    def forward(self,input,training=False):
        # input_sequence: (batch_size, seq_length)
        # Create look-ahead mask for causal attention
        look_ahead_mask = self.lookahead_mask(input.size(1))

        # Create padding mask to ignore padding tokens
        padding_mask = self.padding_mask(input)

        # Combine masks: max ensures padding and future tokens are masked
        combined_mask = torch.max(padding_mask, look_ahead_mask.unsqueeze(0).unsqueeze(0))

        # Pass through decoder with both masks
        decoder_output = self.decoder(input, combined_mask, training)
        model_output = self.connected(decoder_output)
        return model_output
