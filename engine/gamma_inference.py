from engine.gamma_decoderGPT import GammaGPT
import torch
import torch.nn as nn
import pickle
from engine.word_level_tokenizer import WordTokenizer
import numpy as np

app=WordTokenizer(max_length=50)

v=app.load("C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\decoder_token.pkl")


class gammaInference:
    def __init__(self,model:GammaGPT,tokenizer,device,seq_length):
        self.model=model 
        self.tokenizer=tokenizer
        self.device=device
        self.seq_length=seq_length

    def __call__(self,prompt,max_length):
        mapping={index:token for token,index in v.items()}
        #encode input: prompt 
        vectorized=app.vectorize([prompt],v)[0]
        generated=[]
        for val in vectorized:
            if val!=0:
                generated.append(val)


        # vectorized=encoded.ids+[0]*(self.seq_length-len(encoded.ids))
        #convert to torch.long
        vectorized=torch.tensor(vectorized,dtype=torch.long)
        cloned=[val for value in generated]
        generated=torch.tensor(generated,dtype=torch.long)

        for _ in range(max_length-vectorized.size(0)):
            with torch.no_grad():
                logit=self.model(generated.unsqueeze(0).to(self.device))
                
                next_token=logit[:,-1,:]
                k=next_token.topk(7).indices.tolist()[0]

                index=np.random.choice(np.array(k))
                
                # next_token=torch.argmax(next_token,dim=-1,keepdim=True)
                cloned.append(index)
                generated=torch.tensor(cloned,dtype=torch.long)
                print(mapping[index])
                
                
                
                

# Model hyperparameters
dk = 64
dv = 64
h = 8
d_model = 512
dff = 2048
seq_length = 50
number_tokens = 10000
drop_rate = 0.2
n = 6
batch_size = 64
epochs = 10
load_epoch_number=2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_path = "C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\decoder_token.pkl"
model_path=f"C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\models\\weights_epoch_{load_epoch_number}.pt"


#load tokeninzer 
def load_tokenizer(path):
    with open(path,"rb") as file:
        toknizer=pickle.load(file)
    return toknizer


tokenizer=load_tokenizer(decoder_path)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=GammaGPT(dk,dv,h,d_model,dff,n,seq_length,number_tokens,device,drop_rate).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
inference=gammaInference(model,tokenizer,device,seq_length)

inference("teacher",100)


