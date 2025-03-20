# Model hyperparameters
import pickle
import torch
from engine.gamma_decoderGPT import GammaGPT
from engine.gamma_inference import gammaInference
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

def chat(prompt):
    return inference(prompt)
