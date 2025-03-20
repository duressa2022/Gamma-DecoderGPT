import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from engine.decoder_dataset_sub import DecoderDataSetSubWord
from engine.gamma_decoderGPT import GammaGPT
from engine.decoder_dataset_word import DecoderDatasetWord
import pickle

# Define the learning rate scheduler
class LRscheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.steps = 0
    
    def step(self):
        self.steps += 1
        lr = (self.d_model ** -0.5) * min(self.steps ** -0.5, self.steps * (self.warmup_steps ** -1.5))
        for group_params in self.optimizer.param_groups:
            group_params["lr"] = lr

# Define padding mask function
def padding_mask(input):
    return (input != 0).float()

# Define loss function
def loss_fn(target, prediction):
    criterion = nn.CrossEntropyLoss(reduction="none")
    mask = padding_mask(target)
    loss = criterion(prediction.transpose(1, 2), target)
    loss = loss * mask
    return loss.sum() / mask.sum()

# Define accuracy function
def accuracy_fn(target, prediction):
    ids = torch.argmax(prediction, dim=-1)
    mask = padding_mask(target)
    matching = (ids == target).float()
    matching = matching * mask
    return matching.sum() / mask.sum()

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and build dataset
filename = "C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\english_data.json"
decoder_path = "C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\decoder_token.pkl"
dataset = DecoderDatasetWord(filename=filename, tokenizer_path=decoder_path)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Create model, optimizer, and scheduler
model = GammaGPT(dk, dv, h, d_model, dff, n, seq_length, number_tokens, device,drop_rate)
model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = LRscheduler(optimizer, d_model, warmup_steps=4000)

# Training loop
epoch_loss={}
epoch_accc={}

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    for index, x in enumerate(dataloader):
        input_batch = x[:, :-1]  
        targ_batch = x[:, 1:]  


        input_batch = input_batch.to(device=device)
        targ_batch = targ_batch.to(device=device)

        prediction = model(input_batch, training=True)
        loss = loss_fn(targ_batch, prediction)
        accuracy = accuracy_fn(targ_batch, prediction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1

        print(f"Epoch: {epoch+1}/{epochs}, Batch: {index+1}/{len(dataloader)}, "
              f"Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.7f}")

    # Average loss and accuracy for the epoch
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    epoch_loss[epoch]=avg_loss
    epoch_accc[epoch]=avg_accuracy

    print(f"Epoch {epoch+1} Summary - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")


    if (epoch + 1) % 2 == 0:
        model_dir = os.path.join("models", f"weights_epoch_{epoch+1}.pt")
        os.makedirs("models", exist_ok=True)  
        torch.save(model.state_dict(), model_dir)
        print(f"Saved model checkpoint to {model_dir}")

accu_eval_dir=os.path.join("evals","accuracy.pkl")
loss_eval_dir=os.path.join("evals","loss.pkl")
os.makedirs("evals",exist_ok=True)

with open(accu_eval_dir,"wb") as file:
    pickle.dump(avg_loss,file)

with open(loss_eval_dir,"wb") as file:
    pickle.dump(loss_eval_dir)
