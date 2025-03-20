from torch.utils.data import DataLoader,Dataset
from engine.sub_word_tokenizer import SubWordTokenizer
import numpy as np
import json
import torch
class DecoderDataSetSubWord(Dataset):
    def __init__(self,fileName,decoder_path,train_split=0.8,test_split=0.1,Number_seq=1000,number_token=10000,max_length=50):
        super().__init__()
        #set the max seq length 
        self.max_length=max_length
        #load the json data from the file
        with open(file=fileName,mode="r") as file:
            #assign the vcleanedData:var
            cleanedData=json.load(file)
        
        #remove extra data from loaded 
        self.DataSet=cleanedData[:Number_seq]

        #add <start> and end <eos> on the data
        for index,item in enumerate(self.DataSet):
            self.DataSet[index]["data"]=f"<START> {item["data"].lower()} <EOS>"
        
        #randomly shuffle the dataset 
        np.random.shuffle(self.DataSet)

        #prepare length for train set
        train_size=int(Number_seq*train_split)
        #prepare length for test set
        test_size=int(Number_seq*test_split)
        #prepare length for val set
        val_size=int(Number_seq*(1-train_split-test_split))

        #create a training dataset
        self.train_data=self.DataSet[:train_size]
        #create a validation dataset
        self.val_data=self.DataSet[train_size:train_size+val_size]
        #create a testing dataset
        self.test_data=self.DataSet[val_size+train_size:val_size+train_size+test_size]

        #create tokenizer for working with the dataset 
        self.tokenizer=SubWordTokenizer(number_token=number_token)
        #create dataset for tokenzing 
        decoder_set=[data["data"] for data in self.DataSet]
        #train the tokenizer based on data
        vocabs=self.tokenizer.tokenize(input=decoder_set,save_path=decoder_path)

        #collect training data for encoding
        train_data=[data["data"] for data in self.train_data]
        #encode collected data:token and ids
        train_e=self.tokenizer.encode(train_data)
        #vectorize encoded data:to max length
        self.train_v=self.tokenizer.vectorize(train_e,vocabs,self.max_length)

        #collect validation data for encoding
        val_data=[data["data"] for data in self.val_data]
        #encode collected data:tokens ,ids
        val_e=self.tokenizer.encode(val_data)
        #vectorize encoded data:to max length
        self.val_v=self.tokenizer.vectorize(val_e,vocabs,self.max_length)

        #collect validation data for encoding
        test_data=[data["data"] for data in self.test_data]
        #encode collected data:tokens ,ids
        test_e=self.tokenizer.encode(test_data)
        #vectorize encoded data:to max length
        self.test_v=self.tokenizer.vectorize(test_e,vocabs,self.max_length)
    
    def __len__(self):
        #get length of the dataset
        return len(self.train_v)
    def __getitem__(self, index):
        #get the elems at index
        x=self.train_v[index]
        return torch.tensor(x,dtype=torch.long)



        






