import json
from engine.word_level_tokenizer import WordTokenizer
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
class DecoderDatasetWord(Dataset):
    def __init__(self,filename,tokenizer_path,train_split=0.8,test_split=0.1,number_data=10000,max_length=50):
        super(DecoderDatasetWord,self).__init__()
        #defne maxlength for the seq 
        self.max_length=max_length
        #open the file as read mode 
        with open(file=filename,mode="r") as file:
            #load the json file and store to val
            cleaned_data=json.load(file)
        #limit the number of data 
        self.DataSet=cleaned_data[:number_data]
        #randomly shuffle the dataset 
        np.random.shuffle(self.DataSet)
        #process datset by adding <start> and <eos>
        for index,item in enumerate(self.DataSet):
            #add <start> and <eos> of data indicator
            self.DataSet[index]["data"]=f"{item["data"].lower()}"
            
        #create a data for training tokenizer
        data_for_token=[item["data"] for item in self.DataSet]
        #train tokenizer for gen vocabulary 
        self.tokenizer=WordTokenizer(max_length=self.max_length)
        #train the tokenizer based on vocab
        vocabs=self.tokenizer.tokenizer(data_for_token,tokenizer_path)

        #create train data size 
        train_length=int(number_data*train_split)
        #create test data size
        test_length=int(number_data*test_split)
        #create val data size
        val_length=int(number_data*(1-train_split-test_split))

        #create a data for training 
        self.train=self.DataSet[:train_length]
        #create a data for testing 
        self.test=self.DataSet[train_length:train_length+test_length]
        #create a data for validation 
        self.val=self.DataSet[train_length+test_length:train_length+test_length+val_length]

        #create a train data for vectorizing 
        train_data=[item["data"] for item in self.train]
        #create a test data for vectorizing
        test_data=[item["data"] for item in self.test]
        #create a val data for vectorizing
        val_data=[item["data"] for item in self.val]

        #vectorize the traning set 
        self.trainV=self.tokenizer.vectorize(train_data,vocabs)
        #vectorize the testing set 
        self.testV=self.tokenizer.vectorize(test_data,vocabs)
        #vectorize the validation set
        self.valV=self.tokenizer.vectorize(val_data,vocabs)
        
    def __len__(self):
        #get length of dataset 
        return len(self.train)
    def __getitem__(self, index):
        #get item at give index 
        x=self.trainV[index]
        return torch.tensor(x,dtype=torch.long)
    
# #test code 

# #define path for tokenizer
# tokenizer_path="C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\word.pkl"
# #define path for file/data
# filename="C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\test.json"
# #build the dataset
# dataset=DecoderDatasetWord(filename,tokenizer_path)
# #build the dataloader
# loader=DataLoader(dataset,batch_size=32,shuffle=True)
# #print x and y value
# for x,y in loader:
#     print(x)


