from collections import defaultdict
import pickle
import numpy as np
import json
class WordTokenizer:
    def __init__(self,max_length):
        #define max length for indices
        self.max_length=max_length
        #define special tokens 
        self.specials=["[PAD]","[CLS]","[SEP]","[START]","[END]","[UNK]"]
    def tokenizer(self,inputs,save_path):
        #input: corpase of texts
        #path to save tokenized:vocab
        #create:map for token:index 
        vocabulary=defaultdict(int)
        #create inital index:=0
        index=0
        #set the padding index:=0
        vocabulary["[PAD]"]=index
        index=index+1
        vocabulary["[UNK]"]=index
        index=index+1
        #iterate through the inputs
        for input in inputs:
            #split the input: based white space
            tokens=input.split()
            #iterate through tokens 
            for token in tokens:
                #check wether token is registerd
                if token not in vocabulary:
                    #register thr token
                    vocabulary[token]=index
                    #increment the index 
                    index=index+1
        #check if save path is provided
        if save_path:
            #open path as write mode
            with open(file=save_path,mode="wb") as file:
                #dumb binary object of vocaburay
                pickle.dump(vocabulary,file)
        #return the vocabs
        return vocabulary
    
    def load(self,saved_path):
        #open file as read mode
        with open(saved_path,"rb") as file:
            #load file from path 
            vocabulary=pickle.load(file)
            #return:read file for
            return vocabulary
        
    def vectorize(self,inputs,vocabs):
        #define result to store vector 
        result=[]
        #interate through the inputs
        for input in inputs:
            #split thr input based on: " "
            tokens=input.split()
            #collect index for tokes 
            current=[]
            for token in tokens:
                #check wether the token is in vocabs
                if token in vocabs:
                    #append token to the ans
                    current.append(vocabs[token])
                else:
                    #append unkown token 
                    current.append(vocabs["[UNK]"])
            #validate max length 
            current=current[:self.max_length]
            #add padding if less 
            current=current+[0]*(self.max_length-len(current))
            #append thr result 
            result.append(current)
        #change to numpy and return 
        return np.array(result)
    
    def build_token(self,ids,vocabs):
        #define mapping: index-->token 
        mapping={index:token for token,index in vocabs.items()}
        #get token: for corresponding->index 
        return [mapping[index] for index in ids]
    
    def build_sentences(self,tokens):
        #remove special tokens
        tokens=[token for token in tokens if token not in self.specials]
        #join and return 
        return " ".join(tokens)
# #test code 
# app=WordTokenizer(max_length=10)
# filename="C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\test.json"
# with open(file=filename,mode="r") as file:
#     data=json.load(file)
# inputs=[item["data"] for item in data]

# # vocabs=app.tokenizer(inputs,"C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\word.pkl")
# # print(vocabs)
# vocabs=app.load("C:\\Users\\HP\\Documents\\learningpath\\pytorch\\gamma-decoderGPT\\__app_model\\word.pkl")
# print(vocabs)

# test=["your name is what"]
# v=app.vectorize(test,vocabs)
# print(v)

# b=app.build_token(v[0],vocabs)
# print(b)

# s=app.build_sentences(b)
# print(s)


    



        