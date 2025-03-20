import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
accu_eval_dir=os.path.join("evals","accuracy.pkl")
loss_eval_dir=os.path.join("evals","loss.pkl")

with open(accu_eval_dir,"rb") as file:
    acc=pickle.load(file)

with open(loss_eval_dir,"rb") as file:
    loss=pickle.load(file)

epoch=np.array([key for key in acc])
loss=np.array([loss[key] for key in loss])
acc=np.array([acc[key] for key in acc])

plt.plot(epoch,acc,label="Model Accu")
plt.plot(epoch,loss,label="Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss/Acc")
plt.legend(loc="best")
plt.xticks(np.arange(1,len(loss)+1,1))
plt.show()
