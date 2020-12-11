import sklearn.preprocessing
import sklearn.ensemble
import pandas as pd
import numpy as np
import os
import sklearn.model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.manifold
import model
import sklearn
import sklearn.linear_model
import sklearn.svm
import scipy.stats
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms
import pickle as pkl
SEED=0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)

import data
import test
import model
import train

if __name__=="__main__":
    data_dict=data.get_KMPLOT_dataset()
    X_source,Y_source,X_target,Y_target=data_dict["X_source"],data_dict["Y_source"],data_dict["X_target"],data_dict["Y_target"]
    source_dataset_name,target_dataset_name=data_dict["source_dataset_name"],data_dict["target_dataset_name"]
    X_source_train,X_source_test,Y_source_train,Y_source_test=sklearn.model_selection.train_test_split(
        X_source,
        Y_source,
        test_size=0.2,
        random_state=123
    )

    net=model.FCModelNoDropNoBNLarge(n_in_features=X_source.shape[1])
    net=net.cuda()
    trainer=train.DATrainer(net,torch.optim.Adam,{'lr':1e-3},source_dataset_name,target_dataset_name)
    trainer.n_epochs=400
    trainer.fit(
        X_source_train,
        Y_source_train,
        X_target,
        Y_target,
        X_source_val=X_source_test,
        Y_source_val=Y_source_test,
        save=True  
    )
    trainer.alpha = 1
    trainer.domain_adv_coeff = 1e-1
    trainer.ddc_coeff= 1e-2
    score_fp='{}/{}_{}-score.pkl'.format(trainer.model_root,trainer.source_dataset_name,trainer.target_dataset_name)
    with open(score_fp,"rb") as f:
        score=pkl.load(f)
    print("best source domain test accuracy: ",max(score["class_accuracy_source_val"]))
    print("best target domain accuracy: ",max(score["class_accuracy_target"]))

    
