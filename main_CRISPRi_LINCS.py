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
    data_dict=data.get_CRISPRi_LINCS_dataset()
    X_source,Y_source,X_target,Y_target=data_dict["X_source"],data_dict["Y_source"],data_dict["X_target"],data_dict["Y_target"]
    source_dataset_name,target_dataset_name=data_dict["source_dataset_name"],data_dict["target_dataset_name"]
    X_source_train,X_source_test,Y_source_train,Y_source_test=sklearn.model_selection.train_test_split(
        X_source,
        Y_source,
        test_size=0.2,
        random_state=123
    )

    net=model.FCModelNoDropNoBN(n_in_features=X_source.shape[1])
    net=net.cuda()
    trainer=train.DATrainer(net,torch.optim.Adam,{'lr':1e-4},source_dataset_name,target_dataset_name)
    trainer.alpha=1
    trainer.ddc_coeff=0.1
    trainer.domain_adv_coeff=0.1
    trainer.n_epochs=200
    trainer.fit(
        X_source_train,
        Y_source_train,
        X_target,
        Y_target,
        X_source_val=X_source_test,
        Y_source_val=Y_source_test,
        save=True  
    )
    
    model_fp="/data/liz0f/DANN_py3/models/%s_%s-model-epoch_best.pth"%(source_dataset_name,target_dataset_name)
    model_old=torch.load(model_fp)
    model_new=model.FCModelNoDropNoBN(n_in_features=X_source.shape[1])
    model_new.load_state_dict(model_old.state_dict())

    X_source_fc_transformed=model_new.transform(X_source,layer="fc")
    X_target_fc_transformed=model_new.transform(X_target,layer="fc")

    datester_fc_transformed=test.DomainAdaptationTester(X_source_fc_transformed,Y_source,X_target_fc_transformed,Y_target)
    datester_fc_transformed.show_tsne_plot(set="distribution",legend=False,legend_labels={1:"WTC-CRISPRi",2:"LINCS"})

    plt.xlabel("tSNE 1")
    plt.ylabel("tSNE 2")
    plt.legend(bbox_to_anchor=(1.40, 1))

    plt.savefig("./tsne-CRISPRi_LINCS.png",dpi=150,bbox_inches="tight")