import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
import data
from torchvision import datasets
import sklearn.model_selection
import sklearn.metrics
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import contextlib
from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import scipy.stats


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class DomainAdaptationTester(object):
    def __init__(self,X1,Y1,X2,Y2):
        if X2 is not None:
            assert X1.shape[1:]==X2.shape[1:]
        if Y1 is not None:
            assert len(X1)==len(Y1) 
        if Y2 is not None:
            assert len(X2)==len(Y2)
        self.X1,self.Y1,self.X2,self.Y2=X1,Y1,X2,Y2
    def test_on_source(self,model,test_size=0.2):
        assert self.X1 is not None and self.Y1 is not None
        X1_train,X1_test,Y1_train,Y1_test=sklearn.model_selection.train_test_split(self.X1,self.Y1,test_size=test_size,random_state=123)
        if isinstance(model,nn.Module):
            model.fit(X1_train,Y1_train,test_data=(X1_test,Y1_test))
        else:
            model.fit(X1_train,Y1_train)
        scores=model.score(X1_test,Y1_test)
        return scores
    def test_on_target(self,model,test_size=0.2):
        assert self.X2 is not None and self.Y2 is not None
        X2_train,X2_test,Y2_train,Y2_test=sklearn.model_selection.train_test_split(self.X2,self.Y2,test_size=test_size,random_state=123)
        if isinstance(model,nn.Module):
            model.fit(X2_train,Y2_train,test_data=(X2_test,Y2_test))
        else:
            model.fit(X2_train,Y2_train)
        scores=model.score(X2_test,Y2_test)
        return scores
    def test_transfer(self,model):
        assert self.X1 is not None and self.Y1 is not None
        assert self.X2 is not None and self.Y2 is not None

        if isinstance(model,nn.Module):
            model.fit(self.X1,self.Y1,test_data=(self.X2,self.Y2))
        else:
            model.fit(self.X1,self.Y1)
        scores=model.score(self.X2,self.Y2)
        return scores
    def test_distribution(self,model,test_size=0.2):
        assert self.X1 is not None and self.X2 is not None
        X=np.concatenate([self.X1,self.X2],axis=0)
        Y=np.array([0]*len(self.X1)+[1]*len(self.X2))
        X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=test_size,random_state=456)
        if isinstance(model,nn.Module):
            model.fit(X_train,Y_train,test_data=(X_test,Y_test))
        else:
            model.fit(X_train,Y_train)
        scores=model.score(X_test,Y_test)
        return scores

    def __show_tsne_plot(self,X,labels,legend=True,legend_label=None):
        unique_labels=np.unique(labels)
        unique_labels=unique_labels[~np.isnan(unique_labels)]
        print(unique_labels)
        n_labels=len(unique_labels)
        
        tsne=TSNE(n_jobs=50,random_state=789)
        X_2d=tsne.fit_transform(X)
        
        plt.figure()
        X_2d_X=X_2d[:,0]
        X_2d_Y=X_2d[:,1]
        for i,c in enumerate(unique_labels):
            idx=(labels==c)
            if legend_label is None:
                plt.plot(X_2d_X[idx],X_2d_Y[idx],".",markersize=1,label=str(c))
            else:
                plt.plot(X_2d_X[idx],X_2d_Y[idx],".",markersize=1,label=legend_label[c])
        if legend:
            plt.legend(bbox_to_anchor=(1.30, 1))

    def show_tsne_plot(self,set="all",sample=2000,legend=True,legend_labels=None):
        sample=min(sample,len(self.X1),len(self.X2))
        with temp_seed(124):
            idx_sample1=np.random.choice(len(self.X1),size=sample,replace=False)
            if self.X2 is not None:
                idx_sample2=np.random.choice(len(self.X2),size=sample,replace=False)

        if set=="source":
            assert self.X1 is not None and self.Y1 is not None
            X_sample=np.take(self.X1,idx_sample1,0)
            Y_sample=np.take(self.Y1,idx_sample1,0)
        elif set=="target":
            assert self.X2 is not None and self.Y2 is not None
            X_sample=np.take(self.X2,idx_sample2,0)
            Y_sample=np.take(self.Y2,idx_sample2,0)
        elif set=="all":
            assert self.X1 is not None and self.Y1 is not None
            assert self.X2 is not None  

            X_sample1=np.take(self.X1,idx_sample1,0)
            Y_sample1=np.take(self.Y1,idx_sample1,0)
            X_sample2=np.take(self.X2,idx_sample2,0)
            if self.Y2 is not None:
                Y_sample2=np.take(self.Y2,idx_sample2,0)
            else:
                Y_sample2=np.array([-1]*sample)

            X_sample=np.concatenate([X_sample1,X_sample2],axis=0)
            Y_sample=np.concatenate([Y_sample1,Y_sample2],axis=0)
        elif set=="distribution":
            assert self.X1 is not None and self.X2 is not None
            X_sample1=np.take(self.X1,idx_sample1,0)
            X_sample2=np.take(self.X2,idx_sample2,0)
            X_sample=np.concatenate([X_sample1,X_sample2],axis=0)

            domain_label=np.array([1]*sample+[2]*sample)
            Y_sample=domain_label

        else:
            assert False
        self.__show_tsne_plot(X_sample,Y_sample,legend,legend_labels)



def score(model,X,Y,batch_size=128):
    model.eval()
    if np.min(Y)==1:
        Y=Y-1
    else:
        assert np.min(Y)==0

    X_tensor=torch.FloatTensor(X)
    Y_tensor=torch.LongTensor(Y)

    test_dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=0)
    
    pred_list=list()
    device=next(iter(model.parameters())).device
    with torch.no_grad():
        for i,(X_batch,Y_batch) in enumerate(test_loader):
            X_batch=X_batch.to(device)
            Y_batch=Y_batch.to(device)
            pred,_=model(X_batch,1.0)
            pred=pred.argmax(1)
            pred_list.append(pred)
    predictions_tensor=torch.cat(pred_list)
    predictions_arr=predictions_tensor.cpu().numpy()
    score=dict()
    score["accuracy"]=np.mean(predictions_arr==Y)
    score["confusion_matrix"]=sklearn.metrics.confusion_matrix(Y,predictions_arr)

    model.train()
    return score


def predict(model,X,batch_size=128,return_score=True):
    model.eval()
    X_tensor=torch.FloatTensor(X)
    test_dataset=torch.utils.data.TensorDataset(X_tensor)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=0)
    pred_list=list()
    device=next(iter(model.parameters())).device
    with torch.no_grad():
        for _,(X_batch,) in enumerate(test_loader):
            X_batch=X_batch.to(device)
            pred,_=model(X_batch,1.0)
            if return_score:
                pred=torch.softmax(pred,dim=1)
            else:
                pred=pred.argmax(1)
            pred_list.append(pred)
    predictions_tensor=torch.cat(pred_list,0)
    predictions_arr=predictions_tensor.cpu().numpy()
    model.train()
    return predictions_arr

def test(dataset_name):
    assert dataset_name in ['MNIST', 'mnist_m']

    model_root = 'models'
    image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = data.GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        dataset = datasets.MNIST(
            root='dataset',
            train=False,
            transform=img_transform_source,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu


def clone_state(net):
    params_dict=dict()
    for name,param in net.state_dict().items():
        params_dict[name]=param.detach().clone()
    return params_dict

def check_state_equivalence(param_dict1,param_dict2):
    for name, param in param_dict1.items():
        if torch.allclose(param,param_dict2[name]):
            print("%s: same"%(name))
        else:
            max_diff=torch.max(torch.abs(param-param_dict2[name]))
            print("%s: different, max difference: %.5f"%(name,max_diff))

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

class LINCSRankingEvaluator(object):
    def __init__(self,prediction_df):
        level5_fs_table_fp="/data/liz0f/reprogramming/LINCS/level5_logtrans_DE_genes.hdf"
        self.level5_fs_df_T=pd.read_hdf(level5_fs_table_fp).T
        assert set(prediction_df.index)==set(self.level5_fs_df_T.index)

        deccode_plurip_fp="/data/liz0f/reprogramming/my-ad-ae/rankedLINCSprofiles_ testedDrugs_cellClusters.csv"
        self.deccode_plurip_df=pd.read_csv(deccode_plurip_fp)
        deccode_plurip_grouped_df=self.deccode_plurip_df.groupby(["pert_name"]).mean()

        siginfo_fp = "/data/liz0f/reprogramming/LINCS/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt"
        siginfo_df=pd.read_csv(siginfo_fp,sep='\t',index_col=0)
        self.siginfo_df=siginfo_df
        self.prediction_df=prediction_df
        prediction_df_copy=prediction_df.copy()
        prediction_df_copy.insert(0,"pert_name",siginfo_df["pert_iname"])
        prediction_df_copy.insert(1,"cell_id",siginfo_df["cell_id"])

        prediction_mean_df=prediction_df_copy.groupby(["pert_name","cell_id"]).mean().reset_index()
        prediction_grouped_df=prediction_mean_df.groupby(["pert_name"]).mean()

        intersection_index=prediction_grouped_df.index.intersection(deccode_plurip_grouped_df.index)

        self.deccode_plurip_grouped_df=deccode_plurip_grouped_df.loc[intersection_index]
        self.prediction_grouped_df=prediction_grouped_df.loc[intersection_index]

        exp_results_single_fp="/data/liz0f/reprogramming/my-ad-ae/exp_results_single.csv"
        exp_results_single_df=pd.read_csv(exp_results_single_fp).iloc[:,1:]
        self.exp_results_single_df=exp_results_single_df
        exp_results_single_grouped_df=exp_results_single_df.set_index("pert_name")

        self.df={
            'deccode_plurip':self.deccode_plurip_grouped_df,
            'prediction':self.prediction_grouped_df,
            'exp_results_single': exp_results_single_grouped_df
        }

    def scatter_plot(self,x_axis=("prediction",0),y_axis=("deccode_plurip","PS"),axis_label=True,x_axis_min_filter=None):
        assert x_axis[0] in self.df or x_axis[1] in self.df[x_axis[0]].columns
        assert y_axis[0] in self.df or y_axis[1] in self.df[y_axis[0]].columns

        if x_axis_min_filter is not None:
            filter_idx=self.df[x_axis[0]].index[self.df[x_axis[0]][x_axis[1]]>=x_axis_min_filter]
        else:
            filter_idx=self.df[x_axis[0]].index

        filter_idx=filter_idx.intersection(self.df[y_axis[0]][y_axis[1]].index)
        x_axis_filtered_df=self.df[x_axis[0]][x_axis[1]].loc[filter_idx]

        
        y_axis_filtered_df=self.df[y_axis[0]][y_axis[1]].loc[filter_idx]
        
        sr=scipy.stats.spearmanr(x_axis_filtered_df,y_axis_filtered_df)
        sr=sr[0]
        plt.plot(x_axis_filtered_df,y_axis_filtered_df,'.')

        lr=sklearn.linear_model.LinearRegression()
        lr.fit(x_axis_filtered_df.values[:,None],y_axis_filtered_df)
        abline(lr.coef_[0],lr.intercept_)
        ax=plt.gca()
        plt.text(0.5,0.05,"Spearman's Correlation = %.3f"%(sr),transform=ax.transAxes)
        if axis_label:
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
    
    def box_plot(self,x_axis=("prediction",0),y_axis=("deccode_plurip","PS"),axis_label=True,n=30):
        assert x_axis[0] in self.df or x_axis[1] in self.df[x_axis[0]].columns
        assert y_axis[0] in self.df or y_axis[1] in self.df[y_axis[0]].columns

        # x_axis_df=self.df[x_axis[0]][x_axis[1]]
        # y_axis_df=self.df[y_axis[0]][y_axis[1]]

        # x_axis_df_idx_sorted=x_axis_df.argsort()
        # x_axis_df_bottom_n=x_axis_df_idx_sorted[:n]
        # x_axis_df_top_n=x_axis_df_idx_sorted[-n:]
        # x_axis_df_middle=x_axis_df_idx_sorted[n:-n]
        # y_axis_bottom_n=y_axis_df[x_axis_df_bottom_n]
        # y_axis_top_n=y_axis_df[x_axis_df_top_n]
        # y_axis_middle=y_axis_df[x_axis_df_middle]

        x_axis_df_sorted=self.df[x_axis[0]][x_axis[1]].sort_values()
        y_axis_df=self.df[y_axis[0]][y_axis[1]]
        x_axis_df_idx_bottom_n=x_axis_df_sorted[:n].index
        x_axis_df_idx_top_n=x_axis_df_sorted[-n:].index
        x_axis_df_idx_middle=x_axis_df_sorted[n:-n].index

        y_axis_bottom_n=y_axis_df[x_axis_df_idx_bottom_n]
        y_axis_top_n=y_axis_df[x_axis_df_idx_top_n]
        y_axis_middle=y_axis_df[x_axis_df_idx_middle]
        
        plt.boxplot([y_axis_bottom_n,y_axis_middle,y_axis_top_n],showcaps=False,showfliers=False)
        plt.xticks([1,2,3],["Bottom %d drugs"%(n),"Other drugs","Top %d drugs"%(n)])
        if axis_label:
            plt.ylabel(y_axis)
        plt.grid()
    
    def box_plot_experimental_sets(self,drug_set,y_axis=("prediction",0),axis_label=True):
        assert y_axis[0] in self.df or y_axis[1] in self.df[y_axis[0]].columns

        y_axis_df=self.df[y_axis[0]][y_axis[1]]
        y_axis_in_set=y_axis_df[drug_set]
        y_axis_not_in_set=y_axis_df[~np.isin(y_axis_df.index,drug_set)]
        plt.boxplot([y_axis_in_set,y_axis_not_in_set],showcaps=False,showfliers=False)
        plt.xticks([1,2],["In","Not In"])
        if axis_label:
            plt.ylabel(y_axis)
        plt.grid()


