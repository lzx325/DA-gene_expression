import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import numpy as np
from scipy.special import softmax
from scipy.sparse import csc_matrix
from sklearn.metrics import roc_auc_score,confusion_matrix
import collections
import pickle as pkl
from pprint import pprint


from my_tabnet_layers import EmbeddingGenerator,TabNetNoEmbeddings
from pytorch_tabnet.utils import PredictDataset, filter_weights, create_explain_matrix
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from pytorch_tabnet import sparsemax
from functions import ReverseLayerF

class TabNetDA(torch.nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, n_d=8, n_a=8,
                 n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02, device_name='auto',
                 mask_type="sparsemax"):
        super().__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.intermediate_dim=intermediate_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independant can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        # self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        # self.post_embed_dim = self.embedder.post_embed_dim

        self.tabnet = TabNetNoEmbeddings(self.input_dim, self.intermediate_dim, n_d, n_a, n_steps,
                                         gamma, n_independent, n_shared, epsilon,
                                         virtual_batch_size, momentum, mask_type)
        

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.intermediate_dim,15))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(15))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(15, 10))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(10))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(10, self.output_dim))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.intermediate_dim,15))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(15))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(15, 2))


    def forward(self, input_data, alpha=1.0, return_ddc_features=None):
        if return_ddc_features is not None:
            assert return_ddc_features in self.class_classifier._modules or return_ddc_features=="tabnet_features"
        # feature = self.embedder(input_data)
        feature, M_loss=self.tabnet(input_data)
            
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output=feature
        ddc_features=feature
        for k,v in self.class_classifier._modules.items():
            class_output=v(class_output)
            if k==return_ddc_features:
                ddc_features=class_output
        domain_output = self.domain_classifier(reverse_feature)

        if return_ddc_features:
            return class_output,domain_output,ddc_features,M_loss
        else:
            return class_output,domain_output,M_loss

        # feature = self.embedder(input_data)
        # feature, M_loss=self.tabnet(feature)
        # return feature,None,M_loss

    def forward_masks(self, x):
        # x = self.embedder(x)
        return self.tabnet.forward_masks(x)

class TabNetDATrainer(object):
    def __init__(
        self,
        model,
        optimizer_fn,
        optimizer_params,
        source_dataset_name,
        target_dataset_name
        ):
        self.model=model
        self.optimizer=optimizer_fn(self.model.parameters(),**optimizer_params)
        self.lambda_sparse=1e-3
        self.clip_value=1
        self.device=next(self.model.parameters()).device
        self.batch_size=1024
        self.n_epochs=200
        self.alpha = 1
        self.domain_adv_coeff = 0
        self.ddc_coeff= 0
        self.ddc_features="c_fc2"
        self.source_dataset_name=source_dataset_name
        self.target_dataset_name=target_dataset_name
        self.model_root="./models"

    def fit(
        self,
        X_source,
        Y_source,
        X_target,
        Y_target,
        X_source_val=None,
        Y_source_val=None,
        n_epochs=None,
        save=True
    ):
        if n_epochs is None:
            n_epochs=self.n_epochs
        if np.min(Y_source)==1:
            Y_source=Y_source-1
        if np.min(Y_target)==1:
            Y_target=Y_target-1
        if Y_source_val is not None and np.min(Y_source_val)==1:
            Y_source_val=Y_source_val-1
        assert np.min(Y_source)==0 and (Y_target is None or np.min(Y_target)==0) and (Y_source_val is None or np.min(Y_source_val)==0)

        X_source_tensor=torch.FloatTensor(X_source)
        Y_source_tensor=torch.LongTensor(Y_source)
        X_target_tensor=torch.FloatTensor(X_target)

        
        if Y_target is not None:
            Y_target_tensor=torch.LongTensor(Y_target)
        
        ds_source=torch.utils.data.TensorDataset(X_source_tensor,Y_source_tensor)
        if Y_target is not None:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor,Y_target_tensor)
        else:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor)

        dataloader_source=torch.utils.data.DataLoader(ds_source,batch_size=self.batch_size,shuffle=True)
        dataloader_target=torch.utils.data.DataLoader(ds_target,batch_size=self.batch_size,shuffle=True)
        print("Length of dataloaders:")
        print(len(dataloader_source), len(dataloader_target))
        print("Parameters:")
        print("alpha=%.4f,domain_adv_coeff=%.4f,ddc_coeff=%.4f,ddc_features=%s"%(self.alpha,self.domain_adv_coeff,self.ddc_coeff,self.ddc_features))

        best_accu_s=0.0
        best_accu_t=0.0
        scores_dict=collections.defaultdict(list)

        print("before everything")
        source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
        score = self.score(X_source,Y_source,"source")
        accu_s=score["class_accuracy"]
        print('Accuracy of the %s dataset: %f' % (self.source_dataset_name, accu_s))
        print("confusion matrix:")
        print(score["confusion_matrix"])


        for epoch in range(n_epochs):
            self.model.train()
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)
            cumulative_metrics=collections.defaultdict(float)
            for i in range(len_dataloader):
                data_source=data_source_iter.next()
                s_img,s_label=data_source
                s_domain_label = torch.zeros(len(s_img)).long()

                data_target = data_target_iter.next()
                if Y_target is not None:
                    t_img, _ = data_target
                else:
                    t_img, = data_target
                t_domain_label = torch.ones(len(t_img)).long()
                
                self.optimizer.zero_grad()

                s_img=s_img.to(self.device)
                s_label=s_label.to(self.device)
                s_domain_label=s_domain_label.to(self.device)
                
                t_img=t_img.to(self.device)
                t_domain_label=t_domain_label.to(self.device)
                
                img=torch.cat([s_img,t_img],0)

                class_output,domain_output,ddc_features,M_loss=self.model(img,alpha=self.alpha,return_ddc_features=self.ddc_features)
                s_class_output=class_output[:len(s_img)]
                s_domain_output=domain_output[:len(s_img)]
                t_domain_output=domain_output[len(s_img):]
                s_ddc_features=ddc_features[:len(s_img)]
                t_ddc_features=ddc_features[len(s_img):]

                err_s_label=F.cross_entropy(s_class_output,s_label)
                err_s_domain=F.cross_entropy(s_domain_output,s_domain_label)

                err_t_domain=F.cross_entropy(t_domain_output,t_domain_label)
                
                def loss_ddc(f_of_X, f_of_Y):
                    bs1=f_of_X.shape[0]
                    bs2=f_of_Y.shape[0]
                    bs=min(bs1,bs2)
                    delta = f_of_X[:bs,:] - f_of_Y[:bs,:]
                    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
                    return loss

                err_ddc=loss_ddc(s_ddc_features,t_ddc_features)

                err = self.domain_adv_coeff * (err_t_domain + err_s_domain) + self.ddc_coeff*(err_ddc) + err_s_label - self.lambda_sparse*M_loss
                err.backward()
                clip_grad_norm_(self.model.parameters(),self.clip_value)
                self.optimizer.step()

                cumulative_metrics["domain_loss_t_domain"]+=self.domain_adv_coeff*(err_t_domain.cpu().item()/len_dataloader)
                cumulative_metrics["domain_loss_s_domain"]+=self.domain_adv_coeff*(err_s_domain.cpu().item()/len_dataloader)
                cumulative_metrics["class_loss_s_domain"]+=err_s_label.cpu().item()/len_dataloader
                cumulative_metrics["ddc"]+=self.ddc_coeff*err_ddc.cpu().item()/len_dataloader
                cumulative_metrics["M_loss"]+=-self.lambda_sparse*M_loss.cpu().item()/len_dataloader
                cumulative_metrics["loss"]+=err.cpu().item()/len_dataloader

                # with torch.no_grad():
                #     X_arr=X_source[:5]
                #     Y_arr=Y_source[:5]
                #     X_tensor=torch.FloatTensor(X_arr).to(self.device)
                #     Y_tensor=torch.LongTensor(Y_arr).to(self.device)
                #     class_output,domain_output,ddc_features,M_loss=self.model(X_tensor,alpha=self.alpha,return_ddc_features=self.ddc_features)
                #     print(F.cross_entropy(class_output,Y_tensor,reduction="none"))

            print()
            print("Epoch %d"%(epoch+1))

            pprint(cumulative_metrics)
            print("On source train set: ")
            source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
            score_source_train = self.score(X_source,Y_source,"source")
            for k,v in score_source_train.items():
                print(f"\t{k}:\n\t{v}")
                scores_dict[k+"_source_train"].append(v)
            print()
            print("On source val set")
            if X_source_val is not None:
                source_domain_labels=np.zeros((len(X_source_val),),dtype=np.int64)
                score_source_val = self.score(X_source_val,Y_source_val,"source")
                for k,v in score_source_val.items():
                    print(f"\t{k}:\n\t{v}")
                    scores_dict[k+"_source_val"].append(v)

            print()
            print("On target set")
            target_domain_labels=np.ones((len(X_target),),dtype=np.int64)
            score_target = self.score(X_target,Y_target,"target")
            for k,v in score_target.items():
                print(f"\t{k}:\n{v}")
                scores_dict[k+"_target"].append(v)

            if save:
                score_fp='{}/tabnet-{}_{}-score.pkl'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)

                with open(score_fp,'wb') as f:
                    pkl.dump(scores_dict,f)

                current_model_fp='{}/tabnet-{}_{}-model-epoch_current.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                torch.save(self.model, current_model_fp)

                accu_s=score_source_train["class_accuracy"]
                if Y_target is not None:
                    accu_t=score_target["class_accuracy"]
                if (Y_target is not None and accu_t > best_accu_t) or (Y_target is None and accu_s>best_accu_s):
                    best_accu_s = accu_s
                    if Y_target is not None:
                        best_accu_t = accu_t
                    best_model_fp='{}/tabnet-{}_{}-model-epoch_best.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                    torch.save(self.model, best_model_fp)
    '''
    def fit_simple(
        self,
        X_source,
        Y_source,
        X_target,
        Y_target
    ):
        
        if np.min(Y_source)==1:
            Y_source=Y_source-1
        if np.min(Y_target)==1:
            Y_target=Y_target-1
        assert np.min(Y_source)==0 and (Y_target is None or np.min(Y_target)==0)
        
        X_source_tensor=torch.FloatTensor(X_source)
        Y_source_tensor=torch.LongTensor(Y_source)
        X_target_tensor=torch.FloatTensor(X_target)
        if Y_target is not None:
            Y_target_tensor=torch.LongTensor(Y_target)
        
        ds_source=torch.utils.data.TensorDataset(X_source_tensor,Y_source_tensor)
        if Y_target is not None:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor,Y_target_tensor)
        else:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor)
        
        dataloader_source=torch.utils.data.DataLoader(ds_source,batch_size=self.batch_size,shuffle=True)
        dataloader_target=torch.utils.data.DataLoader(ds_target,batch_size=self.batch_size,shuffle=True)
        print("Length of dataloaders")
        print(len(dataloader_source), len(dataloader_target))
        print("alpha=%.4f,domain_adv_coeff=%.4f,ddc_coeff=%.4f,ddc_features=%s"%(self.alpha,self.domain_adv_coeff,self.ddc_coeff,self.ddc_features))

        best_accu_s=0.0
        best_accu_t=0.0
        scores_dict=collections.defaultdict(list)

        print("before everything")
        source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
        score = self.score(X_source,Y_source,"source")
        accu_s=score["class_accuracy"]
        print('Accuracy of the %s dataset: %f' % (self.source_dataset_name, accu_s))
        print("confusion matrix:")
        print(score["confusion_matrix"])

        for epoch in range(self.n_epochs):
            self.model.train()
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)

            err_t_domain_cumulative=0.0
            err_s_domain_cumulative=0.0
            err_s_label_cumulative=0.0
            err_cumulative=0.0
            err_tabnet_cumulative=0.0

            for i in range(len_dataloader):
                data_source=data_source_iter.next()
                s_img,s_label=data_source

                self.optimizer.zero_grad()

                domain_label = torch.zeros(len(s_img)).long()
                s_img=s_img.to(self.device)
                s_label=s_label.to(self.device)
                domain_label=domain_label.to(self.device)
                class_output,_,M_loss=self.model(s_img,'source',alpha=0)
                err_s_label=F.cross_entropy(class_output,s_label)
         
                err = err_s_label - self.lambda_sparse*M_loss
                err.backward()
                clip_grad_norm_(self.model.parameters(),self.clip_value)
                self.optimizer.step()

                err_s_label_cumulative+=err_s_label.cpu().item()*len(s_img)/(len(X_source))
                err_tabnet_cumulative+=err.item()*len(s_img)/(len(X_source))
                err_cumulative+=err.item()*len(s_img)/(len(X_source))

            print('\n')
            print(f"label loss: {err_s_label_cumulative:.4f},\
             Tabnet loss: {err_tabnet_cumulative:.4f},\
             source domain adv_loss: {err_s_domain_cumulative:.4f},\
             target domain adv_loss: {err_t_domain_cumulative:.4f}")
            
            source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
            score = self.score(X_source,Y_source)
            accu_s=score["class_accuracy"]
            print('Accuracy of the %s dataset: %f' % (self.source_dataset_name, accu_s))
            print("confusion matrix:")
            print(score["confusion_matrix"])
            scores_dict["source_domain_accuracy"].append(accu_s)
            scores_dict["source_domain_label_loss"].append(err_s_label_cumulative)
            scores_dict["source_domain_adv_loss"].append(err_s_domain_cumulative)
            scores_dict["target_domain_adv_loss"].append(err_t_domain_cumulative)
            scores_dict["source_domain_confusion_matrix"].append(score["confusion_matrix"])

            print()
    '''
    def predict(self,X,batch_size=None,return_score=True):
        if batch_size is None:
            batch_size=self.batch_size
        self.model.eval()
        try:
            X_tensor=torch.FloatTensor(X)
            dataset=torch.utils.data.TensorDataset(X_tensor)
            loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
            class_output_list=list()
            domain_output_list=list()
            M_loss_cumulative=0.0
            device=next(iter(self.model.parameters())).device
            with torch.no_grad():
                for _,(X_batch,) in enumerate(loader):
                    X_batch=X_batch.to(device)
                    class_output,domain_output,M_loss=self.model(X_batch,alpha=self.alpha,return_ddc_features=None)
                    if return_score:
                        class_output=class_output
                    else:
                        class_output=class_output.argmax(1)
                    class_output_list.append(class_output)
                    domain_output_list.append(domain_output)
                    M_loss_cumulative+=M_loss.item()*len(X_batch)/len(X)

            class_output_tensor=torch.cat(class_output_list,0)
            domain_output_tensor=torch.cat(domain_output_list,0)
            class_output_arr=class_output_tensor.cpu().numpy()
            domain_output_arr=domain_output_tensor.cpu().numpy()
            return class_output_arr,domain_output_arr,M_loss
        finally:
            self.model.train()
    def transform(self,X,batch_size=None,layer="c_fc2"):
        print(layer)
        if batch_size is None:
            batch_size=self.batch_size
        # TODO
        self.model.eval()
        try:
            X_tensor=torch.FloatTensor(X)
            dataset=torch.utils.data.TensorDataset(X_tensor)
            loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
            feature_list=list()
            device=next(iter(self.model.parameters())).device
            with torch.no_grad():
                for _,(X_batch,) in enumerate(loader):
                    X_batch=X_batch.to(device)
                    _,_,features,_=self.model(X_batch,alpha=self.alpha,return_ddc_features=layer)
                    feature_list.append(features)

            feature_tensor=torch.cat(feature_list,0)
            feature_arr=feature_tensor.cpu().numpy()
            return feature_arr
        finally:
            self.model.train()
    def score(self,X,Y,domain):
        if domain=="source":
            domain_labels=np.zeros((len(X),),dtype=np.int64)
        elif domain=="target":
            domain_labels=np.ones((len(X),),dtype=np.int64)
        if Y is not None:
            if np.min(Y)==1:
                Y=Y-1
            elif np.min(Y)!=0:
                assert False

            class_output_arr,domain_output_arr,M_loss=self.predict(X,return_score=True)
            class_output_tensor=torch.FloatTensor(class_output_arr)
            Y_tensor=torch.LongTensor(Y)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            class_loss=F.cross_entropy(class_output_tensor,Y_tensor,reduction="mean")
            # if domain=="source":
            #     with torch.no_grad():
            #         X_arr=X[:5]
            #         Y_arr=Y[:5]
            #         X_tensor=torch.FloatTensor(X_arr).to(self.device)
            #         Y_tensor=torch.LongTensor(Y_arr).to(self.device)
            #         class_output,domain_output,ddc_features,M_loss=self.model(X_tensor,alpha=self.alpha,return_ddc_features=self.ddc_features)
            #         print(F.cross_entropy(class_output,Y_tensor,reduction="none"))
            class_output_idx_arr=class_output_arr.argmax(1)
            class_acc=np.mean(class_output_idx_arr==Y)
            domain_loss=F.cross_entropy(domain_output_tensor,domain_labels_tensor,reduction="mean")
            domain_output_idx_arr=domain_output_tensor.argmax(1).numpy()
            domain_acc=np.mean(domain_output_idx_arr==domain_labels)
            
            if class_output_arr.shape[1]==2:
                auc=roc_auc_score(Y,class_output_arr[:,1])
                return {
                    'class_accuracy':class_acc,
                    'class_loss':class_loss,
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'auc':auc,
                    'M_loss':-self.lambda_sparse*M_loss
                }
            else:
                cf=confusion_matrix(Y,class_output_idx_arr)
                return {
                    'class_accuracy':class_acc,
                    'class_loss':class_loss,
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'confusion_matrix':cf,
                    "M_loss":-self.lambda_sparse*M_loss
                }
        else:
            _,domain_output_arr,M_loss=self.predict(X,return_score=True)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            domain_loss=F.cross_entropy(domain_output_tensor,domain_labels_tensor,reduction="mean")
            domain_output_idx_arr=domain_output_tensor.argmax(1).numpy()
            domain_acc=np.mean(domain_output_idx_arr==domain_labels)
            return {
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'M_loss':-self.lambda_sparse*M_loss
            }
    def explain(self, X):
        """
        Return local explanation

        Parameters
        ----------
        X : tensor: `torch.Tensor`
            Input data

        Returns
        -------
        M_explain : matrix
            Importance per sample, per columns.
        masks : matrix
            Sparse matrix showing attention masks used by network.
        """
        try:
            self.model.eval()
            
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

            res_explain = []
            reducing_matrix = create_explain_matrix(
                self.model.input_dim,
                0,
                [],
                self.model.input_dim
            )
            for batch_nb, data in enumerate(dataloader):
                data = data.to(self.device).float()

                M_explain, masks = self.model.forward_masks(data)
                for key, value in masks.items():
                    masks[key] = csc_matrix.dot(
                        value.cpu().detach().numpy(), reducing_matrix
                    )

                res_explain.append(
                    csc_matrix.dot(M_explain.cpu().detach().numpy(), reducing_matrix)
                )

                if batch_nb == 0:
                    res_masks = masks
                else:
                    for key, value in masks.items():
                        res_masks[key] = np.vstack([res_masks[key], value])

            res_explain = np.vstack(res_explain)

            return res_explain, res_masks
        finally:
            self.model.train()

    def compute_feature_importances(self, X):
        """Compute global feature importance.

        Parameters
        ----------
        loader : `torch.utils.data.Dataloader`
            Pytorch dataloader.

        """

        try:
            X_tensor=torch.FloatTensor(X)
            dataset=torch.utils.data.TensorDataset(X_tensor)
            loader=torch.utils.data.DataLoader(dataset,batch_size=self.batch_size)
            self.model.eval()
            reducing_matrix = create_explain_matrix(
                self.model.input_dim,
                0,
                [],
                self.model.input_dim
            )

            feature_importances_ = np.zeros((self.model.input_dim))
            for data,  in loader:
                data = data.to(self.device).float()
                M_explain, masks = self.model.forward_masks(data)
                feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

            feature_importances_ = csc_matrix.dot(
                feature_importances_, reducing_matrix
            )
            self.feature_importances_ = feature_importances_ / np.sum(feature_importances_)
        finally:
            self.model.train()
class TabNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8,
                 n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02, device_name='auto',
                 mask_type="sparsemax"):
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independant can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(self.post_embed_dim, self.output_dim, n_d, n_a, n_steps,
                                         gamma, n_independent, n_shared, epsilon,
                                         virtual_batch_size, momentum, mask_type)

    def forward(self, input_data, alpha=1.0, return_ddc_features=None):
        feature = self.embedder(input_data)
        feature, M_loss=self.tabnet(feature)
        
        return feature,M_loss

    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)

class TabNetTrainer(object):
    def __init__(
        self,
        model,
        optimizer_fn,
        optimizer_params,
        source_dataset_name,
        target_dataset_name
        ):
        self.model=model
        self.optimizer=optimizer_fn(self.model.parameters(),**optimizer_params)
        self.lambda_sparse=1e-3
        self.clip_value=1
        self.device=next(self.model.parameters()).device
        self.batch_size=128
        self.n_epochs=200
        self.alpha = 1
        self.domain_adv_coeff=0
        self.ddc_coeff=0
        self.ddc_features="c_fc2"
        self.source_dataset_name=source_dataset_name
        self.target_dataset_name=target_dataset_name
        self.model_root="./models"

    def fit(
        self,
        X_train,
        y_train,
        val_data=None,
        n_epochs=200
    ):

        X_val,y_val=val_data
        if np.min(y_train)==1:
            y_train=y_train-1
        else:
            if np.min(y_train)!=0:
                assert False
        if np.min(y_val)==1:
            y_val=y_val-1
        else:
            if np.min(y_val)!=0:
                assert False
        X_train_tensor=torch.FloatTensor(X_train)
        y_train_tensor=torch.LongTensor(y_train)

        ds_train=torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
        
        dataloader_train=torch.utils.data.DataLoader(ds_train,batch_size=self.batch_size,shuffle=True)
        
        once=False
        for epoch in range(n_epochs):
            self.model.train()
            epoch_train_loss=0.0
            for (X_batch,y_batch) in dataloader_train:
                X_batch=X_batch.to(self.device)
                y_batch=y_batch.to(self.device)

                # for param in self.model.parameters():
                #     param.grad = None
                class_output,M_loss=self.model(X_batch)

                loss=F.cross_entropy(class_output,y_batch)
                loss = loss - self.lambda_sparse*M_loss
                self.optimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(self.model.parameters(),self.clip_value)
                self.optimizer.step()
                epoch_train_loss+=loss.item()*len(X_batch)/len(X_train)
            
            eval_vals=self.score(X_val,y_val,epoch)
            print("On epoch %d, train_loss=%.4f, val_loss= %.4f, val_M_loss= %.4f, val_acc= %.4f"%(
                epoch+1,
                epoch_train_loss,
                eval_vals["loss"],
                eval_vals["M_loss"],
                eval_vals["accuracy"]
            ))
            if self.model.output_dim>2:
                print(eval_vals["confusion_matrix"])
            else:
                print(eval_vals["auc"])
        

    def score(self,X,Y,epoch):
        if np.min(Y)==1:
            Y=Y-1
        elif np.min(Y)!=0:
            assert False
        predictions_arr,M_loss=self.predict(X,return_score=True,return_M_loss=True)
        
        predictions_tensor=torch.FloatTensor(predictions_arr)
        Y_tensor=torch.LongTensor(Y)
        class_loss=F.cross_entropy(predictions_tensor,Y_tensor,reduction="mean")
        
        M_loss=-self.lambda_sparse*M_loss
        loss=class_loss+M_loss
        predictions_idx_arr=predictions_arr.argmax(1)
        acc=np.mean(predictions_idx_arr==Y)
        if predictions_arr.shape[1]==2:
            auc=roc_auc_score(Y,predictions_arr[:,1])
            return {
                'accuracy':acc,
                'loss':loss,
                'M_loss':M_loss,
                'auc':auc
            }
        else:
            cf=confusion_matrix(Y,predictions_idx_arr)
            return {
                'accuracy':acc,
                'loss':loss,
                'confusion_matrix':cf,
                'M_loss':M_loss
            }

    def predict(self,X,batch_size=None,return_score=True,return_M_loss=False):
        if batch_size is None:
            batch_size=self.batch_size
        self.model.eval()
        X_tensor=torch.FloatTensor(X)
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        pred_list=list()
        M_loss_cumulative=0.0
        with torch.no_grad():
            for _,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(self.device)
                pred,M_loss=self.model(X_batch)
                if return_score:
                    pred=pred
                else:
                    pred=pred.argmax(1)
                pred_list.append(pred)
                M_loss_cumulative+=M_loss.item()*(len(X_batch)/len(X))

        predictions_tensor=torch.cat(pred_list,0)
        
        predictions_arr=predictions_tensor.cpu().numpy()
        self.model.train()
        if return_M_loss:
            return predictions_arr,M_loss_cumulative
        else:
            return predictions_arr
    
class _TabNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8,
                 n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02, device_name='auto',
                 mask_type="sparsemax"):
        super(_TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independant can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(self.post_embed_dim, output_dim, n_d, n_a, n_steps,
                                         gamma, n_independent, n_shared, epsilon,
                                         virtual_batch_size, momentum, mask_type)

        # Defining device
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)
        self.to(self.device)

    def forward(self, x):
        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)

class _TabNetTrainer(object):
    def __init__(
        self,
        model,
        optimizer_fn,
        optimizer_params,
        ):
        self.model=model
        self.optimizer=optimizer_fn(self.model.parameters(),**optimizer_params)
        self.lambda_sparse=1e-3
        self.clip_value=1
        self.device=next(self.model.parameters()).device
    def fit(
        self,
        X_train,
        y_train,
        val_data=None
    ):
        batch_size=128
        n_epochs=200
        X_val,y_val=val_data
        if np.min(y_train)==1:
            y_train=y_train-1
        if np.min(y_val)==1:
            y_val=y_val-1

        X_train_tensor=torch.FloatTensor(X_train)
        y_train_tensor=torch.LongTensor(y_train)

        ds_train=torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
        
        dataloader_train=torch.utils.data.DataLoader(ds_train,batch_size=batch_size)

        for epoch in range(n_epochs):
            self.model.train()
            epoch_train_loss=0.0
            for (X_batch,y_batch) in dataloader_train:
                X_batch=X_batch.to(self.device)
                y_batch=y_batch.to(self.device)
                output,M_loss=self.model(X_batch)
                loss=F.cross_entropy(output,y_batch)
                loss = loss - self.lambda_sparse*M_loss
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(),self.clip_value)
                self.optimizer.step()
                epoch_train_loss+=loss.item()*len(X_batch)/len(X_train)
            eval_vals=self.evaluate(X_val,y_val)
            print("On epoch %d, train_loss=%.4f, val_loss= %.4f, val_acc= %.4f"%(
                epoch+1,
                epoch_train_loss,
                eval_vals["loss"],
                eval_vals["accuracy"]
            ))
    def predict(self,X,batch_size=128,return_score=True):
        self.model.eval()
        X_tensor=torch.FloatTensor(X)
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        pred_list=list()
        device=next(iter(self.model.parameters())).device
        with torch.no_grad():
            for _,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                pred,_=self.model(X_batch)
                if return_score:
                    pred=torch.softmax(pred,dim=1)
                else:
                    pred=pred.argmax(1)
                pred_list.append(pred)

        predictions_tensor=torch.cat(pred_list,0)
        predictions_arr=predictions_tensor.cpu().numpy()
        self.model.train()
        return predictions_arr

    def evaluate(self,X,Y):
        if np.min(Y)==1:
            Y=Y-1
        elif np.min(Y)!=0:
            assert False
        predictions_arr=self.predict(X,return_score=True)
        predictions_tensor=torch.FloatTensor(predictions_arr)
        Y_tensor=torch.LongTensor(Y)
        loss=F.cross_entropy(predictions_tensor,Y_tensor,reduction="mean")
        predictions_idx_arr=predictions_arr.argmax(1)
        acc=np.mean(predictions_idx_arr==Y)
        if predictions_arr.shape[1]==2:
            auc=roc_auc_score(Y,predictions_arr[:,1])
            return {
                'accuracy':acc,
                'loss':loss,
                'auc':auc
            }
        else:
            return {
                'accuracy':acc,
                'loss':loss
            }