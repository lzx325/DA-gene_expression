
import random
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import collections
import pickle as pkl
from torchvision import datasets
from torchvision import transforms
from pprint import pprint
from sklearn.metrics import confusion_matrix,roc_auc_score

import test
import data

class DATrainer(object):
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
        self.clip_value=1
        self.device=next(self.model.parameters()).device
        self.batch_size=128
        self.n_epochs=200
        self.alpha = 1
        self.domain_adv_coeff = 1e-1
        self.ddc_coeff= 1e-2
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
                # t_img=s_img
                # t_domain_label=s_domain_label
                self.optimizer.zero_grad()

                s_img=s_img.to(self.device)
                s_label=s_label.to(self.device)
                s_domain_label=s_domain_label.to(self.device)
                
                t_img=t_img.to(self.device)
                t_domain_label=t_domain_label.to(self.device)
                
                img=torch.cat([s_img,t_img],0)

                class_output,domain_output,ddc_features=self.model(img,alpha=self.alpha,return_ddc_features=self.ddc_features)
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

                err = self.domain_adv_coeff * (err_t_domain + err_s_domain) + self.ddc_coeff*(err_ddc) + err_s_label
                err.backward()
                clip_grad_norm_(self.model.parameters(),self.clip_value)
                self.optimizer.step()

                cumulative_metrics["domain_loss_t_domain"]+=self.domain_adv_coeff*(err_t_domain.cpu().item()/len_dataloader)
                cumulative_metrics["domain_loss_s_domain"]+=self.domain_adv_coeff*(err_s_domain.cpu().item()/len_dataloader)
                cumulative_metrics["class_loss_s_domain"]+=err_s_label.cpu().item()/len_dataloader
                cumulative_metrics["ddc"]+=self.ddc_coeff*err_ddc.cpu().item()/len_dataloader
                cumulative_metrics["loss"]+=err.cpu().item()/len_dataloader

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
                score_fp='{}/{}_{}-score.pkl'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)

                with open(score_fp,'wb') as f:
                    pkl.dump(scores_dict,f)

                current_model_fp='{}/{}_{}-model-epoch_current.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                torch.save(self.model, current_model_fp)
                

                accu_s=score_source_train["class_accuracy"]
                if Y_target is not None:
                    accu_t=score_target["class_accuracy"]
                if (Y_target is not None and accu_t > best_accu_t) or (Y_target is None and accu_s>best_accu_s):
                    best_accu_s = accu_s
                    if Y_target is not None:
                        best_accu_t = accu_t
                    best_model_fp='{}/{}_{}-model-epoch_best.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                    torch.save(self.model, best_model_fp)
            
    def predict(self,X,batch_size=None,return_score=True):
        if batch_size is None:
            batch_size=self.batch_size
        # TODO
        self.model.eval()
        X_tensor=torch.FloatTensor(X)
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        class_output_list=list()
        domain_output_list=list()
        device=next(iter(self.model.parameters())).device
        with torch.no_grad():
            for _,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                class_output,domain_output=self.model(X_batch,alpha=self.alpha,return_ddc_features=None)
                if return_score:
                    class_output=class_output
                else:
                    class_output=class_output.argmax(1)
                class_output_list.append(class_output)
                domain_output_list.append(domain_output)

        class_output_tensor=torch.cat(class_output_list,0)
        domain_output_tensor=torch.cat(domain_output_list,0)
        class_output_arr=class_output_tensor.cpu().numpy()
        domain_output_arr=domain_output_tensor.cpu().numpy()
        self.model.train()
        return class_output_arr,domain_output_arr

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

            class_output_arr,domain_output_arr=self.predict(X,return_score=True)
            class_output_tensor=torch.FloatTensor(class_output_arr)
            Y_tensor=torch.LongTensor(Y)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            class_loss=F.cross_entropy(class_output_tensor,Y_tensor,reduction="mean")
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
                }
            else:
                cf=confusion_matrix(Y,class_output_idx_arr)
                return {
                    'class_accuracy':class_acc,
                    'class_loss':class_loss,
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'confusion_matrix':cf,
                }
        else:
            _,domain_output_arr=self.predict(X,return_score=True)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            domain_loss=F.cross_entropy(domain_output_tensor,domain_labels_tensor,reduction="mean")
            domain_output_idx_arr=domain_output_tensor.argmax(1).numpy()
            domain_acc=np.mean(domain_output_idx_arr==domain_labels)
            return {
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
            }

def fit(mynet,ds_dict,loss_fn_dict,params):
    best_accu_t = 0.0
    n_epoch,source_dataset_name,target_dataset_name=params["n_epoch"],params["source_dataset_name"],params["target_dataset_name"]
    X_source,Y_source,X_target,Y_target=ds_dict["X_source"],ds_dict["Y_source"],ds_dict["X_target"],ds_dict["Y_target"]
    n_epoch,lr,batch_size=params["n_epoch"],params["lr"],params["batch_size"]
    loss_class=loss_fn_dict["loss_class"]
    loss_domain=loss_fn_dict["loss_domain"]
    device=next(mynet.parameters()).device
    optimizer = optim.Adam(mynet.parameters(), lr=lr)
    model_root="./models"

    if np.min(Y_source)==1:
        Y_source=Y_source-1
    assert np.min(Y_source)==0 
    
    if Y_target is not None and np.min(Y_target)==1:
        Y_target==Y_target-1
        np.min(Y_target)==0

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


    dataloader_source=torch.utils.data.DataLoader(ds_source,batch_size=batch_size,shuffle=True)
    dataloader_target=torch.utils.data.DataLoader(ds_target,batch_size=batch_size,shuffle=True)

    best_accu_s=0.0
    best_accu_t=0.0

    scores_dict=collections.defaultdict(list)

    for epoch in range(n_epoch):
        # TODO
        mynet.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        err_t_domain_cumulative=0.0
        err_s_domain_cumulative=0.0
        err_s_label_cumulative=0.0
        err_cumulative=0.0
        for i in range(len_dataloader):
            
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha=1
            domain_adv_coeff=1
            ddc_coeff=1
            ddc_features="c_fc2"
            # training mynet using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source

            mynet.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            s_img=s_img.to(device)
            s_label=s_label.to(device)
            domain_label=domain_label.to(device)

            class_output, domain_output, source_domain_ddc_features = mynet(input_data=s_img, alpha=alpha,return_ddc_features=ddc_features)

            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training mynet using target data
            data_target = data_target_iter.next()
            if Y_target is not None:
                t_img, _ = data_target
            else:
                t_img, = data_target


            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            
            t_img = t_img.to(device)
            domain_label = domain_label.to(device)

            _, domain_output, target_domain_ddc_features = mynet(input_data=t_img, alpha=alpha, return_ddc_features=ddc_features)
            
            err_t_domain = loss_domain(domain_output, domain_label)

            def loss_ddc(f_of_X, f_of_Y):
                bs1=f_of_X.shape[0]
                bs2=f_of_Y.shape[0]
                bs=min(bs1,bs2)
                delta = f_of_X[:bs,:] - f_of_Y[:bs,:]
                loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
                return loss
            err_ddc=loss_ddc(source_domain_ddc_features,target_domain_ddc_features)

            err = domain_adv_coeff * (err_t_domain + err_s_domain) + ddc_coeff*(err_ddc) + err_s_label
            err.backward()
            optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, alpha: %f, domain_adv_coeff: %f, ddc_coeff: %f' \
                % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                    err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item(),alpha,domain_adv_coeff,ddc_coeff),
                    )
            sys.stdout.flush()
            current_model_fp='{}/{}_{}-model-epoch_current.pth'.format(model_root,source_dataset_name,target_dataset_name)
            torch.save(mynet, current_model_fp)
            err_t_domain_cumulative+=err_t_domain.cpu().item()/len_dataloader
            err_s_domain_cumulative+=err_s_domain.cpu().item()/len_dataloader
            err_s_label_cumulative+=err_s_label.cpu().item()/len_dataloader
            err_cumulative+=err/len_dataloader

        
        print('\n')
        score = test.score(mynet,X_source,Y_source)
        accu_s=score["accuracy"]
        print('Accuracy of the %s dataset: %f' % (source_dataset_name, accu_s))
        print("confusion matrix:")
        print(score["confusion_matrix"])
        scores_dict["source_domain_accuracy"].append(accu_s)
        scores_dict["source_domain_label_loss"].append(err_s_label_cumulative)
        scores_dict["source_domain_adv_loss"].append(err_s_domain_cumulative)
        scores_dict["target_domain_adv_loss"].append(err_t_domain_cumulative)
        scores_dict["source_domain_confusion_matrix"].append(score["confusion_matrix"])

        print()
        if Y_target is not None:
            score = test.score(mynet,X_target,Y_target)
            accu_t=score["accuracy"]
            print('Accuracy of the %s dataset: %f' % (target_dataset_name, accu_t))
            print("confusion matrix:")
            print(score["confusion_matrix"])
            scores_dict["target_domain_accuracy"].append(accu_t)
            scores_dict["target_domain_confusion_matrix"].append(score["confusion_matrix"])
        print()
        score_fp='{}/{}_{}-score.pkl'.format(model_root,source_dataset_name,target_dataset_name)

        with open(score_fp,'wb') as f:
            pkl.dump(scores_dict,f)

        if (Y_target is not None and accu_t > best_accu_t) or (Y_target is None and accu_s>best_accu_s):
            best_accu_s = accu_s
            if Y_target is not None:
                best_accu_t = accu_t
            best_model_fp='{}/{}_{}-model-epoch_best.pth'.format(model_root,source_dataset_name,target_dataset_name)
            torch.save(mynet, best_model_fp)

    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % (source_dataset_name, best_accu_s))
    print('Accuracy of the %s dataset: %f' % (target_dataset_name, best_accu_t))
