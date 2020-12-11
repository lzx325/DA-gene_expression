import torch.nn as nn
import torch.nn.functional as F
import torch
from functions import ReverseLayerF

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
        
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
        
class CNNModel1(nn.Module):

    def __init__(self):
        super(CNNModel1, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

    def forward(self, input_data, alpha, return_ddc_features=None):
        if return_ddc_features is not None:
            assert return_ddc_features in self.class_classifier._modules
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        # class_output = self.class_classifier(feature)
        class_output=feature
        ddc_features=None
        for k,v in self.class_classifier._modules.items():
            class_output=v(class_output)
            if k==return_ddc_features:
                ddc_features=class_output
        domain_output = self.domain_classifier(reverse_feature)
        if return_ddc_features:
            return class_output,domain_output,ddc_features
        else:
            return class_output, domain_output
    def transform(self,X,layer="fc",batch_size=128):
        assert layer in ("feature","fc")
        device=next(self.parameters()).device
        X_tensor=torch.from_numpy(X)
        
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        self.eval()
        X_transformed_tensor_list=list()
        def fc_transformation(fc,val):
            module_list=list(fc._modules.values())[:6]
            for m in module_list:
                val=m(val)
            return val
        with torch.no_grad():
            for i,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                X_batch_transformed_tensor=self.feature(X_batch)
                X_batch_transformed_tensor=X_batch_transformed_tensor.view(X_batch.shape[0],-1)
                if layer=="fc":
                    X_batch_transformed_tensor=fc_transformation(self.class_classifier,X_batch_transformed_tensor)
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor=torch.cat(X_transformed_tensor_list,dim=0)
        X_transformed_arr=X_transformed_tensor.cpu().numpy()
        return X_transformed_arr

class CNNModel2(nn.Module):

    def __init__(self):
        super(CNNModel2, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_flatten', Flatten())
        self.feature.add_module('f2_fc1', nn.Linear(50 * 4 * 4, 100))
        self.feature.add_module('f2_bn1', nn.BatchNorm1d(100))
        self.feature.add_module('f2_relu1', nn.ReLU(True))
        self.feature.add_module('f2_drop1', nn.Dropout())
        self.feature.add_module('f2_fc2', nn.Linear(100, 100))
        self.feature.add_module('f2_bn2', nn.BatchNorm1d(100))
        self.feature.add_module('f2_relu2', nn.ReLU(True))
        self.class_classifier = nn.Sequential()
        
        self.class_classifier.add_module('c_fc1', nn.Linear(100, 50))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(50))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(50,10))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(100, 50))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(50))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(50,10))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

class CNNModel3(nn.Module):
    def __init__(self):
        super(CNNModel3, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_flatten', Flatten())
        self.feature.add_module('f2_fc1', nn.Linear(50 * 4 * 4, 100))
        self.feature.add_module('f2_bn1', nn.BatchNorm1d(100))
        self.feature.add_module('f2_relu1', nn.ReLU(True))
        self.feature.add_module('f2_drop1', nn.Dropout())
        self.feature.add_module('f2_fc2', nn.Linear(100, 100))
        self.feature.add_module('f2_bn2', nn.BatchNorm1d(100))
        self.feature.add_module('f2_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(100, 80))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(80))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(80,40))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(40))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(40,10))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(100, 80))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(80))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(80,40))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(40))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(40,10))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
    def transform(self,X,layer="fc",batch_size=128):
        assert layer in ("feature","fc")
        device=next(self.parameters()).device
        X_tensor=torch.from_numpy(X)
        
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        self.eval()
        X_transformed_tensor_list=list()
        def fc_transformation(fc,val):
            module_list=list(fc._modules.values())[:7]
            for m in module_list:
                val=m(val)
            return val
        with torch.no_grad():
            for i,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                X_batch_transformed_tensor=self.feature(X_batch)
                if layer=="fc":
                    X_batch_transformed_tensor=fc_transformation(self.class_classifier,X_batch_transformed_tensor)
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor=torch.cat(X_transformed_tensor_list,dim=0)
        X_transformed_arr=X_transformed_tensor.cpu().numpy()
        return X_transformed_arr

class FCModel(nn.Module):
    def __init__(self,n_in_features,n_out_classes=4):
        super().__init__()
        self.n_in_features=n_in_features
        self.n_out_classes=n_out_classes
        self.feature = nn.Sequential()
        self.feature.add_module("f_fc1",nn.Linear(n_in_features,100))
        self.feature.add_module("f_bn1",nn.BatchNorm1d(100))
        self.feature.add_module("f_relu1",nn.ReLU(True))
        self.feature.add_module("f_drop1",nn.Dropout())
        self.feature.add_module("f_fc2",nn.Linear(100,80))
        self.feature.add_module("f_bn2",nn.BatchNorm1d(80))
        self.feature.add_module("f_relu2",nn.ReLU(True))
        self.feature.add_module("f_fc3",nn.Linear(80,60))
        self.feature.add_module("f_bn3",nn.BatchNorm1d(60))
        self.feature.add_module("f_relu3",nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(60,40))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(40))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(40, 20))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(20))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(20, n_out_classes))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(60,40))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(40))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(40, 2))

    def forward(self, input_data, alpha=1.0):
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def transform(self,X,layer="fc",batch_size=128):
        assert layer in ("feature","fc")
        self.eval()
        device=next(self.parameters()).device
        X_tensor=torch.FloatTensor(X)
        
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        
        X_transformed_tensor_list=list()
        def fc_transformation(fc,val):
            module_list=list(fc._modules.values())[:6]
            for m in module_list:
                val=m(val)
            return val
        with torch.no_grad():
            for i,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                X_batch_transformed_tensor=self.feature(X_batch)
                if layer=="fc":
                    X_batch_transformed_tensor=fc_transformation(self.class_classifier,X_batch_transformed_tensor)
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor=torch.cat(X_transformed_tensor_list,dim=0)
        X_transformed_arr=X_transformed_tensor.cpu().numpy()
        self.train()
        return X_transformed_arr

class FCModelNoDropNoBN(nn.Module):
    def __init__(self,n_in_features,n_out_classes=4):
        super().__init__()
        self.n_in_features=n_in_features
        self.n_out_classes=n_out_classes
        self.feature = nn.Sequential()
        self.feature.add_module("f_fc1",nn.Linear(n_in_features,100))
        # self.feature.add_module("f_bn1",nn.BatchNorm1d(100))
        self.feature.add_module("f_relu1",nn.ReLU(True))
        # self.feature.add_module("f_drop1",nn.Dropout())
        self.feature.add_module("f_fc2",nn.Linear(100,80))
        # self.feature.add_module("f_bn2",nn.BatchNorm1d(80))
        self.feature.add_module("f_relu2",nn.ReLU(True))
        self.feature.add_module("f_fc3",nn.Linear(80,60))
        # self.feature.add_module("f_bn3",nn.BatchNorm1d(60))
        self.feature.add_module("f_relu3",nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(60,40))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(40))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(40, 20))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(20))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(20, n_out_classes))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(60,40))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(40))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(40, 2))


    def forward(self, input_data, alpha, return_ddc_features=None):
        if return_ddc_features is not None:
            assert return_ddc_features in self.class_classifier._modules
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output=feature
        ddc_features=None
        for k,v in self.class_classifier._modules.items():
            class_output=v(class_output)
            if k==return_ddc_features:
                ddc_features=class_output
        domain_output = self.domain_classifier(reverse_feature)
        if return_ddc_features:
            return class_output,domain_output,ddc_features
        else:
            return class_output,domain_output

    def transform(self,X,layer="fc",batch_size=128):
        assert layer in ("feature","fc")
        
        self.eval()
        device=next(self.parameters()).device
        X_tensor=torch.FloatTensor(X)
        
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        
        X_transformed_tensor_list=list()
        def fc_transformation(fc,val):
            module_list=list(fc._modules.values())[:3]
            for m in module_list:
                val=m(val)
            return val
        with torch.no_grad():
            for i,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                X_batch_transformed_tensor=self.feature(X_batch)
                if layer=="fc":
                    X_batch_transformed_tensor=fc_transformation(self.class_classifier,X_batch_transformed_tensor)
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor=torch.cat(X_transformed_tensor_list,dim=0)
        X_transformed_arr=X_transformed_tensor.cpu().numpy()
        self.train()
        return X_transformed_arr

class FCModelNoDropNoBNLarge(nn.Module):
    def __init__(self,n_in_features,n_out_classes=4):
        super().__init__()
        self.n_in_features=n_in_features
        self.n_out_classes=n_out_classes
        self.feature = nn.Sequential()
        self.feature.add_module("f_fc1",nn.Linear(n_in_features,500))
        # self.feature.add_module("f_bn1",nn.BatchNorm1d(100))
        self.feature.add_module("f_relu1",nn.ReLU(True))
        self.feature.add_module("f_fc2",nn.Linear(500,250))
        # self.feature.add_module("f_bn2",nn.BatchNorm1d(100))
        self.feature.add_module("f_relu2",nn.ReLU(True))
        self.feature.add_module("f_fc3",nn.Linear(250,100))
        # self.feature.add_module("f_bn3",nn.BatchNorm1d(100))
        self.feature.add_module("f_relu3",nn.ReLU(True))
        # self.feature.add_module("f_drop1",nn.Dropout())
        self.feature.add_module("f_fc4",nn.Linear(100,80))
        # self.feature.add_module("f_bn4",nn.BatchNorm1d(80))
        self.feature.add_module("f_relu4",nn.ReLU(True))
        self.feature.add_module("f_fc5",nn.Linear(80,60))
        # self.feature.add_module("f_bn5",nn.BatchNorm1d(60))
        self.feature.add_module("f_relu5",nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(60,40))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(40))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(40, 20))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(20))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(20, n_out_classes))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(60,40))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(40))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(40, 2))


    def forward(self, input_data, alpha, return_ddc_features=None):
        if return_ddc_features is not None:
            assert return_ddc_features in self.class_classifier._modules
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output=feature
        ddc_features=None
        for k,v in self.class_classifier._modules.items():
            class_output=v(class_output)
            if k==return_ddc_features:
                ddc_features=class_output
        domain_output = self.domain_classifier(reverse_feature)
        if return_ddc_features:
            return class_output,domain_output,ddc_features
        else:
            return class_output,domain_output

    def transform(self,X,layer="fc",batch_size=128):
        assert layer in ("feature","fc")
        
        self.eval()
        device=next(self.parameters()).device
        X_tensor=torch.FloatTensor(X)
        
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        
        X_transformed_tensor_list=list()
        def fc_transformation(fc,val):
            module_list=list(fc._modules.values())[:3]
            for m in module_list:
                val=m(val)
            return val
        with torch.no_grad():
            for i,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                X_batch_transformed_tensor=self.feature(X_batch)
                if layer=="fc":
                    X_batch_transformed_tensor=fc_transformation(self.class_classifier,X_batch_transformed_tensor)
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor=torch.cat(X_transformed_tensor_list,dim=0)
        X_transformed_arr=X_transformed_tensor.cpu().numpy()
        self.train()
        return X_transformed_arr
class FCModelNoBN(nn.Module):
    def __init__(self,n_in_features,n_out_classes=4):
        super().__init__()
        self.n_in_features=n_in_features
        self.n_out_classes=n_out_classes
        self.feature = nn.Sequential()
        self.feature.add_module("f_fc1",nn.Linear(n_in_features,100))
        self.feature.add_module("f_relu1",nn.ReLU(True))
        self.feature.add_module("f_fc2",nn.Linear(100,80))
        self.feature.add_module("f_relu2",nn.ReLU(True))
        self.feature.add_module("f_fc3",nn.Linear(80,60))
        self.feature.add_module("f_relu3",nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(60,40))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(40, 20))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(20, n_out_classes))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(60,40))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(40, 2))

    def forward(self, input_data, alpha=1.0):
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def transform(self,X,layer="fc",batch_size=128):
        assert layer in ("feature","fc")
        self.eval()
        device=next(self.parameters()).device
        X_tensor=torch.FloatTensor(X)
        
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        
        X_transformed_tensor_list=list()
        def fc_transformation(fc,val):
            module_list=list(fc._modules.values())[:4]
            for m in module_list:
                val=m(val)
            return val
        with torch.no_grad():
            for i,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                X_batch_transformed_tensor=self.feature(X_batch)
                if layer=="fc":
                    X_batch_transformed_tensor=fc_transformation(self.class_classifier,X_batch_transformed_tensor)
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor=torch.cat(X_transformed_tensor_list,dim=0)
        X_transformed_arr=X_transformed_tensor.cpu().numpy()
        self.train()
        return X_transformed_arr
class TrainableModel(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params=params
        self.loss_fn=params["loss_fn"]
        self.optimizer=None
    def fit(self,X,Y,test_data):
        params=self.params
        self.optimizer=torch.optim.Adam(self.parameters(),lr=params['lr'])
        self.train()
        X_tensor=torch.from_numpy(X)
        Y_tensor=torch.from_numpy(Y)
        
        train_dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
        train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=params["batch_size"])
        device=next(iter(self.parameters())).device
        for epoch in range(1,params["n_epochs"]+1):
            print("Begin Epoch %d"%(epoch))
            for i,(X_batch,Y_batch) in enumerate(train_loader):
                X_batch=X_batch.to(device)
                Y_batch=Y_batch.to(device)
                pred=self(X_batch)
                loss=self.loss_fn(pred,Y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i%100==0:
                    print("step [%d/%d], loss=%.4f"%(i+1,len(train_loader),loss))
            print("End of epoch evaluation:")
            print(self.score(*test_data))
    def score(self,X,Y):
        self.eval()
        X_tensor=torch.from_numpy(X)
        Y_tensor=torch.from_numpy(Y)
        
        test_dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
        test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=params["batch_size"])
        
        pred_list=list()
        device=next(iter(self.parameters())).device
        with torch.no_grad():
            for i,(X_batch,Y_batch) in enumerate(test_loader):
                X_batch=X_batch.to(device)
                Y_batch=Y_batch.to(device)
                pred=self(X_batch)
                pred=pred.argmax(dim=1)
                pred_list.append(pred)
        predictions_tensor=torch.cat(pred_list)
        predictions_arr=predictions_tensor.cpu().numpy()
        score=dict()
        score["accuracy"]=np.mean(predictions_arr==Y)
        return score
            
class FeatureAndClassifierModel(TrainableModel):
    def __init__(self,params):
        super().__init__(params)
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        
        mock_input=torch.rand(2,3,28,28)
        mock_output=self.feature(mock_input)
        self.feature_size=np.prod(mock_output.shape[1:])
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.feature_size, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
    def forward(self,X):
        X=self.feature(X)
        X=X.view(-1,self.feature_size)
        X=self.class_classifier(X)
        return X

class ClassifierModel(TrainableModel):
    def __init__(self,params):
        super().__init__(params)
        self.feature_size=800
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.feature_size, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
    def forward(self,X):
        X=self.class_classifier(X)
        return X