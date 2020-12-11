import torch.utils.data
from PIL import Image
import os
import test
import sklearn.datasets
import numpy as np
import torchvision
import pandas as pd
class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

def random_nn_transformation(X,seed=100,n_layers=3,activation="tanh"):
    assert activation in ("tanh","relu")
    weights_list=list()
    bias_list=list()
    with test.temp_seed(seed):
        for i in range(n_layers):
            weight1=np.random.randn(200,200)*0.03+np.eye(200)
            bias1=np.random.randn(1,200)
            weights_list.append(weight1)
            bias_list.append(bias1)
    for i in range(n_layers):
        X=X.dot(weights_list[i])+bias_list[i]
        if activation=="tanh":
            X=np.tanh(X)
        elif activation=="relu":
            X=X*(X>0)
    return X

def permute_data(X,Y,seed=100):
    assert len(X)==len(Y)
    with test.temp_seed(seed):
        perm=np.random.permutation(X.shape[0])
    X=X[perm,:]
    Y=Y[perm]
    return X,Y

def get_blob_data_1():
    X_latent,Y_latent=sklearn.datasets.make_blobs(n_samples=10000,centers=4,n_features=200,random_state=123)
    X_source=X_latent.copy()
    Y_source=Y_latent.copy()
    X_target=-X_latent.copy()
    Y_target=Y_latent.copy()
    X_source,Y_source=permute_data(X_source,Y_source)
    X_target,Y_target=permute_data(X_target,Y_target)
    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':Y_target,
        'source_dataset_name':"make_blobs",
        'target_dataset_name':"neg_make_blobs"
    }

def get_blob_data_2():
    X_latent,Y_latent=sklearn.datasets.make_blobs(n_samples=10000,centers=4,n_features=200,random_state=123)
    X_source=X_latent.copy()
    Y_source=Y_latent.copy()
    X_target=-2*X_latent.copy()
    Y_target=Y_latent.copy()
    X_source,Y_source=permute_data(X_source,Y_source)
    X_target,Y_target=permute_data(X_target,Y_target)
    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':Y_target,
        'source_dataset_name':"make_blobs",
        'target_dataset_name':"2times_neg_make_blobs"
    }
def get_blob_data_3():
    X_latent,Y_latent=sklearn.datasets.make_blobs(n_samples=10000,centers=4,n_features=200,random_state=123)
    X_source=random_nn_transformation(X_latent,n_layers=3,seed=200,activation="tanh")
    Y_source=Y_latent.copy()
    X_target=random_nn_transformation(X_latent,n_layers=3,seed=300,activation="tanh")
    Y_target=Y_latent.copy()
    X_source,Y_source=permute_data(X_source,Y_source)
    X_target,Y_target=permute_data(X_target,Y_target)
    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':Y_target,
        'source_dataset_name':"tanh_nn1_make_blobs",
        'target_dataset_name':"tanh_nn2_make_blobs"
    }
def get_blob_data_4():
    X_latent,Y_latent=sklearn.datasets.make_blobs(n_samples=10000,centers=4,n_features=200,random_state=123)
    X_source=random_nn_transformation(X_latent,n_layers=10,seed=200,activation="relu")
    Y_source=Y_latent.copy()
    X_target=random_nn_transformation(X_latent,n_layers=10,seed=300,activation="relu")
    Y_target=Y_latent.copy()

    X_source,Y_source=permute_data(X_source,Y_source)
    X_target,Y_target=permute_data(X_target,Y_target)

    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':Y_target,
        'source_dataset_name':"ReLU_nn1_make_blobs",
        'target_dataset_name':"ReLU_nn2_make_blobs"
    }

def get_numpy_dataset(torch_dataset):
    X_list=[]
    Y_list=[]
    for i in range(len(torch_dataset)):
        X,Y=torch_dataset[i]
        if X.shape[0]==1:
            X=X.expand(3,-1,-1)
        elif X.shape[0]==3:
            pass
        else:
            assert False
        X_list.append(X)
        Y_list.append(Y)
    Y_arr=np.array(Y_list)
    X_arr=torch.stack(X_list,dim=0).numpy()
    return X_arr,Y_arr

def get_mnist_mnistm_dataset():
    image_size = 28
    source_dataset_name = 'MNIST'
    target_dataset_name = 'mnist_m'
    source_image_root = os.path.join('dataset', source_dataset_name)
    target_image_root = os.path.join('dataset', target_dataset_name)
    img_transform_source = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset_source = torchvision.datasets.MNIST(
        root='dataset',
        train=True,
        transform=img_transform_source,
        download=True
    )

    X_source,Y_source=get_numpy_dataset(dataset_source)

    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_target
    )


    X_target,Y_target=get_numpy_dataset(dataset_target)
    
    
    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':Y_target,
        'source_dataset_name':"MNIST",
        'target_dataset_name':"MNIST-m"
    }

def get_mnist_negmnist_dataset():
    image_size = 28
    source_dataset_name = 'MNIST'
    target_dataset_name = 'neg_MNIST'
    source_image_root = os.path.join('dataset', source_dataset_name)
    img_transform_source = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])



    dataset_source = torchvision.datasets.MNIST(
        root='dataset',
        train=True,
        transform=img_transform_source,
        download=True
    )

    X_source,Y_source=get_numpy_dataset(dataset_source)


    X_target,Y_target=-X_source,Y_source.copy()
    
    X_target,Y_target=permute_data(X_target,Y_target)
    
    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':Y_target,
        'source_dataset_name':"MNIST",
        'target_dataset_name':"negMNIST"
    }

def get_CRISPRi_LINCS_dataset():
    level5_fs_table_fp="./data/level5_logtrans_DE_genes.hdf"
    level5_fs_df=pd.read_hdf(level5_fs_table_fp)
    CRISPRi_fs_table_fp="./data/CRISPRi_logtrans_DE_genes.hdf"
    CRISPRi_fs_df=pd.read_hdf(CRISPRi_fs_table_fp)
    CRISPRi_fs_values_df=CRISPRi_fs_df.iloc[:,1:]
    cluster_labels_df=pd.read_csv("./data/cells_batches_cluster.txt",sep=",",index_col=0)

    CRISPRi_fs_values_sample_df=CRISPRi_fs_values_df.copy()
    level5_fs_sample_df=level5_fs_df.sample(20000,axis=1,random_state=123)
    
    icindex=[CRISPRi_fs_values_df.columns.get_loc(i) for i in CRISPRi_fs_values_sample_df.columns]
    cluster_labels_sample_df=cluster_labels_df.iloc[icindex,:]

    CRISPRi_fs_values_T=CRISPRi_fs_values_df.values.T
    level5_fs_sample_values_T=level5_fs_sample_df.values.T
    print(CRISPRi_fs_values_T.shape)
    print(level5_fs_sample_values_T.shape)

    X_source=CRISPRi_fs_values_T
    Y_source=cluster_labels_sample_df["cluster"].values
    X_target=level5_fs_sample_values_T

    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':None,
        'source_dataset_name':"CRISPRi",
        'target_dataset_name':"LINCS"
    }
def get_KMPLOT_dataset():
    base_dir="./data"
    original_fp=os.path.join(base_dir,"KMPLOT_BRCA_XP_NORMALIZED_1000_kmeans_CLEANED.tsv")
    batch_label_fp=os.path.join(base_dir,"KMPLOT_BRCA_BATCH_LABELS_CLEANED.tsv")
    kmplot_table_fp=os.path.join(base_dir,"KMPLOT_BRCA_SURVIVAL.txt")

    original_table=pd.read_csv(original_fp,sep='\t',index_col=0)
    batch_label_table=pd.read_csv(batch_label_fp,sep='\t',index_col=0)

    kmplot_table=pd.read_csv(kmplot_table_fp,sep="\t",index_col=0)
    ER_index=original_table.index
    kmplot_table_index=kmplot_table["ER status"].index[~kmplot_table["ER status"].isna()]
    # kmplot_table_index=kmplot_table["ER status"].index
    ER_index=ER_index.intersection(kmplot_table_index)
    X_ER_arr=original_table.loc[ER_index].values
    batch_arr=batch_label_table["0"][ER_index].values
    ER_label_arr=kmplot_table["ER status"].loc[ER_index].values

    source_index=np.isin(batch_arr,[0,1,2,3])
    target_index=np.isin(batch_arr,[4])
    X_source=X_ER_arr[source_index]
    Y_source=ER_label_arr[source_index]
    X_target=X_ER_arr[target_index]
    Y_target=ER_label_arr[target_index]
    
    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':Y_target,
        'source_dataset_name':"KMPLOT_exps1",
        'target_dataset_name':"KMPLOT_exps2"
    }