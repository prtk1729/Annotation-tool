import torch
import argparse
import torch.nn as nn
import torchvision

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

import random
import sys
import json

from sklearn.cluster import KMeans
from scipy import spatial


# global variables
global_val_loss = []
global_train_acc = []
global_val_acc = []
global_train_loss = []
global_req_list = []


# ========= CLI Parsing ===============
arg_container = argparse.ArgumentParser(description='Specify the Operating System')

# should be optional arguments container.add_arguments
arg_container.add_argument('--bs', '-batch_size', type=int, required=True, help='Enter the batch_size for batch training')
arg_container.add_argument('--hs', '-hidden_size', type=int, required=True, help='Enter the batch_size for batch training')
arg_container.add_argument('--lr', '-learning_rate', type=float, required=True, help='Enter the batch_size for batch training')
arg_container.add_argument('--n_epochs', '-epochs', type=int, required=True, help='Enter the batch_size for batch training')

# container.parse_args() and store in args
args = arg_container.parse_args()
# ====================================
print(args.lr)


# =============== hyperparams ====================
INPUT_SIZE = 784 #1*28*28
batch_size = args.bs
hidden_size = args.hs
learning_rate = args.lr
n_epochs = args.n_epochs
OUTPUT_SIZE = 10 #capital means constants
# ================================================



# =========== model blueprint =======================
class LitMNISTClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(LitMNISTClassifier, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() 
        self.l2 = nn.Linear(hidden_size, output_size)

        # set the metrics
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, x, with_feat_vect=False):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # print('\n\n\nforward()\n\n')
        out = self.l1(x) 
        feat_vect = self.relu(out) 
        out = self.l2(out)

        if with_feat_vect:
            return out, feat_vect
        else:
            return out

    def training_step(self, train_batch, batch_idx):
        '''Already has batch_idx and train_batch implemented. NO need to do for loops.
        It implicitly is done here. Therefore you dont see any for loops.'''
        X_train_batch_tensor, y_train_batch_tensor = train_batch
        X_train_batch_tensor = X_train_batch_tensor.reshape(-1, 28*28)  

        # forward and loss calculation
        out = self(X_train_batch_tensor)       
        train_loss = F.cross_entropy(out, y_train_batch_tensor)
        train_acc = self.train_acc(out, y_train_batch_tensor)
        # print('\n\n',train_acc) # tensor(0.1000, device='cuda:0')
        # print('\n\n',train_loss) # tensor(2.3652, device='cuda:0', grad_fn=<NllLossBackward>)

        # print('X_train_batch_tensor.shape', X_train_batch_tensor.shape) #batch_size * 784
        global_train_acc.append(train_acc)
        global_train_loss.append(train_loss)
        # print(global_train_acc, global_train_loss) #[tensor(0.1000, device='cuda:0')] [tensor(2.3652, device='cuda:0', grad_fn=<NllLossBackward>)]

        train_dict = {'loss':train_loss} 
        return train_dict

    def test_step(self, val_batch, batch_idx):
        '''Already has batch_idx and train_batch implemented. NO need to do for loops.
        It implicitly is done here. Therefore you dont see any for loops.'''
        X_val_batch_tensor, y_val_batch_tensor = val_batch
        # reshape
        X_val_batch_tensor = X_val_batch_tensor.reshape(-1, 28*28)  

        # forward and loss calculation
        out = self(X_val_batch_tensor)       
        valid_loss = F.cross_entropy(out, y_val_batch_tensor)
        valid_acc = self.valid_acc(out, y_val_batch_tensor)
        self.log('test_acc', self.valid_acc(out, y_val_batch_tensor), on_step=False, on_epoch=True)


        global_val_acc.append(valid_acc)
        global_val_loss.append(valid_loss)
    #     # self.log('valid_acc_step', valid_acc)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        return optimizer

    def training_epoch_end(self, training_step_outputs):
        self.logger.experiment.flush()


    # def train_dataloader(self):
    #     '''Only train_data amd train_loader obj of preprocessing'''
    #     train_data = torchvision.datasets.MNIST(root='./data',
    #                                              download=True, 
    #                                              train=True, 
    #                                              transform=torchvision.transforms.ToTensor()) #trap torchvision.transforms.ToTensor()

    #     train_loader = torch.utils.data.DataLoader(dataset=train_data, 
    #                                                 shuffle=True, 
    #                                                 num_workers=4,
    #                                                 batch_size=batch_size)
    #     return train_loader


    
    # def val_dataloader(self):
    #     '''Only val_data amd val_loader obj of preprocessing'''
    #     val_data = torchvision.datasets.MNIST(root='./data', 
    #                                              train=False, 
    #                                              transform=torchvision.transforms.ToTensor()) 

    #     val_loader = torch.utils.data.DataLoader(dataset=val_data, 
    #                                             batch_size=batch_size,
    #                                             num_workers=4,
    #                                             shuffle=False)
    #     return val_loader


    #     # val_dict = {"val_loss":loss, "val_acc":valid_acc}
    #     val_dict = {'val_loss':valid_loss}
    #     return val_dict





def find_next_batch(model, images, args):
    '''returns indices of most uncertain images in unqueried'''

#   initialize entropy and feat_vect
    entropy = torch.Tensor(images.shape[0]) #images.shape[0] = batch_size
    feat_vects = torch.Tensor(images.shape[0], args.hs)

    loader = DataLoader(images, args.bs)
    # do forward pass on dataset and get feature vectors and entropy of distributions
    # higher the entropy higher the uncertainity (more closer to uniform distribution)
    start_idx = 0
    for batch in loader:
        print("batch.shape",batch.shape) #batch.shape torch.Size([10, 1, 28, 28])
        # get feat_vect afetr 1st layer relu and based on this vector do Entropy
        out, feat_vect = model.to("cuda").forward(batch.to("cuda"), with_feat_vect=True)

        # Entropy operation to find top k.
        out = (F.softmax(out, dim=1)*-1*F.log_softmax(out, dim=1)).sum(dim=1)

        # entropy.shape = batch_size * hidden_size
        entropy[start_idx: start_idx+out.shape[0]] = out

        feat_vects[start_idx: start_idx+out.shape[0]] = feat_vect
        start_idx += out.shape[0]

    # find the inputs with 1000 highiest prediction entropy    
    feat_vects = feat_vects[torch.topk(entropy, 1000)[1]].detach().numpy()

    # Do k means on above to find cluster centers
    
    kmeans = KMeans(n_clusters=args.bs, random_state=0).fit(feat_vects).cluster_centers_

    # Find points in the dataset which are closest to each of the 100 cluster centers
    kdt = spatial.KDTree(feat_vects)
    indices =[]
    for i in range(kmeans.shape[0]):
        indices.append(kdt.query(kmeans[i])[1])
    return indices

def random_stratergy(model, images, queried, args):
    '''returns indices of next_batch points acc. to random_strategy'''
    indices = []
    while len(indices) < args.bs:
        i = random.randrange(0, images.shape[0])
        if i not in queried:
            indices.append(i)
    return indices



# mappings
# mnist_train == train_data
# mnist_test == val_data


if __name__ == '__main__':
    #========================= prepare data ===============================
    train_data = torchvision.datasets.MNIST(root='./data',
                                                 download=True, 
                                                 train=True, 
                                                 transform=torchvision.transforms.ToTensor()) #trap torchvision.transforms.ToTensor()

    # print(type(train_data)) #torchvision.datasets.mnist.MNIST

    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                                    num_workers= 4,
                                                    batch_size=args.bs)

    # print(type(train_loader)) #torch.utils.data.dataloader.DataLoader



    val_data = torchvision.datasets.MNIST(root='./data', 
                                                 train=False, 
                                                 download=True,
                                                 transform=torchvision.transforms.ToTensor()) 


    # print(type(val_data)) #torchvision.datasets.mnist.MNIST

    
    # ========================================================


    # =========== trainer_obj ============================
    # trainer = Trainer(fast_dev_run=True, gpus=1)
    trainer = Trainer(max_epochs=n_epochs,fast_dev_run=False, gpus=1)
    # ========================================================



    # ======================== initialize =========================
    pool_images = torch.zeros(len(train_data), 1, 28 , 28)
    pool_labels = torch.zeros(len(train_data), dtype=torch.long)
    for i in range(len(train_data)):
        pool_images[i], pool_labels[i] = train_data[i]

    print('\n\n',pool_images.shape) #torch.Size([60000, 1, 28, 28])
    print(pool_labels.shape) #torch.Size([60000])

#   create the dataloader_obj that creates the D_test in img.
    val_loader = torch.utils.data.DataLoader(dataset=val_data, 
                                                batch_size=args.bs,
                                                num_workers=  4)

    
    # # initialise queried i.e train_set_idx and unqueried i.e Pool_set_idx
    pool_dataset_idx = list(range(len(train_data))) #unqueried
    train_dataset_idx = [] #queried
    model = LitMNISTClassifier(INPUT_SIZE, hidden_size, OUTPUT_SIZE) #just create model_obj and no fit

    val_acc_list = []
    # batch_training_loop
    for batch_index in range(200):
        print(f'\n{batch_index}\n')
        # entry_condition
        if batch_index>=3:
            idx_0, idx_1, idx_2 = (batch_index-3), (batch_index-2), (batch_index-1)
            if (val_acc_list[idx_0] > val_acc_list[idx_1]) and (val_acc_list[idx_1] > val_acc_list[idx_2]):
                print("satisfied_cond", batch_index)
                print(val_acc_list[idx_0], val_acc_list[idx_1], val_acc_list[idx_2])
                sys.exit()


        # always reset the trainer that has seen uptil_now_train_dataset (which is getting accumulated)
        trainer.current_epoch = 0 #reset everytime for every reload
        new_indices = random_stratergy(model, pool_images, train_dataset_idx, args)
        # new_indices = find_next_batch(model, pool_images[pool_dataset_idx], args)

        # accumulate new_indies in D_train
        train_dataset_idx.append(new_indices)

        # set difference 
        pool_dataset_idx = [idx for idx in pool_dataset_idx if idx not in new_indices]

        # create test_loader_obj 
        test_loader_obj = DataLoader(TensorDataset(pool_images[train_dataset_idx], pool_labels[train_dataset_idx]), 
                                batch_size=args.bs,
                                num_workers = 4)

        # reinitialize model that would take these modified_pool_images for acc and loss calculation
        model = LitMNISTClassifier(INPUT_SIZE, hidden_size, OUTPUT_SIZE) #just create model_obj and no fit
        trainer.fit(model, test_loader_obj)
        # trainer.test(model, test_loader) 

        test_list = trainer.test(model, val_loader)
        print('hi')
        print("test_list",test_list)

        val_acc_list.append(f"{test_list[0]['test_acc']:.3f}")
        print('\n\nbatch_idx\n\n', batch_index)
        d = {"val_acc":val_acc_list}
        with open('random_strategy_lr_0.001_bs_10.json', 'w') as f:
            f.write(json.dumps(d))




    
        











    # trainer.fit(model)
    # print(len(global_train_acc)) #12000                                                                                                                                                                     
    # print(len(global_train_loss)) #12000                                                                                                                                                                     
    # print(len(global_val_acc)) #2002
    # print(len(global_val_loss)) #2002
