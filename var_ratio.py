from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torchvision.transforms.transforms import ToPILImage
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from torchvision.datasets.mnist import MNIST
from torchvision import transforms
import copy
import random
import pandas as pd
import json
import sys
from scipy import stats


global_val_loss = []
global_train_acc = []
global_val_acc = []
global_train_loss = []
global_req_list = []

class LitClassifier(pl.LightningModule):

    def __init__(self, bundle_size, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.bundle_size = bundle_size
        self.fc = nn.Linear(28*28, 256)
        self.fc1   = nn.Linear(256, 10)
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.2)

        self.train_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x, with_feat_vect=False):
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        out = F.relu(x)
        feat_vect = out #feat

        out = self.dropout(out)
        out = self.fc1(out)
        
        if with_feat_vect:
            return out, feat_vect
        else:
            return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_acc', self.train_accuracy(y_hat, y), on_step=False, on_epoch=True)
        train_acc = self.train_accuracy(y_hat, y)
        train_loss = loss
        global_train_acc.append(train_acc)
        global_train_loss.append(train_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        valid_loss = loss
        valid_acc = self.test_accuracy(y_hat, y)
        self.log('test_acc', self.test_accuracy(y_hat, y), on_step=False, on_epoch=True)

        self.log('test_loss', loss)
        global_val_acc.append(valid_acc)
        global_val_loss.append(valid_loss)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_epoch_end(self, training_step_outputs):
        self.logger.experiment.flush()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--bundle_size', type=int, default=10)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser




def var_ratios(model, images, args):
    # get a random subset of 1000 indices without replacement
    random_subset = np.random.choice(range(len(images)), size=1000, replace=False)  
    print(len(images))


    # create a loader_obj
    # args.batch_size args.bundle_size
    loader_obj = DataLoader(images, args.bundle_size)
    # do forward pass on dataset and get feature vectors and entropy of distributions
    # higher the entropy higher the uncertainity (more closer to uniform distribution)
    

    t_probs_stack = []
    # 10 stochastic forward passes for a particular batch

    T = 100 #'T' stochastic forward passes
    with torch.no_grad():
        for i in range(T):
            logits, feat_vect = model.to("cuda").forward(images[random_subset].to("cuda"), with_feat_vect=True)
            print("logits after T passes: ",logits)
            # print(logits.shape, feat_vect.shape) #[2, 10] [2, 256] for random_sample_size==2
            a = torch.softmax(logits.to('cpu'), dim=1).numpy()
            t_probs_stack.append(a)


    # print("\nt_probs_stack\n",t_probs_stack)
    t_probs_stack =  np.array(t_probs_stack)
    # print(t_probs_stack.shape) #(10, 2, 10) 10-no. of stochastic forward passes; and random_sample_size==2 ; 10- mo. of logits
    # print(np.sum(stochastic_stack[0], axis=-1 ) )
    
    # t_probs_stack.shape ===> (T; bs/random_sample_size ;n_out_units)

    # argmax of t_probs_stack along axis =-1
    y_pred_T_sfp = t_probs_stack.argmax(axis=-1)
    # print("\ny_pred_T_sfp\n", y_pred_T_sfp)
    # What's the mode?
    print(stats.mode(y_pred_T_sfp))
    mode_count_T = stats.mode(y_pred_T_sfp)[-1]
    mode_count_T = np.array(mode_count_T).squeeze(axis=0).squeeze()
    print("\nmode_count_T\n", mode_count_T)

    # shape f_m_T is (random_sample_size, 1).
    # apply argmax 
    # return indices corresponding to bottom k mode_values  
    k = 10
    req_indices = mode_count_T.argsort()[:k]
    print("\nreq_indices\n", req_indices)
    req_indices = list(req_indices)


    return req_indices 




def stopping_condition(val_acc_list, batch_index):
    '''checks convergence or 1000'''
    # idx_0 = batch_index-1
    print(type(val_acc_list), val_acc_list,  batch_index)
    # if float(val_acc_list[batch_index-1]) > 0.4:
    if batch_index == 49:
        print("satisfied_cond", batch_index)
        # print(val_acc_list[idx_0], val_acc_list[idx_1], val_acc_list[idx_2])
        sys.exit()

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser  = ArgumentParser()
    # parser.add_argument('--batch_size', default=10, type=int)
    parser  = pl.Trainer.add_argparse_args(parser)
    parser  = LitClassifier.add_model_specific_args(parser)
    args    = parser.parse_args()

    

    # ------------
    # data
    # ------------
    mnist_train = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test  = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    trainer = pl.Trainer.from_argparse_args(args, gpus=1)

    full_images = torch.zeros(len(mnist_train), 1, 28 , 28)
    full_labels = torch.zeros(len(mnist_train), dtype=torch.long)
    for i in range(len(mnist_train)):
        full_images[i], full_labels[i] = mnist_train[i]
    test_loader     = DataLoader(mnist_test, batch_size=args.bundle_size, num_workers=4)
    

    queried_indices = []
    unqueried_indices = list(range(len(mnist_train)))
    model = LitClassifier(args.bundle_size, args.learning_rate)

    val_acc_list = []

    stop_i = 0
    for batch_index in range(100):
        if stop_i == 1:
            break
        if batch_index>=3:
            # stopping_condition(val_acc_list, batch_index)
            idx_0, idx_1, idx_2 = (batch_index-3), (batch_index-2), (batch_index-1)
            if (val_acc_list[idx_0] > val_acc_list[idx_1]) and (val_acc_list[idx_1] > val_acc_list[idx_2]):
                print("satisfied_cond", batch_index)
                print(val_acc_list[idx_0], val_acc_list[idx_1], val_acc_list[idx_2])
                sys.exit()
        trainer.current_epoch = 0

        # ========================= implement var_ratio approach =========================
        new_indices = var_ratios(model, full_images[unqueried_indices], args)
        # =================================================================================================



        queried_indices += new_indices
        unqueried_indices = [ i for i in unqueried_indices if i not in new_indices ]
        X,Y = full_images[queried_indices], full_labels[queried_indices]
        dl  = DataLoader(TensorDataset(X,Y), batch_size=args.bundle_size)
        model = LitClassifier(args.bundle_size, args.learning_rate)
        trainer.fit(model, dl)

        # print('\nlen(global_train_acc)',len(global_train_acc)) #6000 initially
        # cur_acc = sum(global_train_acc)/len(global_train_acc)
        # val_acc_list.append(cur_acc)
        # print('\n\ncur_acc', cur_acc) 

        test_list = trainer.test(model, test_loader)
        val_acc_list.append(f"{test_list[0]['test_acc']:.3f}")
        print('\n\nbatch_idx\n\n', batch_index)
        d = {"val_acc":val_acc_list}
        with open('./var_ratio_10_lr_0.001_bs_10_500_epoch_2pm_dropout_con3.json', 'w') as f:
            f.write(json.dumps(d))



if __name__ == '__main__':
    cli_main()





