import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

print('torch.cuda.is_available()', torch.cuda.is_available())

# ----------
class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h1 = nn.functional.relu(self.layer1(x))
        h2 = nn.functional.relu(self.layer2(h1))
        do = self.do(h1+h2) #if the network wants to ignore h2 ==> make the value negative
        logits = self.layer3(do)
        return logits

    # pl has a method called configure_optimizers
    def configure_optimizers(self):
        # the model obj has a method parameters()--> use that
        optimizer1 = optim.SGD(self.parameters(), lr=1e-2)

        # If you use gan or something you can use multiple optimizers
        #  and each will have its own training loop
        optimizer2 = optim.SGD(self.parameters(), lr=0.1)
        return optimizer1

    def training_step(self, batch, batch_idx):
        x, y =batch #x = batch_size*c*h*w here b*1*28*28

        # reshape
        batch_size = x.size(0)
        x = x.view(batch_size, -1) #now the x is ready to pass through the nn.Linear

        # 2 Steps req for pl
        # 1. forward -- model is self now.
        logits = self(x) #logits also y_hat this is the output of the last Linear model

        # 2. compute obj. function
        # J = self.loss(logits, y.cuda()) 
        # remove .cuda() as lightning would put the correct devices for you.
        J = self.loss(logits, y)

        acc = accuracy(logits, y)
        pbar = {'training_acc': acc}
        return {'loss': J, 'progress_bar':pbar}  #'progress_bar' and 'loss' are predef keys.
        # the above dict is stored in log_files at the end of every batch
        # return J #returns a dict as shown above

    def train_dataloader(self):
        '''only put datasets.MNIST obj and train_DataLoader obj'''
        train_data = datasets.MNIST('data', download=True, train=True, transform=transforms.ToTensor())
        # train, val = random_split(train_data, [55000, 5000]) #random_split is a utility function to train and test split
        # self.train = train
        # self.val = val

        # print(type(train)) #torch.utils.data.dataset.Subset

        # Create DataLoader obj. one for train other for val.
        # train Trap or datasets obj.
        train_loader = DataLoader(train_data, batch_size=32) #create train_loader obj
        # val_loader = DataLoader(val, batch_size=32) #create val_loader obj
        return train_loader


    # def validation_step(self, batch, batch_idx):
    #     '''this method literally takes the same params as training_step and return same dict'''
    #     # so just call the training_step
    #     req_dict = self.training_step(batch, batch_idx)
    #     req_dict['progress_bar']['val_acc'] = req_dict['progress_bar']['training_acc']
    #     del req_dict['progress_bar']['training_acc']
    #     return req_dict

    # def validation_epoch_end(self, val_step_outputs):
    #     '''Invoked on every validation epoch end'''
    #     # [req_dict_batch1, req_dict_batch_2,....., req_dict_batch_<n_batches>]
    #     # calculate validation loss not just for a batch for for the whole 
    #     # validation_set
    #     avg_val_loss = torch.tensor([batch_i_req_dict['loss'] for batch_i_req_dict in val_step_outputs]).mean()
    #     # at the end of every epoch we need to put this avg_val_loss in the log_files
    #     # callbacks are looking for this pre_def keyword val_loss to stop training

    #     avg_val_acc = torch.tensor([batch_i_req_dict['progress_bar']['val_acc'] for batch_i_req_dict in val_step_outputs]).mean()
    #     pbar = {'avg_val_acc': avg_val_acc}

    #     return {'val_loss': avg_val_loss, 'progress_bar':pbar} 
    #     # if this is missing it wont go for the other loss, it will just wont early-stop


    

    # def val_dataloader(self):
    #     val_loader = DataLoader(self.val, batch_size=32)
    #     return val_loader

# create model obj
model = LitClassifier()

# create Trainer obj.
# trainer = pl.Trainer(max_epochs=5, gpus=1)
trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=5, gpus=1) 
trainer.fit(model)


