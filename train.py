# code for training U-Net model, adapted from GitHub repository: "https://github.com/milesial/Pytorch-UNet"

# import libraries
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.model_selection import KFold

from eval import eval_net
from unet import UNet

from datetime import datetime
now = datetime.now()

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.augmentations import AugDataset
from torch.utils.data import DataLoader, random_split

# specify directories for images, maks and checkpoints
dir_img = './data/train/imgs/'
dir_mask = './data/train/masks/'
dir_checkpoint = './checkpoints/'
dir_cv ='./cv/'

# funtion for training the neural network
def train_net(net, device, dataset, train_set, val_set, fold = 0,  epochs=5, batch_size=1, 
              lr=0.0001, save_cp=True, img_scale=0.5, data_aug = 5):
    
    # track number of batches passed through the network
    global_step = 0
    # function will output txt file with the performance of the model on each epoch and fold
    out_cv.write(f'Fold {fold +1} \n') # write fold number in file
    
    # Define data loaders for training and validation set in this fold
    train_loader = DataLoader(AugDataset(train_set,num = data_aug), batch_size=batch_size, pin_memory=True, num_workers=8)
    val_loader = DataLoader(AugDataset(val_set, transform=None), batch_size=batch_size, pin_memory=True, num_workers=8)

    # initialize optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss() # use CrossEntropyLoss for multi-class segmentation
    else:
        criterion = nn.BCEWithLogitsLoss() # use BCEWithLogitsLoss for binary segmentation

    # run the training loop for defined number of epochs
    for epoch in range(epochs):

        net.train()

        # set initial loss value
        epoch_loss = 0
        # display epoch information and initiate training progress bar
        with tqdm(total=len(train_loader.dataset), desc=f'Fold {fold +1}, Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            # iterate over the DataLoader with training data
            for batch in train_loader:
                # define images and masks
                imgs = batch['image']
                true_masks = batch['mask']
                 # ensure that number of channels in input images are the same as the defined number of channels
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # perform forward pass
                masks_pred = net(imgs)

                # compute loss
                loss = criterion(masks_pred, true_masks)
                # save loss to summary
                epoch_loss += loss.item()
                # print loss
                writer.add_scalar('Loss/train', loss.item(), global_step)
                # display loss
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # zero the gradients
                optimizer.zero_grad()

                # perform backward pass
                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # perform optimization
                optimizer.step()
                
                # update progress bar based on how many images have passed through the model
                pbar.update(imgs.shape[0])
                # update number of batches passed through the model
                global_step += 1
                # save weights and biases to summary
                if global_step % (len(train_loader.dataset) // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # evaluate model
                    val_score = eval_net(net, val_loader, device)
                    # decay learning rate
                    scheduler.step(val_score)
                    # save learning rate to summary
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    # print validation cross entropy or Dice coefficient
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                    
                    # add batch image data to summary
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        # save checkpoint
        if save_cp:
            try:
                # create checkpoint directory if there isn't one
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            # name checkpoint file
            torch.save(net.state_dict(),
                       dir_checkpoint + f'{now.strftime("%b-%d-%Y_%H-%M-%S")}_CP_fold{fold +1}_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1}, fold {fold +1} saved !')
        
        # write dice score of each epoch in output txt file   
        out_cv.write(f'Epoch{epoch +1}: Dice Coeff {val_score} \n')
    # write dice scores of next fold in a new line
    out_cv.write('\n')  

# pass arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batch_size') # batch size was set to 1 as training images had slightly different dimentions
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='img_scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-k', '--k-folds',  dest='k_folds', type=int, default=5,
                        help='Number of folds used in cross-validation')
    parser.add_argument('-d', '--data-augmentations', dest='data_aug', type=int, default=5,
                        help='Data augmentations')
    parser.add_argument('-c', '--checkpoint', dest='save_cp', type=str, default=True,
                        help='Save checkpoints')

    return parser.parse_args()

# initialise logging INFO, for displaying training information
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# get arguments
args = get_args()
# display device used
logging.info(f'Using device {device}')

# convert images and masks into tensors
dataset = BasicDataset(dir_img, dir_mask, args.img_scale)

# writer will output to ./runs/ directory by default
writer = SummaryWriter(comment=f'LR_{args.lr}_BS_{args.batch_size}_SCALE_{args.img_scale}')
# print summary of input parameters
logging.info(f'''Starting training:
    Epochs:          {args.epochs}
    Batch size:      {args.batch_size}
    Learning rate:   {args.lr}
    Dataset size:    {len(dataset)}
    K-folds:         {args.k_folds}
    Checkpoints:     {args.save_cp}
    Device:          {device.type}
    Images scaling:  {args.img_scale}
    Data augmentations per image: {args.data_aug}
''')

# create txt file to save cross-validation data (variable named out_txt)
try:
    os.mkdir(dir_cv) # create directory for saving the cross-validation output txt file
except OSError:
    pass  
# create and name output txt file 
out_cv= open(dir_cv + f'{now.strftime("%b-%d-%Y_%H-%M-%S")}_LR_{args.lr}.txt', 'w')

# define the K-fold Cross Validator
kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state = 8)

# K-fold Cross Validation model evaluation
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

    # display fold number
    logging.info(f'''FOLD {fold +1}''')
    
    # initialise U-Net model and define number of channels, number of classes and up-scaling technique
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # if bilinear interpolation not indicated use transposed conv. up-scaling
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    
    # load saved model parameters
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    
    net.to(device=device)
    
    # Dividing data into folds
    train_set = torch.utils.data.dataset.Subset(dataset,train_idx)
    val_set = torch.utils.data.dataset.Subset(dataset,val_idx)
    
    # close checkpoint file and output txt when training is completed
    try:
        train_net(net=net,
                  dataset = dataset,
                  train_set = train_set,
                  val_set = val_set,
                  fold = fold,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  img_scale=args.img_scale,
                  data_aug=args.data_aug,)
    
    # close checkpoint file and output txt if training interrupted      
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        out_cv.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
# close checkpoint file and output txt when training is completed        
out_cv.close()
writer.close()



