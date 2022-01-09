# Fungal burden quantification with U-Net
*Project 3 for an MSc in Bioinformatics and Theoretical Systems Biology at Imperial College London.*

This project was undertaken in the [Tanaka Group](https://rtanaka.bg-research.cc.ic.ac.uk/) under the supervision of Dr Reiko Tanaka and Dr Rahman Attar. </br>
The images used for building the U-Net model were provided by Prof. Elaine Bignell and Natasha Motsi. 


## Project overview
U-Net model for predicting fungal regions of input lung histology images infected with *Aspergillus fumigatus*.\
The model was built using PyTorch and was adapted from the GitHub repository: https://github.com/milesial/Pytorch-UNet

For a given input RGB lung histology image the U-Net model will output the segmented mask of the infected region (white: fungi, black: non-fungi region):
![input_ouput_unet](https://user-images.githubusercontent.com/77961877/148688541-a06e4b42-b8c3-494d-af65-7df691f36d9e.PNG)

## Contents
- [Training](#training)
- [Prediction](#prediction)
- [Pretrained model](#pretrained-model)
- [Data](#data)
- [Augmentations](#augmentations)
- [Quantifying fungal burden](#quantifying-fungal-burden)


### Training
The Python code for training the U-Net model is `train.py`.The parameters for model training are:
```bash
  --epochs, -e             Number of epochs (default: 5) 
  --batch-size, -b         Batch size (default: 1) 
  --learning-rate, -l      Learning rate (default: 0.0001) 
  --load, -f               Load model from a .pth file (default: False)
  --scale, -s              Input image downscaling factor (default: 0.5)
  --k-folds, -k            Number of folds of k-fold cross-validation (default: 5)
  --data-augmentation, -d  Number of augmented images generated per image of the train folds (default: 5)
  --checkpoint, -c         Save checkpoints (default: True)
```
The `scale` parameter will downscale the input images, reducing the memory requirements for model training. To get the best predictive results use a `scale` of 1.

An example for running `train.py` in the Unix shell is:
```bash 
python train.py -s 1 -d 10 
```

### Prediction
The Python code `predict.py`can be employed for predicting the fungi segmented mask from an input histology image.
The following parameters can be passed for making the predictions:
```bash
  --model, -m            Pretrained model from a .pth file 
  --input, -i            Directory for input histology image(s)
  --output, -o           Directory to store predicted mask(s)
  --viz, -v              Whether images will be visualised as they are being processed (default: False)
  --no-save, n           To not save predicted images (default: False)
  --mask-threshold, -t   Threshold for considering mask pixel as white (default: 0.5)
  --scale, -s            Input image downscaling factor (default: 0.5)
```

### Pretrained model
The pretrained model was trained on four Nvidia Quadro RTX 6000 (24GB) GPUS.
The model was trained with default parameters and a `scale` of 1 and `data-augmentation` of 10. \
The pretrained model can be employed to predict the masks of new unseen images using the `predict.py` file, for example:
```bash 
python predict.py -m pretained_model.pth -i /data/test/imgs/1M05_22.3x_11.jpg -o /data/test/out/1M05_22.3x_11_out.jpg -v -s 1 
```

### Augmentations
The augmentations explored can be found under the `utils` directory in the `augmentations.py` file.\
For each image in the training set a *n* number of augmented images are generated. Both original images and augmented images are used for model training.
Further work can include excluding the original images from the training set.\
The `augmentations.py` file includes both the geometric and colour-space augmentations explored, however, in this study the highest performance was obtained by applying soley geometric augmentations.

### Quantifying fungal burden
Fungal burden was quantified as fungi area per tissue area. The jupyter notebook for quantifying the fungal burden is `fungal_burden.ipynb`.

**Python packages required**
```
- numpy
- PyTorch
- sklearn
- matplotlib
- seaborn
- tqdm
- argparse
- Pillow
- torchvision 
- opencv
- albumentations

```

