# Udacity-AI-Programming-with-Python-Final-Project-Part2

Hello! This is Part 2 (The Command Line App) of the final Project for the Udacity AI with Python Nanodegree.

Train a new network on a data set with train.py
    Basic Usage : python train.py data_directory
    Prints out current epoch, training loss, validation loss, and validation accuracy as the network trains
    Options:
        Set directories to load data and save checkpoints: 
	syntax: python train.py data_dir --save_dir save_directory
        Choose architecture (vgg16, alexnet, densenet121, or vgg13) 
        example: python train.py data_dir --arch "vgg16"
        Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_layer1 1024 --epochs 20
        Use GPU for training: python train.py data_dir --gpu gpu

Predict the top k (default = 5) most likely flower names and probabilties from an image with predict.py

    Basic usage: python predict.py /path/to/image checkpoint
    Options:
        Return top K most likely classes: python predict.py input checkpoint ---top_k 3
        Use a mapping of categories to real names: python predict.py input checkpoint --	category_names cat_To_name.json
        Use GPU for inference: python predict.py input checkpoint --gpu

You can clone this repository using

git clone https://github.com/ebaung/Udacity-AI-Programming-with-Python-Final-Project-Part2

Json file

In order for the network to correctly identify the name of the flower a .json file is required. This .json file sorts the data based on numbered folders in the dataset, and those numbers will correspond to specific names specified in the .json file.

Flowers Dataset

Unfortunately, the flower images dataset used for this application is too large to upload to Github. However, you may use any dataset of .jpg images (images of anything, not just flowers!), meeting the following guidelines: -the data need to comprised of 3 folders, training, validation and testing. -the distribution should generally be: 70% training, 10% validation and 20% testing. -these folders should contain numbered subfolders. Each unique number corresponds to a specific category, clarified in the json file (please view the cat_to_name.json file included in this Github repository to see the format).

Eugene Baung
