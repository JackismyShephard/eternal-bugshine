import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from .config import RESNET34_FULL,BEETLE_DATASET
from src.models import download_model, load_model_weights_and_metrics


class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=(4,4), strides=(2,2),
                 padding=(1,1), **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))




def setup_nets(root_path, config):
    dataset_config = BEETLE_DATASET
    model_config = RESNET34_FULL

    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 448)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(dataset_config['mean'], dataset_config['std'])])

    gamset = ImageFolder(dataset_config['image_folder_path'], transform = transformer)
    dataloader = DataLoader(gamset, batch_size = 128, num_workers = 12)    

    model = download_model(model_config, dataset_config)
    _ = load_model_weights_and_metrics(model, model_config)
    _ = model.eval()

    input_channels = config['latent_dim']
    
    n_G = 64
    net_G = nn.Sequential(
        G_block(in_channels=input_channels, out_channels=n_G*16, kernel_size=(7,14),
                strides=1, padding=0),                  # Output: (64 * 16, 7, 14)
        G_block(in_channels=n_G*16, out_channels=n_G*8), # Output: (64 * 8, 14, 28)
        G_block(in_channels=n_G*8, out_channels=n_G*4), # Output: (64 * 4, 28, 56)
        G_block(in_channels=n_G*4, out_channels=n_G*2), # Output: (64 * 2, 56, 112)
        G_block(in_channels=n_G*2, out_channels=n_G),   # Output: (64, 112, 224)
        nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                        kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh())  # Output: (3, 224, 448)
    
    net_G = net_G.to('cuda')
    net_G.load_state_dict(torch.load(root_path + '_gen.pt'))
    _ = net_G.eval()

    if config['static']:
        return net_G, model, model, dataloader
        
    net_E = models.resnet18(pretrained=False)
    net_E = net_E.to('cuda')
    net_E.load_state_dict(torch.load(root_path + '_enc.pt'))
    _ = net_E.eval()
    return net_G, net_E, model, dataloader


def convert_tensorlist_to_tensor(l):
    batch_size = l[0].shape[0]
    batches = len(l)-1
    remaining = l[-1].shape[0]

    # suboptimal but not important
    if l[0].dim() == 1:
        ret = torch.zeros(batch_size*batches + remaining)
    if l[0].dim() == 2:
        ret = torch.zeros((batch_size*batches + remaining, l[0].shape[1]))

    for i in range(len(l)):
        ret[i*batch_size:(i+1)*batch_size] = l[i]
    
    return ret

def calc_preds_latents(net_G, net_E, model, dataloader, aug_func):
    latents_codes = []
    pred_class = []

    mean = torch.tensor([0.8442649, 0.82529384, 0.82333773]).reshape(3,1,1).to('cuda')
    std = torch.tensor([0.28980458, 0.32252666, 0.3240354]).reshape(3,1,1).to('cuda')

    for X, _ in dataloader:
        X = X.to('cuda')
        with torch.no_grad():
            latent = aug_func(net_E(X))
            gen = (net_G(latent.view(latent.shape[0], latent.shape[1], 1, 1))+1)/2
            pred = torch.argmax(model((gen-mean)/std).cpu(),1)
            latents_codes.append(latent.cpu().reshape(latent.shape[0], latent.shape[1]))
            pred_class.append(pred)
    return convert_tensorlist_to_tensor(pred_class), convert_tensorlist_to_tensor(latents_codes)

def calc_class_acc(real, pred):
    correct = np.bincount((real == pred)*(real.astype(np.int64)+1), minlength=198)[1:]
    nr_class = np.bincount(real.astype(np.int64))
    mask = nr_class != 0
    div = np.ones(nr_class.shape)
    div[mask] = nr_class[mask]
    return (correct/div)

def separate(real, pred, latents): # good stuff
    nr_class = np.bincount(real.astype(np.int64))
    latent = []
    predictions = []
    correct = []
    for i in range(nr_class.shape[0]):
        start = np.sum(nr_class[:i])
        latent.append(latents[start:start+nr_class[i]])
        predictions.append(pred[start:start+nr_class[i]])
        correct.append(pred[start:start+nr_class[i]] == i)

    return latent, predictions, correct


def use_dataset(subset, train_idx, test_idx, val_idx, classes, pred, latents):
    if subset == 'train':
        return classes[train_idx], pred[train_idx], latents[train_idx]
    if subset == 'test':
        return classes[test_idx], pred[test_idx], latents[test_idx]
    if subset == 'val':
        return classes[val_idx], pred[val_idx], latents[val_idx]
    return classes, pred, latents

def confusion_matrix(data_preds):
    classes = len(data_preds)
    ret = np.zeros((classes, classes))
    for i in range(classes):
        if data_preds[i].shape[0] == 0:
            row = np.bincount(data_preds[i].astype(np.int64), minlength=classes)
            ret[i] = row / np.sum(row)

    return ret

def acc_vs_model(acc, model_acc, augment=False):
    if augment:
        model_aug = model_acc - np.min(model_acc)
        model_aug = model_aug/np.max(model_aug)
        acc_aug = acc - np.min(acc)
        acc_aug = acc / np.max(acc)
    else:
        model_aug = model_acc
        acc_aug = acc
        
    fig, ax = plt.subplots(1,1,figsize=(30,10))
    fig.set_facecolor('white')

    ax.set_title('Accuracy comparison between dataset and generated images')
    ax.plot(acc_aug,label='gen_acc')
    ax.plot(model_aug,label='dataset_acc')
    ax.set_xlabel('class')
    ax.set_ylabel('accuracy')
    ax.grid()
    ax.legend()

def show_class(data_latent, data_pred, class_nr, batch, gen):
    latent_code = data_latent[class_nr]
    pred = data_pred[class_nr]

    if batch*5 > latent_code.shape[0]:
        print(f'Only {latent_code.shape[0] // 5 + 1} batches exists')
        return

    path = './data/beetles/images/'
    folders = os.listdir(path)
    folders.sort()

    image_folder = os.listdir(path + folders[class_nr] + '/')
    image_folder.sort()

    image_files = image_folder[batch*5:(batch+1)*5]
    dataset_imgs = [Image.open(path + folders[class_nr] + '/'+ img).resize((448,224)) for img in image_files]
    

    tensors = torch.tensor(latent_code[batch*5:(batch+1)*5], dtype=torch.float)
    tensors = tensors.reshape((tensors.shape[0], tensors.shape[1], 1, 1)).to('cuda')

    gen_img = (gen(tensors).detach().cpu().numpy().transpose(0,2,3,1) + 1)/2

    fig, ax = plt.subplots(gen_img.shape[0],2,figsize=(20,6*gen_img.shape[0]))
    fig.set_facecolor('lightgray')
    if gen_img.shape[0] != 1:
        for i in range(gen_img.shape[0]):
            ax[i,0].imshow(dataset_imgs[i])
            ax[i,0].axis('off')
            ax[i,0].set_title(f'Class {class_nr} file: {image_files[i]}', fontsize=20)
            ax[i,1].imshow(gen_img[i])
            ax[i,1].axis('off')
            if pred[batch*5+i] == class_nr:
                ax[i,1].set_title('Correct', c='b', fontsize=24)
            else:
                ax[i,1].set_title('Incorrect', c='r', fontsize=24)
    else:
        ax[0].imshow(dataset_imgs[0])
        ax[0].axis('off')
        ax[0].set_title(f'Class {class_nr} file: {image_files[0]}', fontsize=20)
        ax[1].imshow(gen_img[0])
        ax[1].axis('off')
        if pred[batch*5] == class_nr:
            ax[1].set_title('Correct', c='b', fontsize=24)
        else:
            ax[1].set_title('Incorrect', c='r', fontsize=24)
    plt.tight_layout()

def visual_insection(gen, enc, model, file_name=None):

    class_0_file = 'data/beetles/images/achenium_humile/_0189_0.jpg'
    class_21_file = 'data/beetles/images/emus_hirtus/_0486_5.jpg'
    class_48_file = 'data/beetles/images/lathrobium_fulvipenne/_0149_5.jpg'
    class_94_file = 'data/beetles/images/philonthus_concinnus/_0370_3.jpg'
    class_153_file = 'data/beetles/images/quedius_maurorufus/_0541_21.jpg'

    class_0_img = Image.open(class_0_file)
    class_21_img = Image.open(class_21_file)
    class_48_img = Image.open(class_48_file)
    class_94_img = Image.open(class_94_file)
    class_153_img = Image.open(class_153_file)

    images = [class_0_img, class_21_img, class_48_img, class_94_img, class_153_img]

    transformation = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 448)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(BEETLE_DATASET['mean'], BEETLE_DATASET['std'])])

    tensor_list = [transformation(img) for img in images]

    tensors = torch.stack(tensor_list).to('cuda')

    latent_code = enc(tensors)

    latent_code = latent_code.view(latent_code.shape[0],latent_code.shape[1],1,1 ).to('cuda')

    mean = torch.tensor([0.8442649, 0.82529384, 0.82333773]).reshape(3,1,1).to('cuda')
    std = torch.tensor([0.28980458, 0.32252666, 0.3240354]).reshape(3,1,1).to('cuda')

    
    imgs = gen(latent_code)
    res = model((((imgs+1)/2) - mean)/std).detach().cpu().numpy()

    clas = np.argmax(res, 1)
    print(clas)

    gen_imgs = imgs.cpu().detach().numpy().transpose(0,2,3,1)
    gen_imgs = (gen_imgs+1)/2
    
    fig, ax = plt.subplots(5,2,figsize=(20,30))
    fig.set_facecolor('white')

    ax[0,0].imshow(class_0_img)
    ax[0,0].axis('off')
    ax[0,0].set_title('Dataset class 0')
    ax[1,0].imshow(class_21_img)
    ax[1,0].axis('off')
    ax[1,0].set_title('Dataset class 21')
    ax[2,0].imshow(class_48_img)
    ax[2,0].axis('off')
    ax[2,0].set_title('Dataset class 48')
    ax[3,0].imshow(class_94_img)
    ax[3,0].axis('off')
    ax[3,0].set_title('Dataset class 94')
    ax[4,0].imshow(class_153_img)
    ax[4,0].axis('off')
    ax[4,0].set_title('Dataset class 153')

    ax[0,1].imshow(gen_imgs[0])
    ax[0,1].axis('off')
    ax[0,1].set_title('Generated class 0')
    ax[1,1].imshow(gen_imgs[1])
    ax[1,1].axis('off')
    ax[1,1].set_title('Generated class 21')
    ax[2,1].imshow(gen_imgs[2])
    ax[2,1].axis('off')
    ax[2,1].set_title('Generated class 48')
    ax[3,1].imshow(gen_imgs[3])
    ax[3,1].axis('off')
    ax[3,1].set_title('Generated class 94')
    ax[4,1].imshow(gen_imgs[4])
    ax[4,1].axis('off')
    ax[4,1].set_title('Generated class 153')
    
    if file_name is not None: 
        fig.savefig(file_name + '.png')

    plt.tight_layout















# ------------------ Aug functions ----------------
def get_aug_function(name):
    if name == 'soft_35':
        return norm_35
    if name == 'soft_75':
        return norm_75
    
    return lambda x : x


def norm_35(y):
    num = torch.exp(y.reshape(-1,y.shape[1])*0.35)
    denom = torch.sum(num, 1).reshape(-1,1)
    return (num/denom).reshape(-1,y.shape[1],1,1)

def norm_75(y):
    num = torch.exp(y.reshape(-1,y.shape[1])*0.75)
    denom = torch.sum(num, 1).reshape(-1,1)
    return (num/denom).reshape(-1,y.shape[1],1,1)*10