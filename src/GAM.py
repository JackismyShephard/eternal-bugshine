import time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import IPython.display as display
from pathlib import Path
import json
from torchvision.utils import make_grid
import os
display.set_matplotlib_formats('svg')

from .utils.visual import show_img, tensor_to_image

from torch.optim import Adam

# Setup device to use gpu if available
device = "cuda:0"


import sys

def norm_img(img, mean, std):
    res = (img + 1) / 2
    return (res - mean) / std

def latent_std(y):
    return y

def GAM_fit_save(config):
    gen, enc, stats = GAM_fit(config)
    name = config['name']
    folder_path = './output/GANs/' + name + '/'
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    #if not os.path.exists(folder_path):
    #    os.mkdir(folder_path)
    simple_config = {
        'name' : config['name'],
        'comp_layer' : config['comp_layer'], 
        'static' : config['static'],
        'latent_dim': config['latent_dim'],
        'latent_noise': config['latent_noise'],
        'num_epochs' : config['num_epochs'],
        'lrs' : config['lrs'],
        'lambdas' : config['lambdas'],
        'lr_decay_gamma' : config['lr_decay_gamma'],
        'latent_aug_name' : config['latent_aug_name'] 
    }
    json_path = folder_path + name + '_config.json'
    with open(json_path, 'w') as json_file:
        json.dump(simple_config, indent = 4, fp = json_file)
    torch.save(gen.state_dict(), folder_path + name + '_gen.pt')
    torch.save(enc.state_dict(), folder_path + name + '_enc.pt')
    np.save(folder_path + name + '_stats.npy', np.array(stats))
    
    
    

def GAM_fit(config):
    """Training for GAM setup, assumes models are on the gpu"""
    # Calculate execution time
    since = time.time()

    dataloaders = config['dataloaders']
    datasizes = config['datasizes']
    if config['split_data']:
        # Split datasets
        dataloader = dataloaders['train']
        dataloader_val = dataloaders['val']
        dataloader_test = dataloaders['test']

        datasize = datasizes['train']
        datasize_val = datasizes['val']
        datasize_test = datasizes['test']
    else:
        dataloader = dataloaders
        datasize = datasizes

    gen = config['generator']
    disc = config['discriminator']
    enc = config['encoder']
    comp = config['comparator']

    if config['latent_aug_name'] is None:
        latent_aug = lambda x : x
    else:
        latent_aug = config['latent_aug_name']


    lambdas = config['lambdas']
    num_epochs = config['num_epochs']
    
    # Setup statistics
    disc_losses = []
    img_losses = []
    adv_losses = []
    feat_losses = []
    gen_losses = []
    val_acc = []
    test_acc = []

    # Setup normalization 
    in_channels = config['latent_dim']
    mean = torch.tensor([0.8442649, 0.82529384, 0.82333773]).reshape(3,1,1).to(device)
    std = torch.tensor([0.28980458, 0.32252666, 0.3240354]).reshape(3,1,1).to(device)

    # Setup optimizers
    optim_g = Adam(gen.parameters(),  lr=config['lrs'][0], betas=(config['lrs'][1], config['lrs'][2]))
    optim_d = Adam(disc.parameters(),  lr=config['lrs'][0], betas=(config['lrs'][1], config['lrs'][2]))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config['lr_decay_gamma'])
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config['lr_decay_gamma'])

    # If the encoder also has to be trained
    if not config['static']:
        optim_e = Adam(enc.parameters(),  lr=config['lrs'][0], betas=(config['lrs'][1], config['lrs'][2]))
        scheduler_e = torch.optim.lr_scheduler.ExponentialLR(optim_e, gamma=config['lr_decay_gamma'])

    # Loss function for gen and disc
    loss_f = torch.nn.BCEWithLogitsLoss(reduction='sum')


    # Calculating images takes up precious vram
    if config['display_images']:

        # Setup latent codes to generate images to show at each epoch
        iterable = iter(dataloader)
        epoch_imgs, epoch_latents = next(iterable)
        epoch_imgs = epoch_imgs.to(device)

        # If the encoder is not static we need to calculate a new latent code at each epoch
        if config['static']:

            # no need to calculate the autograd and use vram for it
            with torch.no_grad():
                epoch_latents = latent_aug(enc(norm_img(epoch_imgs, mean, std)).detach()).reshape(-1, in_channels, 1,1)

    # Setup mixed precision 
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        # Running loss for each epoch
        disc_l_running = 0.0
        img_l_running = 0.0
        adv_l_running = 0.0
        feat_l_running = 0.0
        gen_l_running = 0.0

        # Iterate over data.
        for X, y in dataloader:
            X = X.to(device)

            # Setup autograd for discriminator
            optim_d.zero_grad()

            # Calculate autograd and loss with mixed precision
            with torch.cuda.amp.autocast():

                # Same as calculating display latent codes
                if config['static']:     
                    with torch.no_grad():
                        y = latent_aug(enc(norm_img(X, mean, std)))
                        if config['latent_noise'] is not None:
                            y.data = y.data + torch.normal(config['latent_noise'][0],config['latent_noise'][1], y.shape).to(device)
                        y.data = y.reshape((-1, in_channels, 1, 1)).data

                # If the encoder needs to be trained we need autograd 
                else:
                    y = latent_aug(enc(norm_img(X, mean, std))).view(X.shape[0],in_channels,1,1)

                # Calculate discriminator loss

                gen_x = gen(y)

                disc_real = disc(X)
                # Detach to stop autograd at this value
                disc_fake = disc(gen_x.detach())
                
                ones = torch.ones(disc_fake.shape, device='cuda')
                zeros = torch.zeros(disc_fake.shape, device='cuda')

                loss_adv = loss_f(disc_fake,ones)
                loss_disc = (loss_f(disc_real, ones) + loss_f(disc_fake, zeros))/2

            # Only update if ratio is not to small
            if loss_disc / loss_adv > 0.1:
                # use the mixed precission scaler for the actual update
                scaler.scale(loss_disc).backward()
                scaler.step(optim_d)
                scaler.update()


            # Setup autograd for generator
            optim_g.zero_grad()

            if not config['static']:
                optim_e.zero_grad()

            # Calculate autograd and loss with mixed precision
            with torch.cuda.amp.autocast():
                disc_fake = disc(gen_x)
                comp_real = comp(norm_img(X,mean,std))
                comp_fake = comp(norm_img(gen_x,mean,std))        

                loss_img = ((gen_x - X)**2).sum()

                loss_adv = loss_f(disc_fake,ones.reshape(disc_fake.shape))

                loss_feat = ((comp_fake - comp_real)**2).sum()
                
                loss_gen = lambdas[0] * loss_adv + \
                            lambdas[1] * loss_img + \
                            lambdas[2] * loss_feat 

            # Update generator and encoder
            if config['static']:
                scaler.scale(loss_gen).backward()
                scaler.step(optim_g)
                scaler.update()
            else:
                # retain_graph is need if both needs to be trained
                scaler.scale(loss_gen).backward(retain_graph=True)
                scaler.step(optim_g)
                scaler.update()

                # use the same loss for both the generator and the encoder
                scaler.step(optim_e)
                scaler.update()




            # Update loss statistics
            disc_l_running += loss_disc.cpu().detach().item()
            img_l_running += loss_img.cpu().detach().item()
            adv_l_running += loss_adv.cpu().detach().item()
            feat_l_running += loss_feat.cpu().detach().item()
            gen_l_running += loss_gen.cpu().detach().item()
        
        # Dont know if this makes a difference
        optim_d.zero_grad()
        optim_g.zero_grad()
        scheduler_d.step()
        scheduler_g.step()
        if not config['static']:
            scheduler_e.step()
            optim_e.zero_grad()

        # Save loss for each epoch
        disc_losses.append(disc_l_running / datasize)
        img_losses.append(img_l_running / datasize)
        adv_losses.append(adv_l_running / datasize)
        feat_losses.append(feat_l_running / datasize)
        gen_losses.append(gen_l_running / datasize)

        display_images = config['display_images']
        print_loss = config['print_loss']
        acc = config['calc_acc']

        if acc:
            test_acc_temp = torch.tensor([0], device='cuda')
            val_acc_temp = torch.tensor([0], device='cuda')

            for X, y in dataloader_test:
                X = X.to('cuda')
                y = y.to('cuda')
                with torch.no_grad():
                    lat = enc(X).view(X.shape[0],in_channels,1,1)
                    res = comp._forward_impl(norm_img(gen(lat), mean, std),False)
                    test_acc_temp += torch.sum(torch.argmax(res, 1) == y)

            for X, y in dataloader_val:
                X = X.to('cuda')
                y = y.to('cuda')
                with torch.no_grad():
                    lat = enc(X).view(X.shape[0],in_channels,1,1)
                    res = comp._forward_impl(norm_img(gen(lat), mean, std),False)
                    val_acc_temp += torch.sum(torch.argmax(res, 1) == y)
            test_acc.append(test_acc_temp.cpu().item()/datasize_test) 
            val_acc.append(val_acc_temp.cpu().item()/datasize_val) 



        # Show images and statistics
        display.clear_output(wait=True)
        if print_loss:
            elapsed_time = time.time()
            print(f'epoch {epoch + 1} of {num_epochs}: average epoch time {int((elapsed_time-since)/(epoch+1))}s')
            print(f'discriminator loss: {disc_l_running / datasize}')
            print(f'image loss: {(lambdas[1]*img_l_running )/ datasize}')
            print(f'adversarial loss: {(lambdas[0]*adv_l_running )/ datasize}')
            print(f'feature loss: {(lambdas[2]*feat_l_running ) / datasize}')
            print(f'generator loss: {gen_l_running / datasize}')

        if display_images or acc:
            num_graphs = 1 + display_images + acc
            fig, ax = plt.subplots(num_graphs,1, figsize=(10,num_graphs*5))
            fig.set_facecolor('white')

            loss_idx = display_images + acc
            acc_idx = int(display_images)

            if display_images:
                with torch.no_grad():
                    if not config['static']:
                        epoch_latents = latent_aug(enc(norm_img(epoch_imgs, mean, std))).view(epoch_imgs.shape[0],in_channels,1,1)
                    gen_xs = (gen(epoch_latents)+1)/2

                image_grid = tensor_to_image(make_grid(gen_xs, nrow=int(gen_xs.shape[0]/4)))
                ax[0].imshow(image_grid)

            if acc:
                print(f'test accuracy: {test_acc[-1]}')
                print(f'validation accuracy: {val_acc[-1]}')
                ax[acc_idx].plot(test_acc, label='test accurcacy')
                ax[acc_idx].plot(val_acc, label='validation accurcacy')
                ax[acc_idx].legend()
                ax[acc_idx].grid()

            ax[loss_idx].plot(disc_losses, label='discriminator loss')
            ax[loss_idx].plot(lambdas[1]*np.array(img_losses), label='image space loss')
            ax[loss_idx].plot(lambdas[0]*np.array(adv_losses), label='adversarial loss')
            ax[loss_idx].plot(lambdas[2]*np.array(feat_losses), label='feature space loss')
            ax[loss_idx].plot(gen_losses, label='generator loss')
            ax[loss_idx].legend()
            ax[loss_idx].grid()
    
        else:
            fig, ax = plt.subplots(1,1, figsize=(10,5))
            fig.set_facecolor('white')
            ax.plot(disc_losses, label='discriminator loss')
            ax.plot(lambdas[1]*np.array(img_losses), label='image space loss')
            ax.plot(lambdas[0]*np.array(adv_losses), label='adversarial loss')
            ax.plot(lambdas[2]*np.array(feat_losses), label='feature space loss')
            ax.plot(gen_losses, label='generator loss')
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()
        plt.close()



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    if acc:
        return gen, enc, [disc_losses, img_losses, adv_losses, feat_losses, gen_losses, test_acc, val_acc]
    return gen, enc, [disc_losses, img_losses, adv_losses, feat_losses, gen_losses]
