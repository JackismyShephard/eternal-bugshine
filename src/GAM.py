import time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import IPython.display as display
from torchvision.utils import make_grid
display.set_matplotlib_formats('svg')

from .utils.visual import show_img, tensor_to_image

from torch.optim import Adam

# Setup device to use gpu if available
device = "cuda:0"

import sys
def GAM_fit(gen, disc, comp, norm_func_img, norm_func_latent, dataloader, lrs = [ 0.0002, 0.9, 0.999],  
            lambdas = [100, 2e-6, 0.01], num_epochs=200, enc = None, latent_noise = None, static_enc = True, display_images = True, print_loss = True, input_channels = 197):
    """Training for GAM setup, assumes models are on the gpu"""
    # Calculate execution time
    since = time.time()

    # Setup statistics
    dataset_size = len(dataloader) * dataloader.batch_size
    disc_losses = []
    img_losses = []
    adv_losses = []
    feat_losses = []
    gen_losses = []

    # Setup normalization 
    in_channels = input_channels
    mean = torch.tensor([0.8442649, 0.82529384, 0.82333773]).reshape(3,1,1).to(device)
    std = torch.tensor([0.28980458, 0.32252666, 0.3240354]).reshape(3,1,1).to(device)

    # Setup optimizers
    optim_g = Adam(gen.parameters(),  lr=lrs[0], betas=(lrs[1], lrs[2]))
    optim_d = Adam(disc.parameters(),  lr=lrs[0], betas=(lrs[1], lrs[2]))

    # If the encoder also has to be trained
    if not static_enc:
        optim_e = Adam(enc.parameters(),  lr=lrs[0], betas=(lrs[1], lrs[2]))

    # Loss function for gen and disc
    loss_f = torch.nn.BCEWithLogitsLoss(reduction='sum')

    # Calculating images takes up precious vram
    if display_images:

        # Setup latent codes to generate images to show at each epoch
        iterable = iter(dataloader)
        epoch_imgs, epoch_latents = next(iterable)
        epoch_imgs = epoch_imgs.to(device)

        # If the encoder is not static we need to calculate a new latent code at each epoch
        if static_enc:

            # no need to calculate the autograd and use vram for it
            with torch.no_grad():
                if enc is not None:
                    epoch_latents = norm_func_latent(enc(norm_func_img(epoch_imgs, mean, std)).detach())
                else :
                    epoch_latents = F.one_hot(epoch_latents, in_channels).float().to(device)        
            epoch_latents.data = epoch_latents.reshape(-1, in_channels, 1,1).data

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
                if static_enc:     
                    with torch.no_grad():
                        if enc is not None:
                            y = norm_func_latent(enc(norm_func_img(X, mean, std)))
                        else :
                            y = F.one_hot(y, in_channels).float().to(device)
                        if latent_noise is not None:
                            y.data = y.data + torch.normal(latent_noise[0],latent_noise[0], y.shape).to(device)
                        y.data = y.reshape((-1, in_channels, 1, 1)).data

                # If the encoder needs to be trained we need autograd 
                else:
                    y = norm_func_latent(enc(norm_func_img(X, mean, std))).view(X.shape[0],input_channels,1,1)

                # Calculate discriminator loss

                gen_x = gen(y)

                disc_real = disc(X)
                # Detach to stop autograd at this value
                disc_fake = disc(gen_x.detach())

                ones = torch.ones((X.shape[0],), device=X.device)
                zeros = torch.zeros((X.shape[0],), device=X.device)

                loss_adv = loss_f(disc_fake,ones.reshape(disc_fake.shape))
                loss_disc = (loss_f(disc_real, ones.reshape(disc_real.shape)) + loss_f(disc_fake, zeros.reshape(disc_fake.shape)))/2

            # Only update if ratio is not to small
            if loss_disc / loss_adv > 0.1:
                # use the mixed precission scaler for the actual update
                scaler.scale(loss_disc).backward()
                scaler.step(optim_d)
                scaler.update()


            # Setup autograd for generator
            optim_g.zero_grad()

            if not static_enc:
                optim_e.zero_grad()

            # Calculate autograd and loss with mixed precision
            with torch.cuda.amp.autocast():
                disc_fake = disc(gen_x)
                comp_real = comp(norm_func_img(X,mean,std))
                comp_fake = comp(norm_func_img(gen_x,mean,std))        

                loss_img = ((gen_x - X)**2).sum()

                loss_adv = loss_f(disc_fake,ones.reshape(disc_fake.shape))

                loss_feat = ((comp_fake - comp_real)**2).sum()
                
                loss_gen = lambdas[0] * loss_adv + \
                            lambdas[1] * loss_img + \
                            lambdas[2] * loss_feat 

            # Update generator and encoder
            if static_enc:
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
            

        # Save loss for each epoch
        disc_losses.append(disc_l_running / dataset_size)
        img_losses.append(img_l_running / dataset_size)
        adv_losses.append(adv_l_running / dataset_size)
        feat_losses.append(feat_l_running / dataset_size)
        gen_losses.append(gen_l_running / dataset_size)

        # Show images and statistics
        display.clear_output(wait=True)
        if print_loss:
            elapsed_time = time.time()
            print(f'epoch {epoch + 1} of {num_epochs}: average epoch time {int((elapsed_time-since)/(epoch+1))}s')
            print(f'discriminator loss: {disc_l_running / dataset_size}')
            print(f'image loss: {(lambdas[1]*img_l_running )/ dataset_size}')
            print(f'adversarial loss: {(lambdas[0]*adv_l_running )/ dataset_size}')
            print(f'feature loss: {(lambdas[2]*feat_l_running ) / dataset_size}')
            print(f'generator loss: {gen_l_running / dataset_size}')

        if display_images:
            fig, ax = plt.subplots(2,1, figsize=(10,10))
            fig.set_facecolor('white')

            with torch.no_grad():
                if not static_enc:
                    epoch_latents = norm_func_latent(enc(norm_func_img(epoch_imgs, mean, std))).view(epoch_imgs.shape[0],input_channels,1,1)
                gen_xs = (gen(epoch_latents)+1)/2

            image_grid = tensor_to_image(make_grid(gen_xs, nrow=int(gen_xs.shape[0]/4)))
            ax[0].imshow(image_grid)

            ax[1].plot(disc_losses, label='discriminator loss')
            ax[1].plot(lambdas[1]*np.array(img_losses), label='image space loss')
            ax[1].plot(lambdas[0]*np.array(adv_losses), label='adversarial loss')
            ax[1].plot(lambdas[2]*np.array(feat_losses), label='feature space loss')
            ax[1].plot(gen_losses, label='generator loss')
            ax[1].legend()
            ax[1].grid()
    
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
        
    return gen, enc, [disc_losses, img_losses, adv_losses, feat_losses, gen_losses]
