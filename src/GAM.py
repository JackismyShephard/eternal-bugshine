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

device = "cuda:0"

def GAM_fit(gen, disc, comp, norm_func_img, norm_func_latent, dataloader, lrs = [ 0.0002, 0.9, 0.999],  
            lambdas = [100, 2e-6, 0.01], num_epochs=200, display_images = True, enc = None):
    since = time.time()
    gen = gen.to(device)
    disc = disc.to(device)
    dataset_size = len(dataloader) * dataloader.batch_size
    optim_g = Adam(gen.parameters(),  lr=lrs[0], betas=(lrs[1], lrs[2]))
    optim_d = Adam(disc.parameters(),  lr=lrs[0], betas=(lrs[1], lrs[2]))

    m = torch.tensor([0.8442649, 0.82529384, 0.82333773])
    s = torch.tensor([0.28980458, 0.32252666, 0.3240354])
    mean_tensor = torch.zeros((3,224,448))
    std_tensor = torch.zeros((3,224,448))
    mean_tensor[0,:] = m[0]   
    mean_tensor[1,:] = m[1] 
    mean_tensor[2,:] = m[2] 
    std_tensor[0,:] = s[0]   
    std_tensor[1,:] = s[1] 
    std_tensor[2,:] = s[2] 

    mean_tensor = mean_tensor.to(device)
    std_tensor = std_tensor.to(device)

    disc_losses = []
    img_losses = []
    adv_losses = []
    feat_losses = []
    gen_losses = []

    iterable = iter(dataloader)
    data_imgs, data_latents = next(iterable)
    data_imgs = data_imgs.to(device)
    if enc is not None:
        data_latents = norm_func_latent(enc(   norm_func_img(data_imgs)  )).reshape((-1, 197, 1,1)).to(device)
    else :
        data_latents = F.one_hot(data_latents, 197).float()
        data_latents = data_latents.reshape((-1, 197, 1,1)).to(device)

    loss_f = torch.nn.BCEWithLogitsLoss(reduction='sum')

    for epoch in range(num_epochs):
        disc_l_running = 0.0
        img_l_running = 0.0
        adv_l_running = 0.0
        feat_l_running = 0.0
        gen_l_running = 0.0
        # Iterate over data.
        for X, y in dataloader:
            X = X.to(device)
            if enc is None:
                y = F.one_hot(y, 197).float()
                y = y.reshape((-1, 197, 1,1)).to(device)
            else:
                y = norm_func_latent(enc( norm_func_img(X)  )).reshape((-1, 197, 1,1)).to(device)


            optim_d.zero_grad()

            gen_x = gen(y)
            disc_real = disc(X)
            disc_fake = disc(gen_x.detach())

            ones = torch.ones((X.shape[0],), device=X.device)
            zeros = torch.zeros((X.shape[0],), device=X.device)

            #Update discriminator

            loss_adv = loss_f(disc_fake,ones.reshape(disc_fake.shape))
            loss_disc = (loss_f(disc_real, ones.reshape(disc_real.shape)) + loss_f(disc_fake, zeros.reshape(disc_fake.shape)))/2
            if loss_disc / loss_adv > 0.1:
                loss_disc.backward()

                optim_d.step()

            optim_g.zero_grad()

            disc_fake = disc(gen_x)
            comp_real = comp(norm_func_img(X,mean_tensor,std_tensor))
            comp_fake = comp(norm_func_img(gen_x,mean_tensor,std_tensor))

            loss_img = ((gen_x - X)**2).sum()

            loss_adv = loss_f(disc_fake,ones.reshape(disc_fake.shape))

            loss_feat = ((comp_fake - comp_real)**2).sum()
            
            loss_gen = lambdas[0] * loss_adv + \
                        lambdas[1] * loss_img + \
                        lambdas[2] * loss_feat 

            loss_gen.backward()

            optim_g.step()
            disc_l_running += loss_disc.cpu().detach().item()
            img_l_running += loss_img.cpu().detach().item()
            adv_l_running += loss_adv.cpu().detach().item()
            feat_l_running += loss_feat.cpu().detach().item()
            gen_l_running += loss_gen.cpu().detach().item()
            


        disc_losses.append(disc_l_running / dataset_size)
        img_losses.append(img_l_running / dataset_size)
        adv_losses.append(adv_l_running / dataset_size)
        feat_losses.append(feat_l_running / dataset_size)
        gen_losses.append(gen_l_running / dataset_size)



        gen_xs = (gen(data_latents)+1)/2
        
        image_grid = tensor_to_image(make_grid(gen_xs, nrow=int(gen_xs.shape[0]/4)))

        display.clear_output(wait=True)

        print(disc_l_running / dataset_size)
        print((lambdas[1]*img_l_running )/ dataset_size)
        print((lambdas[0]*adv_l_running )/ dataset_size)
        print((lambdas[2]*feat_l_running ) / dataset_size)
        print(gen_l_running / dataset_size)

        fig, ax = plt.subplots(2,1, figsize=(10,10))
        fig.set_facecolor('white')
        ax[0].plot(disc_losses, label='discriminator loss')
        ax[0].plot(lambdas[1]*np.array(img_losses), label='image space loss')
        ax[0].plot(lambdas[0]*np.array(adv_losses), label='adversarial loss')
        ax[0].plot(lambdas[2]*np.array(feat_losses), label='feature space loss')
        ax[0].plot(gen_losses, label='generator loss')
        ax[0].legend()
        ax[0].grid()

        ax[1].imshow(image_grid)

        plt.tight_layout()
        plt.show()
        plt.close()



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    return gen, [disc_losses, img_losses, adv_losses, feat_losses, gen_losses]
