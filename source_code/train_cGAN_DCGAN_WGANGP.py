# +
from __future__ import print_function
# #%matplotlib inline
import torchvision.utils
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import DCGAN_cGAN
from dataloader import ICLEVRLoader
from evaluator import evaluation_model, test

# Set random seed for reproducibility
#manualSeed = 1998
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# -

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Checkpoint path
ckp_path_G = './models/cGAN_DCGAN_WGANGP/netG/'
ckp_path_D = './models/cGAN_DCGAN_WGANGP/netD/'
time_stamp = '0825_1420'

# Print & store settings
print_every = 10
store_every = 10
# # +
# Number of training epochs
num_epochs = 5000

# Root directory for dataset
dataroot = '../lab5_dataset/iclevr/'

# Number of objects' classes
num_classes = 24

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# +
# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# -

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# For WGAN and WGAN-GP, number of critic iters per gen iter
CRITIC_ITERS = 3

# Gradient penalty lambda hyperparameter
LAMBDA = 10

# Size of test set
test_size = 32


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



train_set = ICLEVRLoader(root=dataroot, mode='train')
dataloader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
test_set = ICLEVRLoader(root=dataroot, mode='test')
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=test_size)

# ### Create the generator

# +

netG = DCGAN_cGAN.Generator(ngpu, nc=nc, nz=nz, ngf=ngf).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
#print(netG)
# -

# ### Create the Discriminator

# +

netD = DCGAN_cGAN.Discriminator(ngpu, nc=nc, nz=nz, ndf=ndf).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
#print(netD)
# -

def inf_img(dataloader):
    i = 0
    while True:
        for data in dataloader:
            i+=1
            yield i, data


def calc_gradient_penalty(netD, real_data, fake_data, cond):
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = Variable(interpolates, requires_grad=True).to(device)

    disc_interpolates = netD(interpolates, cond)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates).to(device),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True,
                             )[0]
    
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


inf_loader = inf_img(dataloader)
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# ## Training Loop


# +
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
acc_list = []
iters = 1
avg_acc = 0.

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for iter_crtic in range(CRITIC_ITERS):
        # Get next data
        i, data = next(inf_loader)
        
        # Conditions of data
        cond = data[1].to(device)
        ############################
        # (1) Update D network
        ###########################
        ## Train with all-real batch
        optimizerD.zero_grad()
        
        # Format batch
        real = Variable(data[0]).to(device)
        b_size = real.size(0)
        # Forward pass real batch with condition through D
        D_real = netD(real,cond.detach())
        D_real = D_real.mean()
        

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        noise = Variable(noise)
        # Generate fake image batch with G
        fake = netG(noise,cond.detach())
        
        # Classify all fake batch with D
        D_fake = netD(fake,cond.detach())
        # Calculate D's loss on the all-fake batch
        D_fake = D_fake.mean()
        
        
        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real.data, fake.data, cond)
        
        # Add the gradients from the all-real and all-fake batches
        D_cost = D_fake - D_real + gradient_penalty
        D_cost.backward()
        Wasserstein_D = D_real - D_fake

        # Update D
        optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    for p in netD.parameters(): 
        p.requires_grad = False # to avoid computation
    optimizerG.zero_grad()
    # Since we just updated D, perform another forward pass of all-fake batch through D
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise,cond.detach())
    output = netD(fake,cond.detach()).view(-1)
    output = output.mean()
        
    G_cost = -output
    G_cost.backward()
    optimizerG.step()

    # Store if the performance is good
    acc = test(netG, test_loader, nz=nz)
    if acc >= 0.8 :
        torch.save(netG, os.path.join(ckp_path_G, 'iter_'+str(epoch)+'_'+time_stamp+"_great"))
        torch.save(netD, os.path.join(ckp_path_D, 'iter_'+str(epoch)+'_'+time_stamp+"_great"))
        print('Store model! Acc = ',acc)
    elif acc >= 0.7:
        torch.save(netG, os.path.join(ckp_path_G, 'iter_'+str(epoch)+'_'+time_stamp+"_good"))
        torch.save(netD, os.path.join(ckp_path_D, 'iter_'+str(epoch)+'_'+time_stamp+"_good"))
        print('Store model! Acc = ',acc)
            

    # Output training stats
    if epoch % print_every == 0 :
        print('[%d/%d]\tD_real = %.4f\t D_fake= %.4f\tD_cost= %.4f\tG_cost= %.4f\tWasserstein_D= %.4f'
                  % (epoch, num_epochs, D_real.item(), D_fake.item(), D_cost.item(), G_cost.item(),Wasserstein_D.item()))
        print('Acc = ', acc)

    # Save Losses for plotting later
    G_losses.append(G_cost.item())
    D_losses.append(D_cost.item())

    # Check how the generator is doing by saving G's output on fixed_noise
    if (iters % store_every == 0) or (epoch == num_epochs-1):
        with torch.no_grad():
            # Create batch of latent vectors that we will use to visualize
            #  the progression of the generator
            for cond in test_loader:
                cond = cond.to(device)
                fixed_noise = torch.randn(len(cond), nz, 1, 1, device=device)
                fake = netG(fixed_noise,cond.detach()).detach().cpu()
                break
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        acc_list.append(acc)
            
        if (epoch == num_epochs-1):
            torch.save(netG, os.path.join(ckp_path_G, 'iter_'+str(iters)+'_'+time_stamp))
            torch.save(netD, os.path.join(ckp_path_D, 'iter_'+str(iters)+'_'+time_stamp))

    iters += 1
# -

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('results/'+str(time_stamp))

torch.save(img_list,'lists/img_list_'+str(time_stamp))
torch.save(acc_list,'lists/acc_list_'+str(time_stamp))


