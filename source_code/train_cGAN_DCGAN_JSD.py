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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import DCGAN_cGAN
from dataloader import ICLEVRLoader
from evaluator import evaluation_model, test

# Set random seed for reproducibility
#manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# -

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Checkpoint path
ckp_path_G = './models/cGAN_DCGAN_JSD/netG/'
ckp_path_D = './models/cGAN_DCGAN_JSD/netD/'
time_stamp = '0826_1026'

# Print & store settings
print_every = 50
store_every = 10
# # +
# Number of training epochs
num_epochs = 300

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
# Plus classes number for cGAN
nz = 100

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

# +
# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# -

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
    for i, data in enumerate(dataloader, 0):
        # Conditions of data
        cond = data[1].to(device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        optimizerD.zero_grad()
        
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch with condition through D
        output = netD(real_cpu,cond.detach()).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise,cond.detach())
        label.fill_(fake_label)
        
        # Classify all fake batch with D
        output = netD(fake.detach(),cond.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake,cond.detach()).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Store if the performance is good
        acc = test(netG, test_loader, nz=nz)
        if acc >= 0.8 :
            torch.save(netG, os.path.join(ckp_path_G, 'iter_'+str(i)+'_'+time_stamp+"_great"))
            torch.save(netD, os.path.join(ckp_path_D, 'iter_'+str(i)+'_'+time_stamp+"_great"))
            print('Store model! Acc = ',acc)
        elif acc >= 0.7:
            torch.save(netG, os.path.join(ckp_path_G, 'iter_'+str(i)+'_'+time_stamp+"_good"))
            torch.save(netD, os.path.join(ckp_path_D, 'iter_'+str(i)+'_'+time_stamp+"_good"))
            print('Store model! Acc = ',acc)
            

        # Output training stats
        if i % print_every == 0 or i == (len(dataloader)-1):
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('Acc = ', acc)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % store_every == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
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
            
            if ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                torch.save(netG, os.path.join(ckp_path_G, 'iter_'+str(iter)+'_'+time_stamp))
                torch.save(netD, os.path.join(ckp_path_D, 'iter_'+str(iter)+'_'+time_stamp))

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
