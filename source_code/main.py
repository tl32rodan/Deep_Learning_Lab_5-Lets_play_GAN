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

from dataloader import ICLEVRLoader
from evaluator import evaluation_model
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# -

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Checkpoint path
ckp_path_G = './models/netG/'
ckp_path_D = './models/netD/'
time_stamp = '0822_2340'

# Print & store settings
print_every = 50
store_every = 500
# # +
# Number of training epochs
num_epochs = 3000

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

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
# Plus classes number for cGAN
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Size of test set
test_size = 32
# -

eval_model = evaluation_model()

train_set = ICLEVRLoader(root=dataroot, mode='train')
dataloader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
test_set = ICLEVRLoader(root=dataroot, mode='test')
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=test_size)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, num_conds=24):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.num_conds = num_conds
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+self.num_conds, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        

    def forward(self, input, cond):
        cond = cond.view(-1,self.num_conds,1,1)
        #int('input.shape = ',input.shape)
        #print('cond.shape = ',cond.shape)
        x = torch.cat((input,cond),1)
        return self.main(x)

# +
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
#print(netG)
# -

class Discriminator(nn.Module):
    def __init__(self, ngpu, num_conds=24):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.num_conds = num_conds
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.extend_cond = nn.Sequential(
            nn.Linear(self.num_conds, 64*64, bias=False),
            nn.ReLU()
        )

    def forward(self, input, cond):
        cond = self.extend_cond(cond).view(-1,1,64,64)
        #print('input.shape = ',input.shape)
        #print('cond.shape = ',cond.shape)
        x = torch.cat((input,cond),1)
        return self.main(x)


# +
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

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

def test(netG, test_loader):
    avg_acc = 0.
    n = 0
    with torch.no_grad():
        for cond in test_loader:
            fixed_noise = torch.randn(len(cond), nz, 1, 1, device=device)
            fake = netG(fixed_noise,cond.to(device))
            acc = eval_model.eval(images=fake.to(device), labels=cond.to(device))
            avg_acc += acc*len(cond)
            n += len(cond)
    avg_acc = avg_acc/n
    return avg_acc


# +
# Training Loop


# +

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
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
        netD.zero_grad()
        
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
        netG.zero_grad()
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
        acc = test(netG, test_loader)
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
            
            if (iters % (store_every*10) == 0):
            
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
plt.show()

