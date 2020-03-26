from __future__ import print_function
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython.display import HTML
import yaml


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def print_images(tensor):
    # The tensor must have shape [batch,3,height,width]
    fig = plt.figure(figsize=(tensor.shape[2], tensor.shape[3]))
    plt.axis("off")
    # Need to de-normalize the image
    ims = [[plt.imshow((np.transpose(i, (1, 2, 0))+1)*0.5, animated=True)] for i in tensor]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

def print_losses(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def obtain_output_model(input,model):
    """
    Input is a tensor of shape [batch,chanels,height,width] and returns another tensor with the ouput dimensions of
    the model
    """
    return model(input).detach().cpu()[0]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(param["train"]["size_input"], 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(64, param["dataset"]["n_channels"], 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(param["dataset"]["n_channels"], 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



if __name__ == "__main__":
    with open("celebA_opt.yaml", 'r') as stream:
        try:
            param = yaml.safe_load(stream)
        except:
            print("Did not found the yaml file")

    # Set random seed for reproducibility
    manualSeed = param["train"]["seed"]

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataset = dset.ImageFolder(root=param["dataset"]["path"],
                               transform=transforms.Compose([
                                   transforms.Resize(param["dataset"]["img_size"]),
                                   transforms.CenterCrop(param["dataset"]["img_size"]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=param["train"]["batch_size"],
                                             shuffle=True, num_workers=param["train"]["workers"])

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and param["train"]["ngpu"] > 0) else "cpu")

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Print models
    print(netG)
    print(netD)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=param["optimizer"]["initial_lr"],
                            betas=(param["optimizer"]["beta"], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=param["optimizer"]["initial_lr"],
                            betas=(param["optimizer"]["beta"], 0.999))

    if param["model"]["load"]:
        checkpoint = torch.load("checkpoint/model.pt")
        netG.load_state_dict(checkpoint['modelG_state_dict'])
        netD.load_state_dict(checkpoint['modelD_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

    else:
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netG.apply(weights_init)
        netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(param["evaluate"]["num_samples"], param["train"]["size_input"], 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0


    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(param["train"]["epochs"]):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, param["train"]["size_input"], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
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
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, param["train"]["epochs"], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % param["train"]["period_check"] == 0) or ((epoch == param["train"]["epochs"] - 1)
                                                                 and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()[0]
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        torch.save({
            'modelG_state_dict': netG.state_dict(),
            'modelD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
        }, "checkpoint/model-{}.pt".format(epoch))

        print_losses(G_losses, D_losses)

    # Save the generator/degenerator to share it with C++
    smG = torch.jit.script(netG)
    smD = torch.jit.script(netD)
    smG.save("models/generator.pt")
    smD.save("models/discriminator.pt")
