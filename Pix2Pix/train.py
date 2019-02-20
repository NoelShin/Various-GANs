import os
import torch
import torch.nn as nn
from utils import *
from option import TrainOption
from pipeline import CustomDataset
from networks import Generator, Discriminator
import datetime


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    opt = TrainOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    if opt.gpu_id != '-1':
        assert torch.cuda.is_available(), print("This server does not have CUDA devices")
        USE_CUDA = True
        device = torch.device('cuda', 0)
    else:
        USE_CUDA = False

    dataset = CustomDataset(opt)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                              shuffle=opt.shuffle, num_workers=opt.n_workers)

    print(len(data_loader))

    G = Generator(opt)
    D = Discriminator(opt)

    G.apply(weight_init)
    D.apply(weight_init)

    print(G)
    print(D)

    if USE_CUDA:
        G = G.cuda()
        D = D.cuda()

    G_optim = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    D_optim = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    GAN_loss = nn.BCELoss()
    L1_loss = nn.L1Loss()

    total_step = 0
    for epoch in range(opt.n_epoch):
        epoch += 1
        for i, (input, real) in enumerate(data_loader):
            total_step += 1
            if USE_CUDA:
                input = input.cuda(device)
                real = real.cuda(device)

            valid_D = D(torch.cat([input, real], dim=1))
            valid_grid = torch.ones_like(valid_D)

            fake = G(input)
            fake_D = D(torch.cat([input, fake.detach()], dim=1))
            fake_grid = torch.zeros_like(fake_D)

            if USE_CUDA:
                valid_label = valid_grid.cuda(device)
                fake_label = fake_grid.cuda(device)

            D_optim.zero_grad()
            real_loss = GAN_loss(valid_D, valid_grid)
            fake_loss = GAN_loss(fake_D, fake_grid)

            D_loss = (real_loss + fake_loss) * 0.5
            D_loss.backward()
            D_optim.step()

            G_optim.zero_grad()
            G_loss = opt.L1_lambda * L1_loss(fake, real) + GAN_loss(D(torch.cat([input, fake], dim=1)), valid_grid)
            G_loss.backward()
            G_optim.step()

            if total_step % opt.report_freq == 0:
                print("Epoch: {} [{}/{}, {:.{prec}}%], G_loss: {:.{prec}}, D_loss: {:.{prec}}"
                      .format(epoch, i + 1, len(data_loader), (i + 1)/len(data_loader) * 100,
                              G_loss.item(), D_loss.item(), prec=4))

            if total_step % opt.display_freq == 0:
                save_image(fake.detach(), os.path.join(opt.checkpoints_dir, 'fake_{}.png'.format(epoch)))
                save_image(real, os.path.join(opt.checkpoints_dir, 'real_{}.png'.format(epoch)))

            if opt.debug:
                break

        if total_step % opt.save_freq == 0:
            torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir, 'G_{}'.format(epoch)))
            torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, 'D_{}'.format(epoch)))

        if opt.debug:
            break

    print("Total time taken: ", datetime.datetime.now() - start_time)

