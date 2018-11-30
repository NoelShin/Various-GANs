import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
from utils import *
from option import TrainOption
from data_pipeline import CustomDataset
from network import Generator, Discriminator
import datetime


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    opt = TrainOption().parse()
    if opt.gpu_id != '-1':
        assert torch.cuda.is_available(), print("This server does not have CUDA devices")
        USE_CUDA = True
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    else:
        USE_CUDA = False

    dataset = CustomDataset(opt)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                              shuffle=opt.shuffle, num_workers=opt.n_workers)

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

    grid_shape = get_grid_shape(opt)
    valid_label = torch.Tensor(*grid_shape).fill_(1.0)
    fake_label = torch.Tensor(*grid_shape).fill_(0.0)

    if USE_CUDA:
        valid_label = valid_label.cuda()
        fake_label = fake_label.cuda()

    total_step = 0
    for epoch in range(opt.n_epoch):
        for i, (input, real) in data_loader:
            total_step += 1
            if USE_CUDA:
                input = input.cuda()
                real = real.cuda()

            D_optim.zero_grad()
            real_loss = GAN_loss(D(real), valid_label)
            print("real_loss shape", real_loss.detach().shape)

            fake = G(input)
            fake_loss = GAN_loss(D(fake.detach), fake_label)
            print("fake_loss shape", fake_loss.detach().shape)

            D_loss = (real_loss + fake_loss)*0.5
            D_loss.backward()
            D_optim.step()

            G_optim.zero_grad()
            G_loss = opt.L1_lambda*L1_loss(fake, real) + GAN_loss(D(fake), valid_label)
            G_loss.backward()
            G_optim.step()

            if total_step % opt.report_freq == 0:
                print("Epoch: {} [{}/{}, {:.{prec}}%], G_loss: {:.{prec}}, D_loss: {:.{prec}}"
                         .format(epoch, i, len(data_loader), float(i/len(data_loader))*100,
                                 G_loss.item(), D_loss.item(), prec=4))

            if total_step % opt.display_freq == 0:
                save_image(fake, os.path.join(opt.checkpoint_dir, 'fake_{}.png'.format(total_step)))
                save_image(real, os.path.join(opt.checkpoint_dir, 'real_{}.png'.format(total_step)))

            if total_step % opt.save_freq == 0:
                torch.save(G.state_dict(), os.path.join(opt.checkpoint_dir, 'G_{}'.format(total_step)))
                torch.save(D.state_dict(), os.path.join(opt.checkpoint_dir, 'D_{}'.format(total_step)))

            if opt.debug:
                break

        if opt.debug:
            break

    print("Total time taken: ", datetime.datetime.now() - start_time)

