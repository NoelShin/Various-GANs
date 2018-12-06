import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

from data_pipeline import CustomDataset
from network import Generator
from option import TestOption
from utils import *

import datetime

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    opt = TestOption().parse()
    USE_CUDA = True if torch.cuda.is_available() else False

    dataset = CustomDataset(opt)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                              shuffle=opt.is_train, num_workers=opt.n_workers)

    G = Generator(opt)
    if opt.color_space != 'RGB':
        G.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'Patch_size_70_{}'.format(opt.color_space), '50000_G.pt')))

    elif opt.color_space == 'RGB':
        if opt.patch_size == 70:
            G.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'Patch_size_70_{}'.format(opt.color_space), '50000_G.pt')))
        else:
            G.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'Patch_size_{}'.format(opt.patch_size), '50000_G.pt')))

    print(G)

    if USE_CUDA:
        G = G.cuda()

    for i, input in enumerate(data_loader):
        if USE_CUDA:
            input = input.cuda()
        fake = G(input)
        save_image(fake.detach(), os.path.join(opt.result_dir, 'Result_{}'.format(i)))
        print('{}/{} [{:.{prec}}%] has done.'.format(i+1, len(data_loader), float((i+1)/len(data_loader)*100), prec=4))

    print("Total time taken: ", datetime.datetime.now() - start_time)






