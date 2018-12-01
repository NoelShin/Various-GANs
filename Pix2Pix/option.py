import os
import argparse


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--dataset_dir', type=str, default='./dataset/Claire')
        self.parser.add_argument('--dataset_format', type=str, default='png')
        self.parser.add_argument('--gpu_id', type=str, default='-1')
        self.parser.add_argument('--image_size', type=int, default=256)
        self.parser.add_argument('--input_channel', type=int, default=1)
        self.parser.add_argument('--output_channel', type=int, default=3)
        self.parser.add_argument('--n_gf', type=int, default=64)
        self.parser.add_argument('--n_df', type=int, default=64)
        self.parser.add_argument('--n_workers', type=int, default=2)


    def parse(self):
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        if not os.path.isdir(self.opt.checkpoint_dir):
            os.mkdir(self.opt.checkpoint_dir)

        if self.opt.debug == True:
            self.opt.report_freq = 1
            self.opt.save_freq = 1


        print("-"*50, "Option start", "-"*50)
        for k, v in sorted(args.items()):
            print("{}: {}".format(str(k), str(v)))
        print("-"*50, "End start", "-"*50)

        return self.opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--debug', action='store_true', default=True)

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.5)
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
        self.parser.add_argument('--display_freq', type=int, default=500)
        self.parser.add_argument('--flip', action='store_true', default=True)
        self.parser.add_argument('--is_train', action='store_true', default=True)
        self.parser.add_argument('--L1_lambda', type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_epoch', type=int, default=400)
        self.parser.add_argument('--patch_size', type=int, default=70)
        self.parser.add_argument('--report_freq', type=int, default=10)
        self.parser.add_argument('--save_freq', type=int, default=10000)
        self.parser.add_argument('--shuffle', action='store_true', default=True)


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        self.parser.add_argument('--is_train', action='store_true', default=False)

