import os
import argparse
from utils import configure

class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--debug', action='store_true', default=True)
        self.parser.add_argument('--gpu_id', type=str, default='3')

        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--color_space', type=str, default='HSV')
        self.parser.add_argument('--dataset_dir', type=str, default='./dataset')
        self.parser.add_argument('--dataset_format', type=str, default='png')
        self.parser.add_argument('--dataset_name', type=str, default='cityscape')
        self.parser.add_argument('--image_height', type=int, default=1024)
        self.parser.add_argument('--input_channel', type=int, default=35)  # For cityscape dataset whose label numbers are 35.
        self.parser.add_argument('--output_channel', type=int, default=3)  # RGB
        self.parser.add_argument('--max_ch', type=int, default=2**9)
        self.parser.add_argument('--n_df', type=int, default=64, help='# of channels for the first layer in D')
        self.parser.add_argument('--n_gf', type=int, default=64, help='# of channels for the first layer in G')
        self.parser.add_argument('--n_workers', type=int, default=2, help='# of threads for loading data')
        self.parser.add_argument('--patch_size', type=int, default=70, help='effective receptive field of D')

    def parse(self):
        opt = self.parser.parse_args()
        model_name = 'Patch_size_{}'.format(str(opt.patch_size))
        if opt.is_train:
            opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, model_name)
            os.makedirs(opt.checkpoints_dir) if not os.path.isdir(opt.checkpoints_dir) else None
            opt.dataset_dir = os.path.join(opt.dataset_dir, opt.dataset_name)
            assert os.path.isdir(opt.dataset_dir), print('There is no {} directory.'.format(opt.dataset_dir))

            log_path = os.path.join(opt.checkpoints_dir, 'Model', 'opt.txt')

            if os.path.isfile(log_path) and not opt.debug:
                permission = input(
                    "{} log already exists. Do you really want to overwrite this log? Y/N. : ".format(
                        model_name + '/opt'))
                if permission == 'Y':
                    pass

                else:
                    raise NotImplementedError("Please check {}".format(log_path))

            args = vars(opt)
            with open(log_path, 'wt') as log:
                log.write('-' * 50 + 'Options' + '-' * 50 + '\n')
                print('-' * 50 + 'Options' + '-' * 50)
                for k, v in sorted(args.items()):
                    log.write('{}: {}\n'.format(str(k), str(v)))
                    print("{}: {}".format(str(k), str(v)))
                log.write('-' * 50 + 'End' + '-' * 50)
                print('-' * 50 + 'End' + '-' * 50)
                log.close()

        else:
            opt.test_image_dir = os.path.join(opt.results)

        if opt.debug:
            opt.report_freq = 1
            opt.save_freq = 1

        self.opt = opt
        return self.opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--display_freq', type=int, default=500)
        self.parser.add_argument('--flip', action='store_true', default=True)
        self.parser.add_argument('--is_train', action='store_true', default=True)
        self.parser.add_argument('--L1_lambda', type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_epoch', type=int, default=200)
        self.parser.add_argument('--report_freq', type=int, default=5)
        self.parser.add_argument('--save_freq', type=int, default=20)
        self.parser.add_argument('--shuffle', action='store_true', default=True)


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        self.parser.add_argument('--is_train', action='store_true', default=False)
        self.parser.add_argument('--shuffle', action='store_true', default=False)

