"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import time

from fjcommon import config_parser
from fjcommon import functools_ext as ft
from fjcommon import timer
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

import vis.summarizable_module
from helpers import logdir_helpers
import pytorch_ext as pe
import vis.safe_summary_writer
from blueprints.multiscale_blueprint import MultiscaleBlueprint
from dataloaders import images_loader
from helpers.global_config import global_config
from helpers.paths import CKPTS_DIR_NAME
from helpers.saver import Saver
from train import lr_schedule
from train.train_restorer import TrainRestorer
from train.trainer import LogConfig, Trainer


class MultiscaleTrainer(Trainer):
    def __init__(self,
                 ms_config_p, dl_config_p,
                 log_dir_root, log_config: LogConfig,
                 num_workers,
                 saver: Saver, restorer: TrainRestorer=None,
                 sw_cls=vis.safe_summary_writer.SafeSummaryWriter):
        """
        :param ms_config_p: Path to the multiscale config file, see README
        :param dl_config_p: Path to the dataloader config file, see README
        :param log_dir_root: All outputs (checkpoints, tensorboard) will be saved here.
        :param log_config: Instance of train.trainer.LogConfig, contains intervals.
        :param num_workers: Number of workers to use for DataLoading, see train.py
        :param saver: Saver instance to use.
        :param restorer: Instance of TrainRestorer, if we need to restore
        """

        # Read configs
        # config_ms = config for the network (ms = multiscale)
        # config_dl = config for data loading
        (self.config_ms, self.config_dl), rel_paths = ft.unzip(map(config_parser.parse, [ms_config_p, dl_config_p]))
        # Update config_ms depending on global_config
        global_config.update_config(self.config_ms)
        # Create data loaders
        dl_train, dl_val = self._get_dataloaders(num_workers)
        # Create blueprint. A blueprint collects the network as well as the losses in one class, for easy reuse
        # during testing.
        self.blueprint = MultiscaleBlueprint(self.config_ms)
        print('Network:', self.blueprint.net)
        # Setup optimizer
        optim_cls = {'RMSprop': optim.RMSprop,
                     'Adam': optim.Adam,
                     'SGD': optim.SGD,
                     }[self.config_ms.optim]
        net = self.blueprint.net
        self.optim = optim_cls(net.parameters(), self.config_ms.lr.initial,
                               weight_decay=self.config_ms.weight_decay)
        # Calculate a rough estimate for time per batch (does not take into account that CUDA is async,
        # but good enought to get a feeling during training).
        self.time_accumulator = timer.TimeAccumulator()
        # Restore network if requested
        skip_to_itr = self.maybe_restore(restorer)
        if skip_to_itr is not None:  # i.e., we have a restorer
            print('Skipping to {}...'.format(skip_to_itr))
        # Create LR schedule to update parameters
        self.lr_schedule = lr_schedule.from_spec(
                self.config_ms.lr.schedule, self.config_ms.lr.initial, [self.optim], epoch_len=len(dl_train))

        # --- All nn.Modules are setup ---
        print('-' * 80)

        # create log dir and summary writer
        self.log_dir = Trainer.get_log_dir(log_dir_root, rel_paths, restorer)
        self.log_date = logdir_helpers.log_date_from_log_dir(self.log_dir)
        self.ckpt_dir = os.path.join(self.log_dir, CKPTS_DIR_NAME)
        print(f'Checkpoints will be saved to {self.ckpt_dir}')
        saver.set_out_dir(self.ckpt_dir)


        # Create summary writer
        sw = sw_cls(self.log_dir)
        self.summarizer = vis.summarizable_module.Summarizer(sw)
        net.register_summarizer(self.summarizer)
        self.blueprint.register_summarizer(self.summarizer)
        # superclass setup
        super(MultiscaleTrainer, self).__init__(dl_train, dl_val, [self.optim], net, sw,
                                                max_epochs=self.config_dl.max_epochs,
                                                log_config=log_config, saver=saver, skip_to_itr=skip_to_itr)

    def modules_to_save(self):
        return {'net': self.blueprint.net,
                'optim': self.optim}

    def _get_dataloaders(self, num_workers, shuffle_train=True):
        assert self.config_dl.train_imgs_glob is not None
        print('Cropping to {}'.format(self.config_dl.crop_size))
        to_tensor_transform = transforms.Compose(
                [transforms.RandomCrop(self.config_dl.crop_size),
                 transforms.RandomHorizontalFlip(),
                 images_loader.IndexImagesDataset.to_tensor_uint8_transform()])
        # NOTE: if there are images in your training set with dimensions <128, training will abort at some point,
        # because the cropper failes. See REAME, section about data preparation.
        min_size = self.config_dl.crop_size
        ds_train = images_loader.IndexImagesDataset(
                images=images_loader.ImagesCached(
                        self.config_dl.train_imgs_glob,
                        self.config_dl.image_cache_pkl,
                        min_size=min_size),
                to_tensor_transform=to_tensor_transform)

        dl_train = DataLoader(ds_train, self.config_dl.batchsize_train, shuffle=shuffle_train,
                              num_workers=num_workers)
        print('Created DataLoader [train] {} batches -> {} imgs'.format(
                len(dl_train), self.config_dl.batchsize_train * len(dl_train)))

        ds_val = self._get_ds_val(
                self.config_dl.val_glob,
                crop=self.config_dl.crop_size,
                truncate=self.config_dl.num_val_batches * self.config_dl.batchsize_val)
        dl_val = DataLoader(
                ds_val, self.config_dl.batchsize_val, shuffle=False,
                num_workers=num_workers, drop_last=True)
        print('Created DataLoader [val] {} batches -> {} imgs'.format(
                len(dl_val), self.config_dl.batchsize_train * len(dl_val)))

        return dl_train, dl_val

    def _get_ds_val(self, images_spec, crop=False, truncate=False):
        img_to_tensor_t = [images_loader.IndexImagesDataset.to_tensor_uint8_transform()]
        if crop:
            img_to_tensor_t.insert(0, transforms.CenterCrop(crop))
        img_to_tensor_t = transforms.Compose(img_to_tensor_t)

        fixed_first = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixedimg.jpg')
        if not os.path.isfile(fixed_first):
            print(f'INFO: No file found at {fixed_first}')
            fixed_first = None

        ds = images_loader.IndexImagesDataset(
                images=images_loader.ImagesCached(
                        images_spec, self.config_dl.image_cache_pkl,
                        min_size=self.config_dl.val_glob_min_size),
                to_tensor_transform=img_to_tensor_t,
                fixed_first=fixed_first)  # fix a first image to have consistency in tensor board

        if truncate:
            ds = pe.TruncatedDataset(ds, num_elemens=truncate)

        return ds

    def train_step(self, i, batch, log, log_heavy, load_time=None):
        """
        :param i: current step
        :param batch: dict with 'idx', 'raw'
        """
        self.lr_schedule.update(i)
        self.net.zero_grad()

        values = Values('{:.3e}', ' | ')

        with self.time_accumulator.execute():
            idxs, img_batch, s = self.blueprint.unpack(batch)

            with self.summarizer.maybe_enable(prefix='train', flag=log, global_step=i):
                out = self.blueprint.forward(img_batch)

            with self.summarizer.maybe_enable(prefix='train', flag=log_heavy, global_step=i):
                loss_pc, nonrecursive_bpsps, _ = self.blueprint.get_loss(out)

            total_loss = loss_pc
            total_loss.backward()
            self.optim.step()

            values['loss'] = loss_pc
            values['bpsp'] = sum(nonrecursive_bpsps)

        if not log:
            return

        mean_time_per_batch = self.time_accumulator.mean_time_spent()
        imgs_per_second = self.config_dl.batchsize_train / mean_time_per_batch

        print('{} {: 6d}: {} // {:.3f} img/s '.format(
                self.log_date, i, values.get_str(), imgs_per_second) + (load_time or ''))

        values.write(self.sw, i)

        # Gradients
        params = [('all', self.net.parameters())]
        for name, ps in params:
            tot = pe.get_total_grad_norm(ps)
            self.sw.add_scalar('grads/{}/total'.format(name), tot, i)

        # log LR
        lrs = list(self.get_lrs())
        assert len(lrs) == 1
        self.sw.add_scalar('train/lr', lrs[0], i)

        if not log_heavy:
            return

        self.blueprint.add_image_summaries(self.sw, out, i, 'train')

    def validation_loop(self, i):
        bs = pe.BatchSummarizer(self.sw, i)
        val_start = time.time()
        for j, batch in enumerate(self.dl_val):
            idxs, img_batch, s = self.blueprint.unpack(batch)

            # Only log TB summaries for first batch
            with self.summarizer.maybe_enable(prefix='val', flag=j == 0, global_step=i):
                out = self.blueprint.forward(img_batch)
                loss_pc, nonrecursive_bpsps, _ = self.blueprint.get_loss(out)

            bs.append('val/bpsp', sum(nonrecursive_bpsps))

            if j > 0:
                continue

            self.blueprint.add_image_summaries(self.sw, out, i, 'val')

        val_duration = time.time() - val_start
        num_imgs = len(self.dl_val.dataset)
        time_per_img = val_duration/num_imgs

        output_strs = bs.output_summaries()
        output_strs = ['{: 6d}'.format(i)] + output_strs + ['({:.3f} s/img)'.format(time_per_img)]
        output_str = ' | '.join(output_strs)
        sep = '-' * len(output_str)
        print('\n'.join([sep, output_str, sep]))


class Values(object):
    """
    Stores values during one training step. Essentially a thin wrapper around dict with support to get a nicely
    formatted string and write to a SummaryWriter.
    """
    def __init__(self, fmt_str='{:.3f}', joiner=' / ', prefix='train/'):
        self.fmt_str = fmt_str
        self.joiner = joiner
        self.prefix = prefix
        self.values = {}

    def __setitem__(self, key, value):
        self.values[key] = value

    def get_str(self):
        """ :return pretty printed version of all values, using default_fmt_str """
        return self.joiner.join('{} {}'.format(k, self.fmt_str.format(v))
                                for k, v in sorted(self.values.items()))

    def write(self, sw, i):
        """ Writes to summary writer `sw`. """
        for k, v in self.values.items():
            sw.add_scalar(self.prefix + k, v, i)
