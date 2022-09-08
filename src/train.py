from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from lib.models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    with open(opt.data_cfg) as f:
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, img_size=(1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # -----------------dev----------------- #
    model_params = []
    for m in model:
        model_params += model[m].parameters()
    optimizer = torch.optim.Adam(model_params, opt.lr)
    # -----------------dev----------------- #

    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0004)
    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    print(model)
    model = torch.nn.DataParallel(model, list(range(len(opt.gpus))))
    print('=' * 100)
    print(model)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'models', 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer, opt.tasks)
        else:
            save_model(os.path.join(opt.save_dir, 'models', 'model_last.pth'),
                       epoch, model, optimizer, opt.tasks)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'models', 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer, opt.tasks)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= opt.num_epochs - 5:
            save_model(os.path.join(opt.save_dir, 'models', 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer, opt.tasks)
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
