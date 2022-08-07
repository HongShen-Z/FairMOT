import numpy as np

from lib.datasets.dataset.jde import JointDataset
from lib.opts import opts
import json
from torchvision.transforms import transforms as T
import torch.utils.data

if __name__ == '__main__':
    opt = opts().parse()
    with open('lib/cfg/crowdhuman.json') as f:
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(opt, dataset_root, trainset_paths, img_size=(1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    for iter_id, batch in enumerate(train_loader):
        print('#' * 100)
        break

