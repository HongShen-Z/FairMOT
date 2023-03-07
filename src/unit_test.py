# import numpy as np
#
# from lib.datasets.dataset.jde import JointDataset
# from lib.opts import opts
# import json
# from torchvision.transforms import transforms as T
# import torch.utils.data
#
# if __name__ == '__main__':
#     opt = opts().parse()
#     with open('lib/cfg/crowdhuman.json') as f:
#         data_config = json.load(f)
#         trainset_paths = data_config['train']
#         dataset_root = data_config['root']
#     transforms = T.Compose([T.ToTensor()])
#     dataset = JointDataset(opt, dataset_root, trainset_paths, img_size=(1088, 608), augment=True, transforms=transforms)
#     opt = opts().update_dataset_info_and_set_heads(opt, dataset)
#     train_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=opt.num_workers,
#         pin_memory=True,
#         drop_last=True
#     )
#     for iter_id, batch in enumerate(train_loader):
#         print(sum(batch['box_target'].view(-1) > 0), len(batch['box_target'].view(-1)))
#         break
#


import os

# 指定要统计行数的文件夹路径
folder_path = 'E:\Postgra\projects\FairMOT\src\data'

# 获取文件夹下所有文件名
file_names = ['mot17.train']

# 定义变量用于累计行数
total_lines = 0

# 遍历所有文件，统计行数
for file_name in file_names:
    # 组合文件路径
    file_path = os.path.join(folder_path, file_name)

    # 判断是否是文件
    if os.path.isfile(file_path):
        # 统计文件行数
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines += len(lines)

# 打印结果
print('共有 %d 行数据' % total_lines)


