import os
import numpy as np
import torch
from config import cfg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


def data_reader(image_path, is_rgb=True):
    image = Image.open(image_path)
    if is_rgb:
        image_grey = image.convert('L')
        return image_grey, image
    else:
        return image
class FusionDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, is_rgb=False):
        """
        Args:
            root_dir (string)
            transform (callable, optional)
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.is_rgb = is_rgb

        self.image_names = [f for f in os.listdir(os.path.join(dataset_dir, "Ir")) if os.path.isfile(os.path.join(dataset_dir, "Ir", f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 根据idx读取四类图像
        img_name = self.image_names[idx]
        ir_path = os.path.join(self.dataset_dir, "Ir", img_name)
        vis_path = os.path.join(self.dataset_dir, "Vis", img_name)
        ir_image = data_reader(ir_path, is_rgb=False)
        vis_image, vis_rgb = data_reader(vis_path, is_rgb=True)

        # 数据预处理
        if self.transform:
            ir_image = self.transform(ir_image)
            vis_image = self.transform(vis_image)
            vis_rgb = self.transform(vis_rgb)

        # 返回图像数据
        return img_name, ir_image, vis_image, vis_rgb


class coco_dataset(Dataset):
    def __init__(self, dataset_dir, transform):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_names = os.listdir(self.dataset_dir)
        self.image_names = self.image_names[:80000]
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_dir = os.path.join(self.dataset_dir, img_name)
        image = Image.open(img_dir).convert('L')
        image = np.array(image)

        if self.transform:
            image = self.transform(image)
        return image


def transform_act():
    # 数据预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Resize(cfg.DATA.DATA_SIZE)
        # 其他预处理步骤可以加在这里 (例如transforms.Normalize等)
    ])
    return transform

def data_loader(data_path, data_name, batch_size, is_rgb):
    data_list = ['M3FD', 'MSRS', 'RoadScene', 'TNO', 'COCO']
    assert data_name in data_list, f"{data_name} is not in supported data!"
    dataset_dir = data_path
    if data_name == 'M3FD' or data_name == 'RoadScene':
        fusion_dataset = FusionDataset(dataset_dir=dataset_dir, transform=transform_act(),
                                       is_rgb=is_rgb)

        train_indices, test_indices = train_test_split(np.arange(len(fusion_dataset)), test_size=0.2, random_state=42)

        train_dataset = torch.utils.data.Subset(fusion_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(fusion_dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif data_name == 'MSRS':
        train_dir = os.path.join(dataset_dir, 'train')
        test_dir = os.path.join(dataset_dir, 'test')
        train_dataset = FusionDataset(dataset_dir=train_dir, transform=transform_act(),
                                      is_rgb=is_rgb)
        test_dataset = FusionDataset(dataset_dir=test_dir, transform=transform_act(),
                                     is_rgb=is_rgb)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif data_name == 'TNO':
        test_dataset = FusionDataset(dataset_dir=dataset_dir, transform=transform_act(),
                                     is_rgb=is_rgb)
        train_loader = None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif data_name == 'COCO':
        coco_data = coco_dataset(dataset_dir=dataset_dir, transform=transform_act())
        train_loader = DataLoader(coco_data, batch_size=batch_size, shuffle=True)
        test_loader = None

    return train_loader, test_loader


def loader_size(dataloader):
    batch = next(iter(dataloader))
    batch_size = len(batch)
    data_size = batch[0].shape  # 假设第一个元素是数据
    # print(f"Batch size: {batch_size}, Data size: {data_size}")
    return batch_size, data_size

def test():
    # 实例化数据集
    root_dir = 'your_dataset_directory'  # 替换为你的数据集文件夹
    fusion_dataset = FusionDataset(dataset_dir=root_dir, transform=transform_act())

    # 拆分数据集为训练集和测试集
    train_indices, test_indices = train_test_split(np.arange(len(fusion_dataset)), test_size=0.2, random_state=42)

    # 创建数据子集
    train_dataset = torch.utils.data.Subset(fusion_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(fusion_dataset, test_indices)

    # 创建DataLoader来批量加载数据
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 使用DataLoader
    for train_data in train_loader:
        ir_L_image, vis_L_image, ir_H_image, vis_H_image = train_data
        # 进行训练相关的操作...

    # 在测试集上进行测试
    for test_data in test_loader:
        ir_L_image, vis_L_image, ir_H_image, vis_H_image = test_data
        # 进行测试相关的操作...