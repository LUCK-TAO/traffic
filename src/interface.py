import os

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score
from torch import nn

from src import models_net_mamba


class NetMambaModel(nn.Module):
    def __init__(self, num_classes, drop_path_rate, **kwargs):
        """
        初始化NetMamba模型

        参数:
            model_name (str): 模型的名字，对应 `models_net_mamba` 中的类名
            num_classes (int): 类别数量
            drop_path_rate (float): DropPath的比率
            **kwargs: 其他需要传递给模型的超参数
        """
        super(NetMambaModel, self).__init__()
        # 从 models_net_mamba 动态加载模型类
        self.model = models_net_mamba.__dict__[models_net_mamba](
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            **kwargs
        )

    def forward(self, x):
        """ 前向传播 """
        return self.model(x)

    def load_checkpoint(self, checkpoint_path, map_location='cpu'):
        """ 加载预训练模型的参数 """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded successfully!")

    def save_checkpoint(self, checkpoint_path):
        """ 保存当前模型的参数 """
        torch.save({
            'model': self.model.state_dict()
        }, checkpoint_path)
        print("Checkpoint saved successfully!")

    def get_model(self):
        """ 返回封装的模型对象 """
        return self.model


class ModelInferenceInterface(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        """
        加载模型的权重文件
        """
        pass

    @abstractmethod
    def predict(self, data_loader):
        """
        进行推理，返回预测结果
        """
        pass



class TrafficModelInference(ModelInferenceInterface):
    def __init__(self, model_class, num_classes=7):
        self.model = model_class(num_classes=num_classes)
        self.model.eval()  # 设置为评估模式

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])

    def evaluate(self, data_loader):
        all_labels = []
        all_predictions = []

        # 对整个数据集进行推理
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy



# 模型管理器
class ModelManager:
    def __init__(self):
        self.models = {}

    def load_model(self, model_name, model_class, model_path, num_classes, **kwargs):
        """
        加载模型，并存储到模型管理器中
        :param model_name: 模型的名称，作为模型的标识符
        :param model_class: 模型的类（NetMamba）
        :param model_path: 模型权重文件 (.pth) 的路径
        :param num_classes: 分类数量
        :param kwargs: 其他参数，如设备、drop_path_rate等
        """
        if model_name in self.models:
            print(f"Model {model_name} already loaded.")
            return

        # 初始化模型
        # model = model_class(num_classes=num_classes, **kwargs)


        model = NetMambaModel(
            num_classes=num_classes,
            drop_path_rate=0.1
        )

        # 加载模型权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint)
            model.eval()  # 设置为评估模式
            self.models[model_name] = model
            print(f"Model {model_name} loaded successfully.")
        else:
            print(f"Model path {model_path} does not exist.")

    def get_model(self, model_name):
        """
        获取已经加载的模型
        :param model_name: 模型的名称
        :return: 对应的模型
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Please load the model first.")
            return None
        return self.models[model_name]





