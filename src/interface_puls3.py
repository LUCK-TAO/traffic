import os
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms
import models_net_mamba
from engine import evaluate
import json
from sklearn.metrics import confusion_matrix
import random


def load_model(model_name, checkpoint_path, device, num_classes, drop_path_rate):
    """加载模型并恢复权重"""
    model = models_net_mamba.__dict__[model_name](
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Loading model checkpoint from:", checkpoint_path)
        model.load_state_dict(checkpoint['model'], strict=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
    return model.to(device)


def load_dataset(dataset_path, is_train, batch_size, num_workers, pin_mem, num_classes):
    """加载数据集"""
    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(dataset_path)
    dataset = datasets.ImageFolder(root, transform=transform)

    # 限制数据集只包含指定类别
    dataset.targets = [target for target in dataset.targets if target < num_classes]
    dataset.samples = [sample for i, sample in enumerate(dataset.samples) if dataset.targets[i] < num_classes]

    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False
    )
    return data_loader, dataset.classes


def inference(model, data_loader, device, output_dir, array_dir, test_dir, dataset_classes, if_stat=False):
    """执行推理"""
    # 执行推理，获取结果
    test_stats, png_to_json_path = evaluate(data_loader, model, device, test_dir=test_dir, array_dir=array_dir,
                                            if_stat=if_stat)

    # 获取准确率和其他统计指标
    accuracy = round(test_stats.get('acc1', 0), 4)  # top-1 accuracy
    precision = round(test_stats.get('weighted_pre'), 4)  # precision for each class
    recall = round(test_stats.get('weighted_rec'), 4)  # recall for each class
    f1_score = round(test_stats.get('weighted_f1'), 4)  # f1 score for each class

    pred_all = test_stats['pred_all']
    target_all = test_stats['target_all']
    image_paths = []
    correct_image_path = []
    correct_json_path = []
    all_extracted_data = []  # 存储所有提取的数据
    num = 0

    # 根据预测的类别索引获取类别名称
    predicted_classes = [dataset_classes[pred] for pred in pred_all]  # 将类别数字转换为名称

    # 计算每个类别的数量
    class_counts = {cls: predicted_classes.count(cls) for cls in dataset_classes}

    # 查找正确的预测图像和相应的 JSON 路径
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.endswith('.png'):
                image_paths.append(os.path.join(root, f))

    for i in range(len(pred_all)):
        if pred_all[i] == target_all[i]:
            correct_image_path.append(image_paths[i])

    for i in range(len(correct_image_path)):
        json1 = png_to_json_path.get(str(correct_image_path[i]))
        correct_json_path.append(json1)

    correct_json_path1 = correct_json_path[0:1000000]  # 只考虑前100个图像路径

    # 初始化提取的数据字典（排除 "benign" 类）
    class_data = {cls: [] for cls in dataset_classes if cls != "benign" and cls != "BEGIN"}  # 排除 BEGIN 类

    # 遍历 JSON 文件路径并提取信息
    for i, file_path in enumerate(correct_json_path1):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                # 提取 "details" 字段的第一部分数据
                details_first_part = json_data.get("details", [])[0] if json_data.get("details") else {}
                # 提取流量分类信息
                predicted_class = predicted_classes[i]  # 预测的分类结果

                # 只提取 "benign" 类以外的数据，每个类别最大提取100条数据
                if predicted_class != "benign" and len(class_data[predicted_class]) < 100:
                    extracted_data = {
                        "src_ip": details_first_part.get("src_ip"),
                        "dst_ip": details_first_part.get("dst_ip"),
                        "src_port": details_first_part.get("src_port"),
                        "dst_port": details_first_part.get("dst_port"),
                        "class": predicted_class  # 使用预测的分类作为 "class"
                    }
                    class_data[predicted_class].append(extracted_data)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 合并所有类别的提取数据，并随机打乱顺序
    all_extracted_data = []
    for cls in class_data:
        all_extracted_data.extend(class_data[cls])

    # 打乱数据顺序，确保不全是一个类别的
    random.shuffle(all_extracted_data)

    # 计算混淆矩阵
    cm = confusion_matrix(target_all, pred_all)

    # 计算 TP, TN, FP, FN 对于每个类
    tp = np.diag(cm)  # 对角线元素就是 TP
    fn = cm.sum(axis=1) - tp  # 每一行的和减去 TP 就是 FN
    fp = cm.sum(axis=0) - tp  # 每一列的和减去 TP 就是 FP
    tn = cm.sum() - (tp + fn + fp)  # 总数减去 TP, FN, FP 就是 TN

    # 打印混淆矩阵
    print("Confusion Matrix:\n", cm)
    print("True Positives (TP):", tp)
    print("False Negatives (FN):", fn)
    print("False Positives (FP):", fp)
    print("True Negatives (TN):", tn)

    # 打印最终结果
    result = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score,
        'Class Counts': class_counts,
        'Extracted Data': all_extracted_data,
        'Confusion Matrix': cm.tolist(),  # 将混淆矩阵转化为可保存的格式
        'TP': tp.tolist(),  # 将 TP 转换为可保存的格式
        'FN': fn.tolist(),  # 将 FN 转换为可保存的格式
        'FP': fp.tolist(),  # 将 FP 转换为可保存的格式
        'TN': tn.tolist()   # 将 TN 转换为可保存的格式
    }

    # 将结果保存到文件
    with open(os.path.join(output_dir, 'test_stats_1.json'), 'w') as f:
        new_stats = {k: v.tolist() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in result.items()}
        json.dump(new_stats, f, indent=2)

    return result


def run_inference(model_path, test_path, output_dir, array_path, model_name='net_mamba_classifier',
                  num_classes=7, drop_path_rate=0.1, batch_size=256, device='cuda', num_workers=8, pin_mem=True,
                  if_stat=False):
    """
    主推理接口函数，根据提供的路径加载模型和数据集，并执行推理。
    - num_classes: 可变的推理类别数
    """
    # 设置设备
    device = torch.device(device)

    # 加载模型
    model = load_model(model_name, model_path, device, num_classes, drop_path_rate)

    # 加载数据集
    data_loader, dataset_classes = load_dataset(test_path, is_train=False, batch_size=batch_size, num_workers=num_workers,
                               pin_mem=pin_mem, num_classes=num_classes)

    # 执行推理
    return inference(model, data_loader, device, output_dir, array_path, test_path, dataset_classes, if_stat)


# 调用示例
if __name__ == "__main__":
    model_path = './output/finetune-CHANGZHOU/ISCXVPN2016/checkpoint-best.pth'  # 模型路径
    test_path = './Net-Mamba-data/root/Vim/dataset/ISCXVPN2016/dataset_sampled/train'  # 测试数据路径
    output_dir = './output/inference-CHANGZHOU/ISCXVPN2016'  # 输出目录
    array_path = '/home/wang/下载/NetMamba/dataset/ISCXVPN2016/array_sampled'  # JSON 数据路径
    num_classes = 8  # 设置推理类别数（动态设置）

    # 调用推理接口
    test_stats = run_inference(model_path, test_path, output_dir, array_path, num_classes=num_classes)

    # 打印推理结果
    # print("Inference result:", test_stats)
