import os
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms
import models_net_mamba
from engine import evaluate
import json
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import random
from tqdm import tqdm
import time
from datetime import datetime


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
    print("\n" + "="*80)
    print("📊 数据集加载阶段")
    print("="*80)
    
    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(dataset_path)
    
    print(f"📁 数据集根目录: {root}")
    dataset = datasets.ImageFolder(root, transform=transform)
    
    print(f"📈 原始数据集大小: {len(dataset)} 个样本")
    print(f"📋 所有类别: {dataset.classes}")
    print(f"🔢 原始类别数量: {len(dataset.classes)}")

    # 限制数据集只包含指定类别
    original_size = len(dataset.targets)
    dataset.targets = [target for target in dataset.targets if target < num_classes]
    dataset.samples = [sample for i, sample in enumerate(dataset.samples) if dataset.targets[i] < num_classes]
    filtered_size = len(dataset.targets)
    
    print(f"✂️  过滤后数据集大小: {filtered_size} 个样本 (过滤掉 {original_size - filtered_size} 个超出类别范围的样本)")
    print(f"✅ 最终使用的类别数: {num_classes}")
    
    # 统计每个类别的样本数量
    class_distribution = {}
    for target in dataset.targets:
        class_name = dataset.classes[target]
        class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
    
    print("\n📊 各类别样本分布:")
    for class_name, count in sorted(class_distribution.items()):
        percentage = (count / filtered_size) * 100
        print(f"   ├─ {class_name}: {count} 个样本 ({percentage:.2f}%)")
    
    # 获取第一个样本的形状信息
    if len(dataset) > 0:
        sample_img, sample_label = dataset[0]
        print(f"\n🖼️  单个样本数据形状: {tuple(sample_img.shape)}")
        print(f"🏷️  标签格式: {sample_label} (类型: {type(sample_label).__name__})")

    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False
    )
    
    num_batches = len(data_loader)
    print(f"\n🚀 DataLoader配置:")
    print(f"   ├─ Batch Size: {batch_size}")
    print(f"   ├─ Num Workers: {num_workers}")
    print(f"   ├─ Pin Memory: {pin_mem}")
    print(f"   ├─ Total Batches: {num_batches}")
    print("="*80 + "\n")
    
    return data_loader, dataset.classes


def inference(model, data_loader, device, output_dir, array_dir, test_dir, dataset_classes, if_stat=False):
    """执行推理"""
    print("\n" + "="*80)
    print("🤖 模型推理阶段")
    print("="*80)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 模型信息:")
    print(f"   ├─ 总参数量: {total_params:,}")
    print(f"   ├─ 可训练参数: {trainable_params:,}")
    print(f"   ├─ 推理设备: {device}")
    print(f"   ├─ 模型类别数: {len(dataset_classes)}")
    print(f"   └─ 类别列表: {dataset_classes}")
    
    # 执行推理，获取结果
    print("\n⏳ 开始执行推理...")
    start_time = time.time()
    
    test_stats, png_to_json_path = evaluate(data_loader, model, device, test_dir=test_dir, array_dir=array_dir,
                                            if_stat=if_stat)
    
    elapsed_time = time.time() - start_time
    print(f"✅ 推理完成! 总耗时: {elapsed_time:.2f} 秒")

    # 获取准确率和其他统计指标
    accuracy = round(test_stats.get('acc1', 0), 4)  # top-1 accuracy
    precision_weighted = round(test_stats.get('weighted_pre'), 4)  # weighted precision (scalar)
    recall_weighted = round(test_stats.get('weighted_rec'), 4)  # weighted recall (scalar)
    f1_score_weighted = round(test_stats.get('weighted_f1'), 4)  # weighted f1 score (scalar)
    
    # 获取每个类别的精确率、召回率、F1分数（数组形式）
    precision_per_class = test_stats.get('pre_per_class', np.array([]))  # precision for each class
    recall_per_class = test_stats.get('rec_per_class', np.array([]))  # recall for each class
    f1_score_per_class = test_stats.get('f1_per_class', np.array([]))  # f1 score for each class

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
    
    print(f"\n📊 预测结果统计:")
    print(f"   ├─ 总样本数: {len(pred_all)}")
    print(f"   ├─ 正确预测数: {sum([1 for p, t in zip(pred_all, target_all) if p == t])}")
    print(f"   ├─ 错误预测数: {sum([1 for p, t in zip(pred_all, target_all) if p != t])}")
    print(f"   └─ 准确率: {accuracy * 100:.2f}%")
    
    print(f"\n📈 各类别预测分布:")
    for cls, count in sorted(class_counts.items()):
        print(f"   ├─ {cls}: {count} 个")

    # 查找正确的预测图像和相应的 JSON 路径
    print("\n🔍 正在收集正确预测的样本路径...")
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.endswith('.png'):
                image_paths.append(os.path.join(root, f))

    for i in range(len(pred_all)):
        if pred_all[i] == target_all[i]:
            correct_image_path.append(image_paths[i])
    
    print(f"✅ 找到 {len(correct_image_path)} 个正确预测的样本")

    for i in range(len(correct_image_path)):
        json1 = png_to_json_path.get(str(correct_image_path[i]))
        correct_json_path.append(json1)

    correct_json_path1 = correct_json_path[0:1000000]  # 只考虑前100个图像路径

    # 初始化提取的数据字典（排除 "benign" 类）
    class_data = {cls: [] for cls in dataset_classes if cls != "benign" and cls != "BEGIN"}  # 排除 BEGIN 类

    # 遍历 JSON 文件路径并提取信息
    print(f"\n📄 正在从 {len(correct_json_path1)} 个JSON文件中提取详细信息...")
    extraction_progress = tqdm(correct_json_path1, desc="提取数据", unit="file")
    
    for i, file_path in enumerate(extraction_progress):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                # 提取 "details" 字段的第一部分数据
                details_first_part = json_data.get("details", [])[0] if json_data.get("details") else {}
                # 提取流量分类信息
                predicted_class = predicted_classes[i]  # 预测的分类结果
    
                # 只提取 "benign" 类和 "BEGIN" 类以外的数据,每个类别最大提取100条数据
                if predicted_class != "benign" and predicted_class != "BEGIN" and len(class_data.get(predicted_class, [])) < 100:
                    extracted_data = {
                        "src_ip": details_first_part.get("src_ip"),
                        "dst_ip": details_first_part.get("dst_ip"),
                        "src_port": details_first_part.get("src_port"),
                        "dst_port": details_first_part.get("dst_port"),
                        "class": predicted_class  # 使用预测的分类作为 "class"
                    }
                    class_data[predicted_class].append(extracted_data)
                    extraction_progress.set_postfix({"已提取": sum(len(v) for v in class_data.values())})
    
        except (KeyError, IndexError, TypeError) as e:
            # 静默跳过处理错误的文件,不输出错误信息
            pass
        except Exception as e:
            # 其他异常也静默处理
            pass
    
    extraction_progress.close()

    # 合并所有类别的提取数据，并随机打乱顺序
    all_extracted_data = []
    for cls in class_data:
        all_extracted_data.extend(class_data[cls])

    # 打乱数据顺序，确保不全是一个类别的
    random.shuffle(all_extracted_data)
    
    print(f"✅ 共提取 {len(all_extracted_data)} 条详细数据记录")

    # 计算混淆矩阵
    print("\n📐 计算评估指标...")
    cm = confusion_matrix(target_all, pred_all)

    # 计算 TP, TN, FP, FN 对于每个类
    tp = np.diag(cm)  # 对角线元素就是 TP
    fn = cm.sum(axis=1) - tp  # 每一行的和减去 TP 就是 FN
    fp = cm.sum(axis=0) - tp  # 每一列的和减去 TP 就是 FP
    tn = cm.sum() - (tp + fn + fp)  # 总数减去 TP, FN, FP 就是 TN

    # 打印详细混淆矩阵
    print("\n" + "="*80)
    print("📊 混淆矩阵 (Confusion Matrix)")
    print("="*80)
    print("\n混淆矩阵:")
    print(cm)
    
    print("\n各类别详细指标:")
    print("-" * 80)
    print(f"{'类别':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 80)
    
    for i, class_name in enumerate(dataset_classes):
        prec_i = precision_per_class[i] if i < len(precision_per_class) else 0.0
        rec_i = recall_per_class[i] if i < len(recall_per_class) else 0.0
        f1_i = f1_score_per_class[i] if i < len(f1_score_per_class) else 0.0
        print(f"{class_name:<20} {tp[i]:>6} {fp[i]:>6} {fn[i]:>6} {tn[i]:>8} "
              f"{prec_i:>10.4f} {rec_i:>10.4f} {f1_i:>10.4f}")
    
    print("-" * 80)
    print(f"{'加权平均':<20} {'-':>6} {'-':>6} {'-':>6} {'-':>8} "
          f"{precision_weighted:>10.4f} {recall_weighted:>10.4f} {f1_score_weighted:>10.4f}")
    print("="*80)
    
    # 打印完整的分类报告
    print("\n📋 完整分类报告:")
    print(classification_report(target_all, pred_all, target_names=dataset_classes, digits=4))

    # 打印最终结果
    result = {
        'Accuracy': accuracy,
        'Precision': precision_weighted,
        'Recall': recall_weighted,
        'F1': f1_score_weighted,
        'Precision_Per_Class': precision_per_class.tolist() if isinstance(precision_per_class, np.ndarray) else precision_per_class,
        'Recall_Per_Class': recall_per_class.tolist() if isinstance(recall_per_class, np.ndarray) else recall_per_class,
        'F1_Per_Class': f1_score_per_class.tolist() if isinstance(f1_score_per_class, np.ndarray) else f1_score_per_class,
        'Class Counts': class_counts,
        'Extracted Data': all_extracted_data,
        'Confusion Matrix': cm.tolist(),  # 将混淆矩阵转化为可保存的格式
        'TP': tp.tolist(),  # 将 TP 转换为可保存的格式
        'FN': fn.tolist(),  # 将 FN 转换为可保存的格式
        'FP': fp.tolist(),  # 将 FP 转换为可保存的格式
        'TN': tn.tolist(),   # 将 TN 转换为可保存的格式
        'Total Samples': len(pred_all),
        'Correct Predictions': int(sum([1 for p, t in zip(pred_all, target_all) if p == t])),
        'Wrong Predictions': int(sum([1 for p, t in zip(pred_all, target_all) if p != t])),
        'Inference Time (seconds)': round(elapsed_time, 2),
        'Samples Per Second': round(len(pred_all) / elapsed_time, 2) if elapsed_time > 0 else 0
    }

    # 将结果保存到文件
    output_file = os.path.join(output_dir, 'test_stats_1.json')
    print(f"\n💾 保存结果到: {output_file}")
    with open(output_file, 'w') as f:
        new_stats = {k: v.tolist() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in result.items()}
        json.dump(new_stats, f, indent=2, ensure_ascii=False)
    print("✅ 结果保存成功!")
    print("="*80 + "\n")

    return result


def run_inference(model_path, test_path, output_dir, array_path, model_name='net_mamba_classifier',
                  num_classes=7, drop_path_rate=0.1, batch_size=256, device='cuda', num_workers=8, pin_mem=True,
                  if_stat=False):
    """
    主推理接口函数，根据提供的路径加载模型和数据集，并执行推理。
    - num_classes: 可变的推理类别数
    """
    print("\n" + "#"*80)
    print("# NetMamba 推理系统启动")
    print("#"*80)
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置设备
    device = torch.device(device)
    print(f"💻 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 加载模型
    print("\n🔄 正在加载模型...")
    model_start = time.time()
    model = load_model(model_name, model_path, device, num_classes, drop_path_rate)
    model_time = time.time() - model_start
    print(f"✅ 模型加载完成! 耗时: {model_time:.2f} 秒")

    # 加载数据集
    print("\n🔄 正在加载数据集...")
    data_start = time.time()
    data_loader, dataset_classes = load_dataset(test_path, is_train=False, batch_size=batch_size, num_workers=num_workers,
                               pin_mem=pin_mem, num_classes=num_classes)
    data_time = time.time() - data_start
    print(f"✅ 数据集加载完成! 耗时: {data_time:.2f} 秒")

    # 执行推理
    print("\n🚀 开始执行推理任务...")
    inference_start = time.time()
    result = inference(model, data_loader, device, output_dir, array_path, test_path, dataset_classes, if_stat)
    inference_time = time.time() - inference_start
    
    total_time = time.time() - model_start
    
    # 打印总结
    print("\n" + "#"*80)
    print("# 推理任务完成总结")
    print("#"*80)
    print(f"📊 总体统计:")
    print(f"   ├─ 模型加载时间: {model_time:.2f} 秒")
    print(f"   ├─ 数据加载时间: {data_time:.2f} 秒")
    print(f"   ├─ 推理执行时间: {inference_time:.2f} 秒")
    print(f"   ├─ 总耗时: {total_time:.2f} 秒")
    print(f"   ├─ 处理样本数: {result['Total Samples']}")
    print(f"   ├─ 推理速度: {result['Samples Per Second']:.2f} 样本/秒")
    print(f"   ├─ 准确率: {result['Accuracy']*100:.2f}%")
    print(f"   ├─ 精确率: {result['Precision']*100:.2f}%")
    print(f"   ├─ 召回率: {result['Recall']*100:.2f}%")
    print(f"   └─ F1分数: {result['F1']*100:.2f}%")
    print("#"*80)
    print(f"✅ 所有任务完成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return result


# 调用示例
if __name__ == "__main__":
    model_path = './output/finetune-CHANGZHOU/USTC-TFC2016/checkpoint-best.pth'  # 模型路径
    test_path = './Net-Mamba-data/root/Vim/dataset/USTC-TFC2016/dataset_sampled/test'  # 测试数据路径
    output_dir = './output/inference-CHANGZHOU/USTC-TFC2016'   # 输出目录
    array_path = '/home/wang/下载/NetMamba/dataset/USTC-TFC2016/array_sampled'  # JSON 数据路径
    num_classes = 20 # 设置推理类别数（动态设置）

    # 调用推理接口
    test_stats = run_inference(model_path, test_path, output_dir, array_path, num_classes=num_classes)

    # 打印推理结果
    # print("Inference result:", test_stats)
