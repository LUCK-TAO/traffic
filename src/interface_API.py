from flask import Flask, request, jsonify, session
import os
import torch
from pathlib import Path
import json

from flask_cors import CORS

import models_net_mamba
from engine import evaluate
from torchvision import datasets, transforms
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")  # 允许所有域名访问



# def load_model(model_name, checkpoint_path, device, num_classes, drop_path_rate):
#     """加载模型并恢复权重"""
#     model = models_net_mamba.__dict__[model_name](
#         num_classes=num_classes,
#         drop_path_rate=drop_path_rate
#     )
#     try:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         print("Loading model checkpoint from:", checkpoint_path)
#         model.load_state_dict(checkpoint['model'], strict=False)
#     except Exception as e:
#         print(f"Failed to load checkpoint: {e}")
#     return model.to(device)
#
#
# def load_dataset(dataset_path, is_train, batch_size, num_workers, pin_mem, num_classes):
#     """加载数据集"""
#     mean = [0.5]
#     std = [0.5]
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#     root = os.path.join(dataset_path, 'train' if is_train else 'test')
#     dataset = datasets.ImageFolder(root, transform=transform)
#
#     # 限制数据集只包含指定类别
#     dataset.targets = [target for target in dataset.targets if target < num_classes]
#     dataset.samples = [sample for i, sample in enumerate(dataset.samples) if dataset.targets[i] < num_classes]
#
#     sampler = torch.utils.data.SequentialSampler(dataset)
#     data_loader = torch.utils.data.DataLoader(
#         dataset, sampler=sampler,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_mem,
#         drop_last=False
#     )
#     return data_loader, dataset.classes  # 返回类别名称
#
#
# def inference(model, data_loader, device, output_dir, array_dir, test_dir, dataset_classes, if_stat=False):
#     """执行推理"""
#     test_stats, png_to_json_path = evaluate(data_loader, model, device, test_dir=test_dir, array_dir=array_dir,
#                                             if_stat=if_stat)
#
#     pred_all = test_stats['pred_all']
#     target_all = test_stats['target_all']
#     image_paths = []
#     correct_image_path = []
#     correct_json_path = []
#
#     # 根据预测的类别索引获取类别名称
#     predicted_classes = [dataset_classes[pred] for pred in pred_all]  # 将类别数字转换为名称
#
#     for root, _, files in os.walk(test_dir):
#         for f in files:
#             if f.endswith('.png'):
#                 image_paths.append(os.path.join(root, f))
#
#     for i in range(len(pred_all)):
#         if pred_all[i] == target_all[i]:
#             correct_image_path.append(image_paths[i])
#
#     for i in range(len(correct_image_path)):
#         json1 = png_to_json_path.get(str(correct_image_path[i]))
#         correct_json_path.append(json1)
#
#     result = {
#         "predicted_classes": predicted_classes,
#         "test_stats": test_stats
#     }
#
#     return result

SCENARIO_FILE_PATHS = {
    "CICIDS2017": "/home/wang/下载/NetMamba/src/output/GAI/CICIDS2017/test_stats.json",
    # "CICDDOS2019": "/home/wang/下载/NetMamba/src/output/inference-CHANGZHOU/TEST-15/test_stats.json",
    "CICDDOS2019": "/home/wang/下载/NetMamba/src/output/GAI/CICDDOS2019/test_stats.json",
    "USTC-TFC2016": "/home/wang/下载/NetMamba/src/output/GAI/USTC-TFC2016/test_stats.json",
    "ISCXVPN2016": "/home/wang/下载/NetMamba/src/output/GAI/ISCXVPN2016/test_stats.json",
    "DNSoH": "/home/wang/下载/NetMamba/src/output/GAI/DNSoH/test_stats.json",
    "勒索软件": "/home/wang/下载/NetMamba/src/output/GAI/Malware/test_stats.json",
    # 添加更多场景和对应的文件路径
}

@app.route('/inference/', methods=['GET'])
def run_inference_api():
    # 获取前端传递的场景参数
    scenario = request.args.get('scenario', default="CICIDS2017")
    # scenario = request.args.get('scenario')
    # if scenario:
    #     session['current_scenario'] = scenario
    # else:
    #     scenario = session.get('current_scenario', 'CICIDS2017')
    # 根据场景参数选择对应的JSON文件路径
    file_path = SCENARIO_FILE_PATHS.get(scenario)
    print(f"Received scenario: {scenario}")
    print(file_path)
    if not file_path:
        return jsonify({"error": "Invalid scenario"}), 400

    # 检查文件是否存在
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # 检查文件是否为空
    if os.path.getsize(file_path) == 0:
        return jsonify({"error": "JSON file is empty"}), 400

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            parsed_json = json.load(file)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"JSON file format error: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

    # 返回JSON数据
    return jsonify(parsed_json)














# @app.route('/inference/', methods=['GET'])
# def run_inference_api():
#     file_path = "/home/wang/下载/NetMamba/src/output/inference/CICIDS2017/test_stats.json"
#     if os.path.getsize(file_path) == 0:
#         print("JSON 文件为空！")
#     else:
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 parsed_json = json.load(file)
#                 raw_content = file.read()
#                 # print("文件内容:", raw_content)
#         except json.JSONDecodeError as e:
#             print(f"JSON 文件格式错误: {e}")
#         except FileNotFoundError:
#             print("文件不存在！")
#
#     return parsed_json


    # data = request.json
    # model_path = data.get('model_path')
    # dataset_path = data.get('dataset_path')
    # test_path = data.get('test_path')
    # output_dir = data.get('output_dir')
    # array_path = data.get('array_path')
    # model_name = data.get('model_name', 'net_mamba_classifier')
    # num_classes = data.get('num_classes', 7)
    # drop_path_rate = data.get('drop_path_rate', 0.1)
    # batch_size = data.get('batch_size', 64)
    # device = data.get('device', 'cuda')
    # num_workers = data.get('num_workers', 8)
    # pin_mem = data.get('pin_mem', True)
    # if_stat = data.get('if_stat', False)
    #
    # # 设置设备
    # device = torch.device(device)
    #
    # # 加载模型
    # model = load_model(model_name, model_path, device, num_classes, drop_path_rate)
    #
    # # 加载数据集
    # data_loader, dataset_classes = load_dataset(dataset_path, is_train=False, batch_size=batch_size,
    #                                             num_workers=num_workers, pin_mem=pin_mem, num_classes=num_classes)
    #
    # # 执行推理
    # result = inference(model, data_loader, device, output_dir, array_path, test_path, dataset_classes, if_stat)
    # # 创建 JSON 响应
    # response = jsonify(result)
    #
    # return response


@app.after_request
def add_header(response):
    # 禁用缓存
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)