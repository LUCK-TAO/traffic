import argparse
import datetime
import json
from collections import Counter

import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import os
from torchvision import datasets, transforms
import util.misc as misc
import models_net_mamba
from engine import evaluate, evaluate_speed_test

def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba inference for traffic classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    # Model parameters
    parser.add_argument('--model', default='net_mamba_classifier', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--data_path', default='./Net-Mamba-data/root/Vim/dataset/CICDDOS2019/dataset_sampled', type=str,   #New_Traffic
                        help='dataset path')
    parser.add_argument('--array_path', default='/home/wang/下载/NetMamba/dataset/CICDDOS2019/array_sampled', type=str,
                        help='json_dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output/inference-CHANGZHOU/CICDDOS2019', help='path where to save predictions')
    parser.add_argument('--device', default='cuda', help='device to use for inference')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/home/wang/下载/NetMamba/src/output/finetune/CICDDOS2019-CHANGZHOU/checkpoint-best.pth',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # spatio temporal features
    parser.add_argument('--if_stat', action='store_true',
                        help='whether to use statistical features')
    # evaluation settings
    parser.add_argument('--speed_test', action='store_true')

    return parser


def build_dataset(is_train, args):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(args.data_path, 'train' if is_train else 'test')
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset, root


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_val, val_root = build_dataset(is_train=True, args=args)

    # Create DataLoader for inference (no target labels)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Load the model
    model = models_net_mamba.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.1
    )
    try:
        checkpoint = torch.load(args.resume, map_location=device)
        print("Loading model checkpoint from:", args.resume)
        model.load_state_dict(checkpoint['model'], strict=False)
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        print("Evaluate the model from scratch.")
    model.to(device)

    # Perform inference only
    model.eval()  # Set the model to evaluation mode
    predictions = []

    # Iterate over the data_loader for inference
    with torch.no_grad():
        for data in data_loader_val:
            inputs, _ = data  # We don't need the labels here
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get the predicted class
            predictions.extend(preds.cpu().numpy())

    # Convert numpy int64 to Python int before saving to JSON
    predictions = [int(pred) for pred in predictions]

    # Count occurrences of each class
    class_counts = Counter(predictions)

    # Save predictions and class counts to a file
    predictions_file = os.path.join(args.output_dir, 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)

    class_counts_file = os.path.join(args.output_dir, 'class_counts.json')
    with open(class_counts_file, 'w') as f:
        json.dump(class_counts, f)

    print(f"Inference completed. Predictions saved to {predictions_file}")
    print(f"Class counts saved to {class_counts_file}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
