import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

from models.multi_head import ResNetCropMultiAttr
from losses.focal_loss import MultiAttributeLoss
from datasets.coco_loader import build_dataloader
from utils.train_utils import load_config

def evaluate(model, dataloader, criterion, device, class_names_dict=None, save_dir=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    losses_by_type = {'type': 0.0, 'font': 0.0, 'italic': 0.0, 'total': 0.0}
    num_batches_by_type = {'type': 0, 'font': 0, 'italic': 0, 'total': 0}
    correct = {'type': 0, 'font': 0, 'italic': 0}
    total = {'type': 0, 'font': 0, 'italic': 0}
    all_labels = {'type': [], 'font': [], 'italic': []}
    all_preds = {'type': [], 'font': [], 'italic': []}

    attr_list = ['type', 'font', 'italic']

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 支持多种batch格式
            if isinstance(batch, dict):
                # 适配dict格式（如Swin/MaskRCNN等）
                if 'image' in batch:
                    images = batch['image'].to(device)
                else:
                    images = batch['images'].to(device)
                # labels为dict或tensor
                if isinstance(batch.get('labels', None), dict):
                    labels = {k: v.to(device) for k, v in batch['labels'].items()}
                else:
                    # labels为tensor [N,3]
                    labels = batch['labels'].to(device)
                # boxes可选
                boxes = batch.get('boxes', None)
                if boxes is not None:
                    boxes = boxes.to(device)
            else:
                # 适配tuple格式（如CropBBoxDataset）
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(device)
                if isinstance(labels, dict):
                    labels = {k: v.to(device) for k, v in labels.items()}
                else:
                    labels = labels.to(device)
                boxes = None

            # 推理
            if boxes is not None:
                predictions = model(images, boxes)
            else:
                predictions = model(images)

            # 损失
            if callable(criterion):
                loss, losses = criterion(predictions, labels)
            else:
                loss = criterion(predictions, labels)
                losses = {'type': loss, 'font': loss, 'italic': loss, 'total': loss}

            total_loss += loss.item()
            num_batches += 1

            for k in attr_list:
                v = losses[k]
                losses_by_type[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
                num_batches_by_type[k] += 1

            if 'total' in losses:
                v = losses['total']
                losses_by_type['total'] += v.item() if isinstance(v, torch.Tensor) else float(v)
                num_batches_by_type['total'] += 1
            else:
                losses_by_type['total'] += loss.item()
                num_batches_by_type['total'] += 1

            # 统一labels格式：dict或[N,3] tensor
            if isinstance(labels, dict):
                for attr in attr_list:
                    pred_classes = torch.argmax(predictions[attr], dim=1)
                    correct[attr] += (pred_classes == labels[attr]).sum().item()
                    total[attr] += labels[attr].numel()
                    all_labels[attr].append(labels[attr].cpu().numpy())
                    all_preds[attr].append(pred_classes.cpu().numpy())
            else:
                # labels: [N,3] tensor
                for i, attr in enumerate(attr_list):
                    pred_classes = torch.argmax(predictions[attr], dim=1)
                    correct[attr] += (pred_classes == labels[:, i]).sum().item()
                    total[attr] += labels[:, i].numel()
                    all_labels[attr].append(labels[:, i].cpu().numpy())
                    all_preds[attr].append(pred_classes.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_losses = {k: (losses_by_type[k] / num_batches_by_type[k] if num_batches_by_type[k] > 0 else 0.0) for k in losses_by_type}
    accuracy = {attr: correct[attr] / total[attr] if total[attr] > 0 else 0.0 for attr in correct}

    # 混淆矩阵
    if class_names_dict is None:
        class_names_dict = {
            'type': ["Editable", "NonEditable"],
            'font': ["Regular", "Bold"],
            'italic': ["No", "Yes"]
        }

    for attr in attr_list:
        labels_flat = np.concatenate(all_labels[attr])
        preds_flat = np.concatenate(all_preds[attr])
        cm = confusion_matrix(labels_flat, preds_flat)
        print(f"\nConfusion Matrix for {attr}:\n{cm}")

        # 可视化
        plt.figure(figsize=(5, 4))
        class_names = class_names_dict[attr]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {attr}')
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'confusion_matrix_{attr}.png'))
        plt.close()
    return avg_loss, avg_losses, accuracy

def measure_inference_speed(model, dataloader, device, num_batches=50):
    model.eval()
    times = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, dict):
                images = batch['image'].to(device) if 'image' in batch else batch['images'].to(device)
                boxes = batch.get('boxes', None)
                if boxes is not None:
                    boxes = boxes.to(device)
            else:
                images = batch[0].to(device)
                boxes = None
            start = time.time()
            if boxes is not None:
                _ = model(images, boxes)
            else:
                _ = model(images)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    avg_time = np.mean(times)
    print(f"平均单张推理时间: {avg_time * 1000:.2f} ms")
    print(f"FPS: {1.0 / avg_time:.2f}")
    return avg_time, 1.0 / avg_time

def measure_gpu_memory(model, dataloader, device, num_batches=10):
    if not device.startswith('cuda'):
        print("仅支持CUDA设备的显存测量")
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, dict):
                images = batch['image'].to(device) if 'image' in batch else batch['images'].to(device)
                boxes = batch.get('boxes', None)
                if boxes is not None:
                    boxes = boxes.to(device)
            else:
                images = batch[0].to(device)
                boxes = None
            if boxes is not None:
                _ = model(images, boxes)
            else:
                _ = model(images)
    max_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    print(f"推理最大显存占用: {max_mem:.2f} MB")
    return max_mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/data_fast/danielslbai/text_classifier_v2/configs/base.yaml", help='config yaml path')
    parser.add_argument('--model', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--save_dir', type=str, default='./eval_results', help='dir to save confusion matrix')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载配置、模型、数据
    cfg = load_config(args.config)
    model = ResNetCropMultiAttr(cfg)
    checkpoint = torch.load(args.model, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    # 构建dataloader
    dataloader = build_dataloader(cfg,json_file=cfg.data.val_json, is_train=False, batch_size=args.batch_size, num_workers=4)

    # 损失函数
    criterion = MultiAttributeLoss(cfg)

    avg_loss, avg_losses, accuracy = evaluate(
        model, dataloader, criterion, args.device, save_dir=args.save_dir
    )
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Losses by type: {avg_losses}")
    print(f"Accuracy: {accuracy}")
    avg_time, fps = measure_inference_speed(model, dataloader, args.device)
    max_mem = measure_gpu_memory(model, dataloader, args.device)
    print(f"AVG_time: {avg_time:.4f}s, FPS: {fps:.2f}, Max_mem: {max_mem:.2f} MB")

if __name__ == '__main__':
    main()