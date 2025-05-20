import torch
import yaml
import os
from types import SimpleNamespace
import json
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 递归将字典转换为SimpleNamespace对象
    def dict_to_namespace(d):
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace
    
    return dict_to_namespace(config_dict)



def prepare_batch(batch, device):
    images = batch['image'].to(device)
    batch_boxes = []
    batch_idx = []
    type_labels = []
    font_labels = []
    italic_labels = []
    meta = []  # 新增

    for i, (boxes, t, f, it, img_id) in enumerate(zip(
        batch['bboxes'], batch['type_labels'], batch['font_labels'], batch['italic_labels'], batch['image_id']
    )):
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        if len(boxes) == 0:
            continue
        batch_boxes.append(boxes)
        batch_idx.append(torch.full((boxes.shape[0], 1), i, dtype=boxes.dtype))
        type_labels.append(t)
        font_labels.append(f)
        italic_labels.append(it)
        # 记录每个bbox的image_id、在该图片中的索引、batch_idx
        for j in range(boxes.shape[0]):
            meta.append({
                'image_id': img_id.item() if isinstance(img_id, torch.Tensor) else img_id,
                'box_idx_in_image': j,
                'batch_idx': i  # 新增
            })

    if batch_boxes:
        boxes = torch.cat([torch.cat([idx, box], dim=1) for idx, box in zip(batch_idx, batch_boxes)], dim=0)
        boxes = boxes.to(device)
        type_labels = torch.cat(type_labels).to(device)
        font_labels = torch.cat(font_labels).to(device)
        italic_labels = torch.cat(italic_labels).to(device)
    else:
        boxes = torch.empty((0, 5)).to(device)
        type_labels = torch.empty((0,), dtype=torch.long).to(device)
        font_labels = torch.empty((0,), dtype=torch.long).to(device)
        italic_labels = torch.empty((0,), dtype=torch.long).to(device)
    labels = {
        'type': type_labels,
        'font': font_labels,
        'italic': italic_labels,
    }
    return images, boxes, labels, meta

def validate(model, dataloader, criterion, device, writer=None, global_step=None, class_names_dict=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    num_batches = 0
    losses_by_type = {'type': 0.0, 'font': 0.0, 'italic': 0.0, 'total': 0.0}
    num_batches_by_type = {'type': 0, 'font': 0, 'italic': 0, 'total': 0}
    correct = {'type': 0, 'font': 0, 'italic': 0}
    total = {'type': 0, 'font': 0, 'italic': 0}

    all_labels = {'type': [], 'font': [], 'italic': []}
    all_preds = {'type': [], 'font': [], 'italic': []}

    attr2idx = {'type': 0, 'font': 1, 'italic': 2}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            crops, labels = batch
            crops = crops.to(device)
            labels = labels.to(device)
            predictions = model(crops)
            loss, losses = criterion(predictions, labels)  # loss和losses都是mean

            total_loss += loss.item()
            num_batches += 1

            for k in ['type', 'font', 'italic']:
                v = losses[k]
                if isinstance(v, torch.Tensor):
                    losses_by_type[k] += v.item()
                    num_batches_by_type[k] += 1
                else:
                    losses_by_type[k] += float(v)
                    num_batches_by_type[k] += 1

            if 'total' in losses:
                v = losses['total']
                if isinstance(v, torch.Tensor):
                    losses_by_type['total'] += v.item()
                    num_batches_by_type['total'] += 1
                else:
                    losses_by_type['total'] += float(v)
                    num_batches_by_type['total'] += 1
            else:
                losses_by_type['total'] += loss.item()
                num_batches_by_type['total'] += 1

            # accuracy统计
            for attr in ['type', 'font', 'italic']:
                idx = attr2idx[attr]
                pred_classes = torch.argmax(predictions[attr], dim=1)
                correct[attr] += (pred_classes == labels[:, idx]).sum().item()
                total[attr] += labels[:, idx].numel()
                all_labels[attr].append(labels[:, idx].cpu().numpy())
                all_preds[attr].append(pred_classes.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_losses = {}
    for k in losses_by_type:
        if num_batches_by_type[k] > 0:
            avg_losses[k] = losses_by_type[k] / num_batches_by_type[k]
        else:
            avg_losses[k] = 0.0

    accuracy = {attr: correct[attr] / total[attr] if total[attr] > 0 else 0.0 for attr in correct}

    # 混淆矩阵
    for attr in ['type', 'font', 'italic']:
        labels_flat = np.concatenate(all_labels[attr])
        preds_flat = np.concatenate(all_preds[attr])
        cm = confusion_matrix(labels_flat, preds_flat)
        print(f"\nConfusion Matrix for {attr}:\n{cm}")

        # 可视化
        plt.figure(figsize=(5, 4))
        if class_names_dict is None:
            class_names_dict = {
                'type': ["Editable", "NonEditable"],
                'font': ["Regular", "Bold"],
                'italic': ["No", "Yes"]
            }
        class_names = class_names_dict[attr]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {attr}')
        plt.tight_layout()
        plt.show()

        if writer is not None and global_step is not None:
            fig = plt.gcf()
            writer.add_figure(f'Confusion_Matrix/{attr}', fig, global_step)
            plt.close(fig)
    return avg_loss, avg_losses, accuracy




def train_iterations(
    model, train_loader, val_loader, criterion, optimizer, device,
    total_iters, validate_interval, checkpoint_interval, output_dir, writer,
    scheduler=None,
    log_interval=10  # 新增参数：每多少步记录一次训练信息
):
    best_val_loss = float('inf')
    iter_count = 0
    train_iter = itertools.cycle(train_loader)
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    pbar = tqdm(total=total_iters, desc="Training (by iteration)")
    while iter_count < total_iters:
        model.train()
        batch = next(train_iter)
        optimizer.zero_grad()
        batch = next(train_iter)
        crops, labels = batch
        crops = crops.to(device)
        labels = labels.to(device)
        predictions = model(crops)
        loss, losses = criterion(predictions, labels)
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # 记录训练 loss、lr、各属性 loss 到 TensorBoard
        if iter_count % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/train', loss.item(), iter_count)
            writer.add_scalar('LR', current_lr, iter_count)
            for attr in ['type', 'font', 'italic']:
                v = losses[attr]
                writer.add_scalar(f'Loss_{attr}/train', v.mean().item() if hasattr(v, 'mean') else float(v), iter_count)

        # validate
        if (iter_count + 1) % validate_interval == 0:
            class_names_dict = {
                'type': ["editable", "noneditable"],
                'font': ["regular", "bold"],
                'italic': ["no", "italic"]
            }

            val_loss, val_losses_by_type, val_accuracy = validate(
                model, val_loader, criterion, device,
                writer=writer, global_step=iter_count, class_names_dict=class_names_dict
            )

            writer.add_scalar('Loss/val', val_loss, iter_count)
            for attr in ['type', 'font', 'italic']:
                writer.add_scalar(f'Loss_{attr}/val', val_losses_by_type[attr], iter_count)
                writer.add_scalar(f'Accuracy/{attr}/val', val_accuracy[attr], iter_count)
            print(f"[Iter {iter_count+1}] Val Loss: {val_loss:.4f}, Acc: {val_accuracy}")
            metrics_history['val_loss'].append({
                'iter': iter_count,
                'total': val_loss,
                **val_losses_by_type
            })
            metrics_history['val_accuracy'].append({
                'iter': iter_count,
                **val_accuracy
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, iter_count, best_val_loss,
                    os.path.join(output_dir, 'checkpoints', 'best_model.pth')
                )
        # checkpoint
        if (iter_count + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, iter_count, best_val_loss,
                os.path.join(output_dir, 'checkpoints', f'checkpoint_iter_{iter_count+1}.pth')
            )
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': optimizer.param_groups[0]['lr']
        })
        iter_count += 1
        pbar.update(1)
    pbar.close()
    writer.close()
    print("Training completed!")
    return metrics_history
def save_checkpoint(model, optimizer, epoch, best_val_loss, filename):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, filename)