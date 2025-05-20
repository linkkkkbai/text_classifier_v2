
import os
import torch
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.multi_head import ResNetCropMultiAttr
from utils.train_utils import load_config

SAVE_DIR = './visualization_results_Resnetcrop_v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/data_fast/danielslbai/text_classifier_v2/output/checkpoints/best_model.pth' # Your Model path
COCO_JSON = '/data_fast/danielslbai/text_classifier/utils/for_visual.json' # Your COCO annotations for visualization
ROOT_DIR = '/data_fast/danielslbai/text_classifier_v2/datasets/coco_format' # Your COCO dataset path


CLASS_NAMES_TYPE = ['Editable', 'NonEditable']
CLASS_NAMES_FONT = ['Regular', 'Bold']
CLASS_NAMES_ITALIC = ['No', 'Yes']

def load_resnet_model(model_path, device):
    cfg = load_config("/data_fast/danielslbai/text_classifier_v2/configs/base.yaml")
    model = ResNetCropMultiAttr(cfg)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
def visualize_and_save(orig_img, bboxes, preds, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_img)
    ax = plt.gca()
    for i, box in enumerate(bboxes):
        x, y, w, h = [int(round(v)) for v in box]
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        type_idx, font_idx, italic_idx = preds[i]
        label_str = f'Type:{CLASS_NAMES_TYPE[type_idx]}, Font:{CLASS_NAMES_FONT[font_idx]}, Italic:{CLASS_NAMES_ITALIC[italic_idx]}'
        ax.text(x, y-2, label_str, fontsize=6, color='blue', backgroundcolor='white')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
def main():
    import os, json
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    os.makedirs(SAVE_DIR, exist_ok=True)
    # 加载COCO标注
    with open(COCO_JSON, 'r') as f:
        coco_data = json.load(f)
    imgid_to_imginfo = {img['id']: img for img in coco_data['images']}
    imgid_to_anns = {}
    for ann in coco_data['annotations']:
        imgid_to_anns.setdefault(ann['image_id'], []).append(ann)
    image_ids = list(imgid_to_imginfo.keys())
    transform = get_transform()
    model = load_resnet_model(MODEL_PATH, DEVICE)

    for img_id in tqdm(image_ids, desc="Visualizing"):
        img_info = imgid_to_imginfo[img_id]
        anns = imgid_to_anns.get(img_id, [])
        img_path = os.path.join(ROOT_DIR, img_info['file_name'])
        orig_img_pil = Image.open(img_path).convert('RGB')
        orig_img_np = np.array(orig_img_pil)
        bboxes = [ann['bbox'] for ann in anns]
        if len(bboxes) == 0:
            continue

        crops = []
        for bbox in bboxes:
            x, y, w, h = [int(round(v)) for v in bbox]
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_img_pil.width, x2)
            y2 = min(orig_img_pil.height, y2)
            if x2 <= x1 or y2 <= y1:
                # 跳过异常框
                crops.append(torch.zeros(3, 224, 224))
                continue
            crop = orig_img_pil.crop((x1, y1, x2, y2))
            crop = transform(crop)
            crops.append(crop)
        crops_tensor = torch.stack(crops, dim=0).to(DEVICE)  # [N, 3, 224, 224]

        with torch.no_grad():
            outputs = model(crops_tensor)
            type_idx = torch.argmax(outputs['type'], dim=1).cpu().numpy()
            font_idx = torch.argmax(outputs['font'], dim=1).cpu().numpy()
            italic_idx = torch.argmax(outputs['italic'], dim=1).cpu().numpy()
            preds = list(zip(type_idx, font_idx, italic_idx))

        save_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
        visualize_and_save(orig_img_np, bboxes, preds, save_path)

if __name__ == '__main__':
    main()

