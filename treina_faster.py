import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms as T

class CocoDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

        num_objs = len(anns)
        boxes = []
        labels = []

        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# Path to your dataset
root = os.path.join(os.getcwd(), 'train')
annotation = os.path.join(root, '_annotations.coco.json')

# Create the dataset and dataloader
dataset = CocoDataset(root, annotation, transforms=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create the model
num_classes = 3  # 1 class (person) + background
model = create_model(num_classes)
model.to(device)

# Parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 100

# Directory to save models
output_dir = os.getcwd()
os.makedirs(output_dir, exist_ok=True)

best_loss = np.inf
model_path = os.path.join(output_dir, "modelo_treinado_faster.pth")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item()}")
        
    epoch_loss /= len(data_loader)
    print(f"Epoch {epoch} finished with loss {epoch_loss}")

    if epoch_loss < best_loss:
        print(f"Best Loss Found: {epoch_loss}, saving model...")
        torch.save(model.state_dict(), model_path)
        best_loss = epoch_loss

    lr_scheduler.step()
