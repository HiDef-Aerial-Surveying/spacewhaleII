##########################################################################################
############SPACEWHALE Project: Whales detection based on deep learning method###########
#######training step
####Author: Amel Ben Mahjoub
####15.01.2021
##########################################################################################
import torch
import os
import argparse
import numpy as np
import cv2
from xml.dom.minidom import parse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
from PIL import Image
import time



# Parse args
parser = argparse.ArgumentParser(description='Faster R-CNN algorithm for whales detection-- training step')
parser.add_argument('--model_path',type=str, help="path to save the training model")
parser.add_argument('--num_classes',type=int,default=5, help="number of classes")
parser.add_argument('--batch_size',type=int, help="used batch size")
parser.add_argument('--num_epochs',type=int,default=30, help="number of epochs for the training step")
args = parser.parse_args()


class MarkDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
 
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")        

        dom = parse(bbox_xml_path)
        data = dom.documentElement
        objects = data.getElementsByTagName('object')        
        boxes = []
        labels = []
        for object_ in objects:
            # Get the contents of the label
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  
            if name == 'whale':
                labels.append(1)  
            elif name == 'cloud':
                labels.append(2)
            elif name == 'wave':
                labels.append(3) 
            elif name == 'ship':
                labels.append(4)
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])        
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)        
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:

            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)

###define the pre-trained model resnet50
def get_object_detection_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800, box_nms_thresh=0.2)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model


########data transformstions
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:

        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.Adjust_contrast())
        transforms.append(T.Adjust_brightness())
        transforms.append(T.Adjust_saturation())
        transforms.append(T.lighting_noise())
      
    return T.Compose(transforms)



start_train = time.time()
####data used for the training step: minke whales + 4 satellite images and ships images
root = r'PascalVOC-export-sup'
root1 = r'PascalVOC-pansh_for_image1'
root2 = r'PascalVOC-pansh_for_image2'
root3 = r'PascalVOC-pansh_for_image3'
root5 = r'PascalVOC-pansh_for_image5'
root_ship = r'PascalVOC-MAXAR-ships'


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = args.num_classes  # we have 5 classes  background + whale + cloud + waves and ship


dataset = MarkDataset(root, get_transform(train=True))
dataset1 = MarkDataset(root1, get_transform(train=True))
dataset2 = MarkDataset(root2, get_transform(train=True))
dataset3 = MarkDataset(root3, get_transform(train=True))
dataset5 = MarkDataset(root5, get_transform(train=True))
dataset_ship = MarkDataset(root_ship, get_transform(train=True))


num_epochs =args.num_epochs

list_ap=[]
model = get_object_detection_model(num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
#####optimized hyperparameters defined after applying optuna approach
optimizer =torch.optim.Adam(params, lr=1.22e-05,weight_decay=0.0005)


##########concatenate all the data for the cross validation method
concate_dataset1 = torch.utils.data.ConcatDataset([dataset, dataset1])
concate_dataset2 = torch.utils.data.ConcatDataset([concate_dataset1, dataset2])
concate_dataset3 = torch.utils.data.ConcatDataset([concate_dataset2, dataset3])
concate_dataset4 = torch.utils.data.ConcatDataset([concate_dataset3, dataset_ship])
print("len(concate_dataset)", len(concate_dataset4))
#####training step
train_loader = torch.utils.data.DataLoader(concate_dataset4, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
start_train_perepoch = time.time()
for epoch in range(num_epochs):   
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)    
end_train = time.time()
print("training time ", end_train - start_train_perepoch)

######saving the trained model
torch.save(model.state_dict(), args.model_path +'resnet50_img123_5class.pth')

print('')
print('==================================================')
print('')
print("That's it!")



