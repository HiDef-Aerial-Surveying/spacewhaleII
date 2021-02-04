##########################################################################################
############SPACEWHALE Project: Whales detection based on deep learning method###########
#######testing step
####Author: Amel Ben Mahjoub and Grant Humphries
####15.01.2021
##########################################################################################
import torch
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
import time
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Parse args
parser = argparse.ArgumentParser(description='Faster R-CNN algorithm for whales detection-- testing step')
parser.add_argument('--model_path',type=str, help="path to download the training model")
parser.add_argument('--input_path',type=str, help="path to the large input image")
parser.add_argument('--output_path',type=str, help="path to save the output detections")
parser.add_argument('--num_classes',type=int,default=5, help="number of classes")
parser.add_argument('--box_score',type=float,default=0.01, help="box score thresh")
parser.add_argument('--box_nms',type=float,default=0.2, help="box Non Maximum Suppression thresh")
parser.add_argument('--chopsize',default=800,type=int, help="size of sliding window")
parser.add_argument('--overlap',default=0.5,type=float, help="overlapping thresh")

args = parser.parse_args()

#####transformations
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


def get_object_detection_model(num_classes): 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800, box_score_thresh=args.box_score, box_nms_thresh=args.box_nms) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model


#device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 5 classes
num_classes =args.num_classes
model = get_object_detection_model(num_classes)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cpu_device = torch.device('cpu')
####loading the model
model.load_state_dict(torch.load(args.model_path))
model.to(device)


class Orthoimage_Data():
    def __init__(self, image_ids, images, transforms=None):
        super().__init__()
        self.image_ids = image_ids        
        self.images = images
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]                
        image = self.images[index]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)   
        image /= 255.0        
        target = {}
        target['image_id'] = torch.tensor([index])
        if self.transforms is not None:
            image, target = self.transforms(image,target)
        return image, image_id

    def __len__(self):
        return len(self.image_ids)

#####sliding widows with size split_width X split_height over the image overlapping with 0.5 
overlap=args.overlap
def ortho_image_splitter(Im_name,chopsize,overlap):
    images = []
    image_ids = []
    img = Image.open(Im_name)
    width, height = img.size
    stride = int(chopsize * (1-overlap))

    for x0 in range(0, width, stride):
        for y0 in range(0, height, stride):
            outerRect = (x0, y0,
                    x0+chopsize if x0+chopsize <  width else  width - 1,
                    y0+chopsize if y0+chopsize < height else height - 1)        
            x0,y0,x1,y1 = outerRect
            crop_img = img.crop(outerRect)
            crop_img = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
            xdiff = x1-x0
            ydiff = y1-y0
            xpad = 800 - xdiff
            ypad = 800 - ydiff
            crop_img = cv2.copyMakeBorder(crop_img, 0, ypad, xpad, 0, cv2.BORDER_CONSTANT)               
            savename = 'chop.x0%03d.y0%03d.x1%03d.y1%03d.png' % ( x0, y0, x1, y1) 
            images.append(crop_img)
            image_ids.append(savename)
    return images, image_ids



directory = args.input_path

start_test_perlargeimg = time.time()
for filename in os.listdir(directory):
    k=1   
    a=1
    
    if filename.endswith(".png"): 
        with open(args.output_path +'names/'+ filename[19:-29] + '.'  +'names.txt', 'w') as f:
            images, image_ids = ortho_image_splitter(directory+filename,chopsize=args.chopsize, overlap = args.overlap)    
            for j in range(len(images)):
              image_id = [image_ids[j]]  
              image = [images[j]]
              test_dataset = Orthoimage_Data(image_id,image,get_transform(train=False))
              test_data_loader = torch.utils.data.DataLoader(
                  test_dataset,
                  batch_size=10,
                  shuffle=False,
                  num_workers=0,
                  collate_fn=utils.collate_fn
              )
              image_id=str(image_id)
              model.eval()
              cpu_device = torch.device("cpu")
              imgs, img_ids = next(iter(test_data_loader))
              imgs = list(img.to(device) for img in imgs)
              output = model(imgs)
              output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
              boxes = output[0]['boxes'].data.cpu().numpy()
              scores = output[0]['scores'].data.cpu().numpy()
              classes= output[0]['labels'].data.cpu().numpy()
              for img in imgs:
                  sn = ""
                  dd= ""
                  img = img.permute(1,2,0)
                  img = (img * 255).byte().data.cpu()  # * 255, float to 0-255
                  img = np.array(img)  # tensor → ndarray
                  bx=0
                  for y in range(output[0]['boxes'].cpu().shape[0]):
                      xmin = round(output[0]['boxes'][y][0].item())
                      ymin = round(output[0]['boxes'][y][1].item())
                      xmax = round(output[0]['boxes'][y][2].item())
                      ymax = round(output[0]['boxes'][y][3].item())
                      label = output[0]['labels'][y].item()
                      if (xmax-xmin)>= 3 and (xmax-xmin)< 120 and (ymax-ymin)>= 3 and (ymax-ymin)< 120:
                            
                          if label == 1:
                              dd+='box' + str(bx)
                              cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=1)
                              cv2.putText(img, str(bx), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),thickness=1)
                            
                              sn+= str(bx) + '_' + str(xmin) + '_' + str(ymin) + '_'+ str(xmax) + '_' + str(ymax) + '_'
    
                    
                              bx+=1
                  if  dd :
                      plt.imsave(args.output_path+filename[19:-29]+ '.' + str(a)+ '.'+ dd+'.png', img)
                      f.write(filename[19:-29]+ '.' +str(a)+ image_id[6:-6] + '.'+sn + '\n')
                      a+=1


end_test = time.time()
print("test time per one large image", end_test - start_test_perlargeimg)  
  