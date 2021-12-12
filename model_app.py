import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from ensemble_boxes2 import weighted_boxes_fusion
from effdet import get_efficientdet_config
from effdet import create_model_from_config
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import timm
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

transform = A.Compose([
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255, p=1.0),
                ToTensorV2(p=1.0)
                ])


def detect(im, device='cpu',  conf_thres=0.1):
    """[summary]

    Args:
        im (tensor): image you want to predict
        device (str, optional): Defaults to 'cpu'.
        conf_thres (float, optional): Minimun you want to keep. Defaults to 0.1.

    Returns:
        [type]: Dataframe of the differents finding and the prob of the classifier
    """
    w, h = im.size
    
    boxes = []
    scores = []
    labels = []
    
    
    #EFFDET PART
    im = im.resize((1024, 1024), Image.LANCZOS)
    
    image = np.array(im.convert('RGB')).astype('uint8')
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.int32)
    image_transformed = transform(image=image)
    img = image_transformed['image'].unsqueeze(0)
    images = img.to(device).float()
    
    base_config = get_efficientdet_config('tf_efficientdet_d4')
    base_config.image_size = (1024, 1024)
    net = create_model_from_config(base_config, bench_task='predict', bench_labeler=True,
                                            num_classes=14)                                       
    weights_dirs_effdet = [f'weights/weight_effdet/best_fold{i}_effdet.pth' for i in range(1)]
    with torch.no_grad():
        for weights_dir in weights_dirs_effdet :
            checkpoint = torch.load(weights_dir, map_location=device)
            net.load_state_dict(checkpoint)
            net.eval()
            outputs = net(images)
            box = outputs[0].detach().cpu().numpy()[:, :4]
            scr = outputs[0].detach().cpu().numpy()[:, 4]
            lbl = outputs[0].detach().cpu().numpy()[:, 5]

            selected = scr >= conf_thres
            boxes.append((box[selected].astype(np.int32) / 1024).tolist())
            scores.append(scr[selected].tolist())
            labels.append((lbl[selected].astype(np.int32)-1).tolist())
            
    # RCNN PART
    eff_net = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
    modules = list(eff_net.children())[:-3]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 1792
    del eff_net

    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                       aspect_ratios=((0.25, 0.5, 1.0, 2, 4),))
    model = FasterRCNN(backbone,
                      num_classes=15,
                      rpn_anchor_generator=anchor_generator)
    checkpoint = torch.load('weights/weight_rcnn/rcnn_best_fold0.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    
    with torch.no_grad():
        model.eval()
        outputs = model(images)
        box = outputs[0]['boxes'].data.cpu().numpy()
        scr = outputs[0]['scores'].data.cpu().numpy()
        lbl = outputs[0]['labels'].data.cpu().numpy()
        
        selected = scr >= conf_thres
        boxes.append((box[selected].astype(np.int32) / 1024).tolist())
        scores.append(scr[selected].tolist())
        labels.append((lbl[selected].astype(np.int32)-1).tolist())
    

    
    # New image Size for YOLO and Classifier
    im = im.resize((640, 640), Image.LANCZOS)
    
    image = np.array(im.convert('RGB')).astype('uint8')
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.int32)
    image_transformed = transform(image=image)
    img = image_transformed['image'].unsqueeze(0)
    images = img.to(device).float()
    
    # CLASSIFIER PART
    classifier = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=1)
    checkpoint = torch.load('weights/best_vin_fold0_classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint)
    
    sigmoid = nn.Sigmoid()
    classifier.eval()
    with torch.no_grad():
        outputs = sigmoid(classifier(images)).data.cpu().numpy()
        prob_classifier = round(outputs[0][0], 4)
    
    # YOLO PART     
    import os
    cwd = os.getcwd()
    os.chdir('yolov5')
    from utils.general import non_max_suppression
    weights_dirs_yolo = [f'weights/best_fold{i}_yolo.pt' for i in range(1)]
    
    with torch.no_grad():
        for weights_dir in weights_dirs_yolo:
            model = torch.load(weights_dir, map_location=device)['model'].float()
            model.to(device).eval()

            preds = model(img, augment=False)
            iou_thres = 0.5
            pred_nms = non_max_suppression(preds[0], conf_thres, iou_thres)
            pred = pred_nms[0].data.cpu().numpy()
            boxes.append((pred[:, :4] / 640).tolist())
            scores.append(pred[:, 4].tolist())
            labels.append(pred[:, 5].tolist())
        
        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.3)
        
        dict = {'class_id': labels_wbf, 'scores': scores_wbf, 'x_min': boxes_wbf[:,0]*w,
                'y_min': boxes_wbf[:,1]*h, 'x_max': boxes_wbf[:,2]*w, 'y_max': boxes_wbf[:,3]*h}
    os.chdir(cwd)
    return pd.DataFrame(dict), round(1-prob_classifier, 4)