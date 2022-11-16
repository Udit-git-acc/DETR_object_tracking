import cv2
import pyrealsense2 as rs
import numpy as np
import cv2
from torchvision import transforms
import torch
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from PIL import Image, ImageDraw
import numpy as np
import cv2

def prepare_inp(dct):
  dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dct['pixel_values'] = dct['pixel_values'].to(dvc)
  dct['pixel_mask'] = dct['pixel_mask'].to(dvc)

  return dct


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50', num_labels=1,
                                                          ignore_mismatched_sizes=True, do_resize=False)
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50').to(device)
model.load_state_dict(torch.load('model_0.193.pt', map_location=device))


or_size = (1,1)
target_size = (640, 480)

def resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h):
    x_scale = target_w / orig_w
    y_scale = target_h / orig_h
    orig_left, orig_top, orig_right, orig_bottom = bbox
    x = int(np.round(orig_left * x_scale))
    y = int(np.round(orig_top * y_scale))
    width = int(np.round(orig_right * x_scale))
    height = int(np.round(orig_bottom * y_scale))
    return [x, y, (width+height)/4]

#hello world

camera = cv2.VideoCapture(0)
while True:
    # Get a frame
    ret_val, frame = camera.read()
    # Show the frame
    cv2.imshow('Webcam Video Feed', frame)
    # print(np.array(Image.fromarray(frame)).shape)

    inputs = feature_extractor(images=frame.transpose(2, 0, 1)/255, return_tensors="pt")
    inputs = prepare_inp(inputs)


    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze()[:, 0]
    # print(logits.shape)

    idx = torch.argmax(logits, axis=-1)
    # print(idx)
    bboxes = outputs.pred_boxes.squeeze()
    # print(bboxes.shape)
    #image = Image.fromarray(frame)
    #img1 = ImageDraw.Draw(image)


    out_bbox = bboxes[idx].reshape(-1,4).detach().cpu().numpy()
    
    final_bbox = [resize_align_bbox(i,*or_size,*target_size) for i in out_bbox]
    # print(final_bbox)


    for bbox in final_bbox:
        bbox = list(bbox)
        x,y,radius=bbox
        cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
        #img1.rectangle([tuple(bbox[:2]), (bbox[0]+2,bbox[1]+2)])

    # Stop the capture by hitting the 'esc' key
    cv2.imshow("preds:",frame)
    if cv2.waitKey(1) == 27:
        break
# Dispose of all open windows
cv2.destroyAllWindows()
