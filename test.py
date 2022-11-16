

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
    return [x- (width/2), y-(height/2), x + (width/2), y + (height/2)]


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        inputs = feature_extractor(images=Image.fromarray(color_image), return_tensors="pt")
        inputs = prepare_inp(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze()[:, 0]
        # print(logits.shape)

        idx = torch.argmax(logits, axis=-1)
        # print(idx)
        bboxes = outputs.pred_boxes.squeeze()
        # print(bboxes.shape)
        image = Image.fromarray(color_image)
        img1 = ImageDraw.Draw(image)


        out_bbox = bboxes[idx].reshape(-1,4).detach().cpu().numpy()
        
        final_bbox = [resize_align_bbox(i,*or_size,*target_size) for i in out_bbox]
        # print(final_bbox)


        for bbox in final_bbox:
            bbox = list(bbox)
            img1.rectangle([tuple(bbox[:2]), tuple(bbox[2:])])

        print(color_image.shape)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((depth_colormap, np.asarray(image)))
        else:
            images = np.hstack((depth_colormap, np.asarray(image)))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()