from ultralytics import YOLO
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

yolo_model = YOLO('yolov8n.pt')

sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

class_colors = {}

input_video_path = 'data/video/序列 01.mp4'
output_video_path = 'data/video/out01.mp4'
cap = cv2.VideoCapture(input_video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

alpha = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_copy = image_rgb.copy()

    results = yolo_model.predict(image_rgb)
    detections = results[0].boxes

    boxes = []
    labels = []
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        class_id = int(detection.cls[0].item())
        boxes.append([x1, y1, x2, y2])
        labels.append(class_id)

        if class_id not in class_colors:
            class_colors[class_id] = np.random.randint(0, 255, size=3)

    predictor.set_image(image_rgb)

    all_masks = []

    for box, label in zip(boxes, labels):
        input_box = np.array(box)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        all_masks.append((masks[0], label))


    for mask, label in all_masks:
        color = class_colors[label]
        colored_mask = (mask[:, :, None] * color).astype(np.uint8)

        image_copy = np.where(mask[:, :, None],
                              (1 - alpha) * image_copy + alpha * colored_mask,
                              image_copy).astype(np.uint8)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        color = [int(c) for c in class_colors[label]]
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

    image_bgr = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)

    out.write(image_bgr)

cap.release()
out.release()
print("视频检测与分割处理完成！")
