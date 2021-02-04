import cv2
import numpy as np

# Load Yolo model
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(100, 255, size=(len(classes), 3))

videoPath = 'walking.mp4'
camera = cv2.VideoCapture(videoPath)
_,img = camera.read()
height, width, channels = img.shape

# Fetching video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_video.avi", fourcc , 25, (width, height))

camera = cv2.VideoCapture(videoPath)

while True:
    _,img = camera.read()
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Drawing informations
    class_ids = []
    confidences = []
    boxes = []
    for outt in outs:
        for detection in outt:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            font_scale = 0.7
            thickness = 2
            text = "{}: {:.1f}%".format(label,
				confidences[i] * 100)

            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]

            # set the text start position
            text_offset_x = x
            text_offset_y = y + 20

            # make the coords of the box with a small padding of two pixels
            txt_box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 8))
            cv2.rectangle(img, txt_box_coords[0], txt_box_coords[1], color, cv2.FILLED)
            cv2.putText(img, text, (text_offset_x + 2, text_offset_y - 5), font, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    out.write(img);        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

#Output the video   
out.release()
camera.release()
cv2.destroyAllWindows()