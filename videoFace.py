import cv2
import numpy as np
from scipy import spatial
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/wrong.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# Load Yolo model
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(100, 255, size=(len(classes), 3))

videoPath = 'wrong.mp4'
camera = cv2.VideoCapture(videoPath)
fps = camera.get(cv2.CAP_PROP_FPS)
_,img = camera.read()
height, width, channels = img.shape

# Fetching video
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter(videoPath.split('.')[0] + 'O.mp4', fourcc , fps, (width, height))

camera = cv2.VideoCapture(videoPath)

conf_threshold = 0.5

lastFrame = {}

def stablizer(last, coord, label, minDistance, isOn):
    if(not isOn):
        return coord
    
    if(label not in last):
        return coord
    
    points = last[label]
    tree = spatial.KDTree(points)
    result = tree.query([coord])
    
    if(result[0][0] < minDistance):
        return points[result[1][0]]
    
    return coord

thickness = 4
font_scale = 1

while True:
    _,img = camera.read()
    height, width, channels = img.shape

    if not _:
        print("End of video")
        break

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = img[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)

    	if conf>=4 and conf <= 85:
    		#print(conf)
    		#print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2

            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]

            text_offset_x, text_offset_y = x, y+h
           
            txt_box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 8))
            cv2.rectangle(img, txt_box_coords[0], txt_box_coords[1], (0,0,0), cv2.FILLED)
            cv2.putText(img, text, (text_offset_x + 2, text_offset_y - 5), font, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

            cv2.putText(img, name, (x,y+h-thickness), font, 1, color, stroke, cv2.LINE_AA)
           

    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), (0,0,0), thickness)

    # Drawing informations
    class_ids = []
    confidences = []
    boxes = []
    for outt in outs:
        for detection in outt:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    currentFrame = {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            x, y = stablizer(lastFrame, (x, y), label, 5, True)
            if(label not in currentFrame):
                currentFrame[label] = [(x, y)]
            else:
                currentFrame[label].append((x, y))


            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness + 2)
            

            showLabel = label
            #if(label == 'person'):
                #showLabel = 'nerd'
            text = "{}: {:.0f}%".format(showLabel,
				confidences[i] * 100)

            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]

            # set the text start position
            text_offset_x, text_offset_y = x, y
            #text_offset_y = text_offset_y + text_height + thickness
           
            # make the coords of the box with a small padding of two pixels
            txt_box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 8))
            cv2.rectangle(img, txt_box_coords[0], txt_box_coords[1], color, cv2.FILLED)
            cv2.putText(img, text, (text_offset_x + 2, text_offset_y - 5), font, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    lastFrame = currentFrame

    out.write(img);        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

#Output the video   
out.release()
camera.release()
cv2.destroyAllWindows()