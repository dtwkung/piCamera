import cv2 as cv
import numpy as np
import time



class moveDetect:
    def __init__(self, model='yolo'):
        # start video camera
        self.vs = cv.VideoCapture(0)
        if not self.vs.isOpened():
            raise RuntimeError('Could not start camera.')
        self.f1 = self.vs.read()[1]
        self.f2 = None

        # Set the count for number of cycles the reference image gets updated
        self.UPDATE_CYCLE = 10
        self.cnt = 0



        # Get the object detection model
        if model == 'yolo':
            self.model = yolo(threshold=0.4, size=416)
        elif model == 'mobilenet':
            self.model = mobilenet(threshold=0.5, pbPath="mobilenet_v2_frozen_inference_graph.pb", pbtxtPath="mobilenet_v2_graph.pbtxt")
        else:
            print("Unknown model")
            return

    def getFrame(self):
        self.f2 = self.vs.read()[1]
        diff = cv.absdiff(self.f1, self.f2)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)[1]
        dilated = cv.dilate(gray, np.ones((7, 7)), iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        coord = None
        maxArea = 0


        movements = []
        for contour in contours:
            if cv.contourArea(contour) < 9000:
                continue
            (x, y, w, h) = cv.boundingRect(contour)
            movements.append((x, y, x+w, y+h))
        img = np.copy(self.f2)
        #Show the largest movement ROI
        if len(movements) > 0:
            for movement in movements:
                x, y, w, h = movement
                mover = img[y:h, x:w]
                prediction = self.model.predict(mover)
                if prediction is not None:
                    box, label, confidence = prediction
                    x1, y1, w1, h1 = box.astype(int)
                    cv.rectangle(img, (x + x1, y + y1), (x + x1 + w, y + y1 + + h), (0, 255, 0), 2)
                    cv.putText(img, label + ": " + str(np.round(confidence,2)), (x + x1, y + y1 - 5),
                               cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            #cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.f1 = self.f2




        jpeg = cv.imencode('.jpg', img)[1]
        time.sleep(0.5)
        return jpeg.tobytes()

    def __del__(self):
        self.vs.release()




class mobilenet:

    def __init__(self, threshold = 0.5, pbPath = 'ssd_mobilenet_v1_coco_2017_11_17.pb', pbtxtPath = 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt'):
        # minimum confidence
        self.threshold = threshold
        # Load in the model
        self.model = cv.dnn.readNetFromTensorflow(pbPath,pbtxtPath)
        self.classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    def predict(self, img):
        # set scaling factors
        y_factor, x_factor = img.shape[:-1]
        #perform prediction
        start = time.time()
        blob = cv.dnn.blobFromImage(img, size = (300,300), swapRB=True, crop=False)
        self.model.setInput(blob)
        #model.setInput(cv.dnn.blobFromImage(image, swapRB=True))
        output = self.model.forward()[0,0]

        #keep only the predictions that meet the threshold
        thresholdIdx = np.where(output[:,2]>self.threshold)
        # Get the label text for the predictions that meet the threshold
        labels = [self.classNames.get(i) for i in output[thresholdIdx, 1][0]]
        confidences = np.round(output[thresholdIdx, 2], 2)[0]

        #Get the bounding boxes of the confident predictions and rescale them to fit original image
        output[thresholdIdx, 3:7] = output[thresholdIdx, 3:7].clip(min=0)
        boxes = (output[thresholdIdx, 3:7] * np.array([x_factor, y_factor, x_factor, y_factor])[0].astype(int))

        #Draw the bounding boxes onto the image
        if len(labels) > 0:
            # mostConfident = np.argmax(confidences)
            # print("most confident", confidences[mostConfident], "label is", labels[mostConfident])
            #
            #
            # print("largest box is", labels[largestBox])

            # Only return the largest identified object
            boxSize = (boxes[0, :, 2] - boxes[0, :, 0]) * (boxes[0, :, 3] - boxes[0, :, 1])
            largestBox = np.argmax(boxSize)
            # print(largestBox)
            # print(boxes.shape)
            box = boxes[0][largestBox].astype(int)
            label = labels[largestBox]
            confidence = confidences[largestBox]

            # for label, box, confidence in zip(labels, boxes[0], confidences):
            #     x, y, w, h = box.astype(int)
            #     #y-=80
            #     # h-=80
            #     cv.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            #     cv.putText(img, label + ": " + str(confidence), (x, y-5), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            print("Mobilenet v3 time taken:", time.time()-start)

            return (box, label, confidence)
        print("Mobilenet v3 time taken:", time.time() - start)
        return None


class yolo:

    def __init__(self, threshold=0.5, size = 320, configPath="yolov3.cfg", weightsPath="yolov3.weights"):
        # Load in the model
        self.model = cv.dnn.readNetFromDarknet(configPath, weightsPath)
        # Set minimum confidence threshold
        self.threshold = threshold
        self.size = (size,size)
        layerNames = self.model.getLayerNames()
        self.layerNames = [layerNames[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        self.labels = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane',
                       5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                       10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                       14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                       20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                       25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                       30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                       35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                       39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                       45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                       51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                       57: 'sofa', 58: 'pottedplant', 59: 'bed', 60: 'diningtable', 61: 'toilet',
                       62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                       67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                       72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                       78: 'hair drier', 79: 'toothbrush'}

    def predict(self, img):
        (y_factor, x_factor) = img.shape[:2]

        # perform prediction
        start = time.time()
        # blob = cv.dnn.blobFromImage(cv.resize(image_org, (300, 300)), size = (300, 300), swapRB=True)
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, self.size, swapRB=True, crop=False)
        self.model.setInput(blob)
        # model.setInput(cv.dnn.blobFromImage(image, swapRB=True))
        layerOutputs = self.model.forward(self.layerNames)

        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.threshold:
                    box = detection[0:4] * np.array([x_factor, y_factor, x_factor, y_factor])
                    (x, y, w, h) = box
                    x = int(x - (w / 2))
                    w += x
                    y = int(y - (h / 2))
                    h += y
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # non-maximal suppression
        idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.8, top_k=1)

        if len(idxs) > 0:
            i = idxs.flatten()[0]
            print("yolov3 time taken:", time.time() - start)
            return (np.array(boxes[i]), self.labels.get(classIDs[i]), confidences[i])
            # for i in idxs.flatten():
            #     x, y, w, h = boxes[i]
            #     cv.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            #     cv.putText(img, self.labels.get(classIDs[i]) + ": " + str(round(confidences[i], 2)), (x, y - 5),
            #                cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 52, 64), 1)
        print("yolov3 time taken:", time.time() - start)

        return None
#detector = moveDetect(model='mobilenet')

# while(1):
#     detector.getFrame()
