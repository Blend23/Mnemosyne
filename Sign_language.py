import cv2

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math 
import time


cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
classifier= Classifier("C:\saanvi_code\SignLanguage\Model\keras_model.h5","C:\saanvi_code\SignLanguage\Model\labels.txt")
offset=20
imgSize = 300

folder = r"C:\saanvi_code\SignLanguage\Data\O"
counter = 0
lables=[ 'A'
,'B'
, 'C'
, 'D'
, 'E'
, 'F'
, 'G'
, 'H'
,'I'
, 'J'
, 'K'
, 'L'
, 'M'
, 'N'
, 'O'
, 'P'
, 'Q'
, 'R'
, 'S'
, 'T'
, 'U'
, 'V'
, 'W'
, 'X'
, 'Y'
, 'Z'
]
#imgWhite= cv2.namedWindow("Image White", cv2.WINDOW_NORMAL)
imgWhite= cv2.namedWindow("Image White", cv2.WINDOW_NORMAL)
while True:
    success, img = cap.read()
    imgOutput=img.copy()
    hands, img = detector.findHands(img)
    if hands:

        hand=hands[0]
        x, y, w, h = hand["bbox"]
        imgCrop = img[y-20:y+h+20, x-20:x+w+20]
                
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
       
        imgCropShape=imgCrop.shape

    

        aspectRatio = h/w



        if aspectRatio>1:
                k=imgSize / h
                wCal=math.ceil(k * w)
                imgResize= cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap=math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
           #     prediction,index=classifier.getPrediction(imgWhite,draw=False)
           #     print(prediction,index)
        else :
                k=imgSize/w
                hCal=math.ceil(k * h)
                imgResize= cv2.resize(imgCrop,(imgSize, hCal))
                imgResizeShape=imgResize.shape
                hGap=math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
     #           prediction,index=classifier.getPrediction(imgWhite,draw=False)


        cv2.rectangle(imgOutput,(x-20,y-20-50),(x-20+90,y-20-50+50),(255,0,255),cv2.FILLED)   
        #cv2.putText(imgOutput,lables[index],(x,y-26),cv2.FONT_ITALIC,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-20,y-20),(x+w+20,y+h+20),(255,0,255),4)    
                


            # cv2.imshow("Hand2",imgCrop2)


        cv2.imshow("Image Crop",imgCrop)
        cv2.imshow("Image White", imgWhite)
    cv2.imshow("Image", imgOutput)
    try:
        if len(hands)==0:
            cv2.destroyWindow("Image Crop")
            cv2.destroyWindow("Image White")

    except:
        pass

    if cv2.waitKey(1)==ord("s"):
        counter += 1 
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print (counter)

    if (-1 != cv2.waitKey(1) and cv2.waitKey(1)==27):
        cv2.destroyAllWindows()
