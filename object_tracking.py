import cv2
import copy
from detectors import Detectors
from tracker import Tracker
import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout
import matplotlib.pyplot as plt
import sys
from keras.models import load_model
model=load_model('my_model.h5')
img_width, img_height = 32, 32
def main():
    # Create opencv video capture object
    cap = cv2.VideoCapture('/home/deepak/innovation_lab_files/vid1_new.mp4')

    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker
    tracker = Tracker(160, 30, 5, 100)

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False
    # Infinite loop to process video frames
    mainlist=[[None,None]]*1000
    CarCount=0
    NoneCarCount=0
    NoneVehicle=0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Make copy of original frame
        orig_frame = copy.copy(frame)

        # Skip initial frames that display logo
        if (skip_frame_count < 15):
            skip_frame_count += 1
            continue

        # Detect and return centeroids of the objects in the frame
        centers = detector.Detect(frame)
        newcenter=[]
        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)
            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            # print(len(tracker.tracks))
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 4):
                    # print(tracker.tracks[i].trace)
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)
                    cv2.putText(frame,str(tracker.tracks[i].track_id), (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (0, 0, 255), 2)
                    newcenter.append([int(x1),int(y1),tracker.tracks[i].track_id])

            # Display the resulting tracking frame
            # cv2.line(frame,(0,0),(100,100),(22,122,222),8)
            cv2.line(frame,(200,600),(960,600),(139,0,0),8)
            cv2.putText(frame,'Car Count =' + str(CarCount),(30,30),cv2.FONT_HERSHEY_SIMPLEX, 0.75,(92, 142, 215),3)
            cv2.putText(frame,'Non Car Count =' + str(NoneCarCount), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (92, 142, 215),
                        3)
            print(CarCount+NoneCarCount)
            # cv2.line(frame,(150,450),(280,370),(139,0,0),8)
            cv2.imshow('Tracking', frame)
        for j in range(len(centers)):
            for i in range(len(newcenter)):
                a=newcenter[i][0]
                b=newcenter[i][1]
                e=newcenter[i][2]
                c=centers[j][0][0]
                d=centers[j][1][0]
                temp_len=np.sqrt((a-c)*(a-c)+(b-d)*(b-d))

                if(temp_len<7):
                    if(mainlist[e][0]!=None):
                        c=mainlist[e][0]
                        d=mainlist[e][1]
                        if((d<=600) and (c>=200) and (c<=960) and (a>=200) and (a<=960) and b>=600):
                            CarCount+=1
                            s1=orig_frame.shape[0]
                            s2=orig_frame.shape[1]
                            # print('this')
                            # print(s1)
                            # print(s2)
                            # print(a)
                            # print(b)
                            # print('this')
                            # if((a-120>=0) and (a+120<=s2) and (b-120>=0) and (b+120<=s1)):
                            try:
                                img=orig_frame[a-80:a+80,b-80:b+80]
                                # cv2.imshow("cropped", img)
                                img = cv2.resize(img, (img_width, img_height))
                                arr = np.array(img).reshape((3, img_width, img_height))
                                arr = np.expand_dims(arr, axis=0)
                                prediction = model.predict(arr)[0]
                                # print(prediction)
                                bestclass = ''
                                bestconf = -1
                                best = ['non-vehicle', 'vehicle', 'non-vehicle', 'non-vehicle', 'non-vehicle',
                                        'non-vehicle', 'non-vehicle', 'non-vehicle', 'non-vehicle', 'vehicle']
                                for n in [0, 1, 2]:
                                    if (prediction[n] > bestconf):
                                        bestclass = n
                                        bestconf = prediction[n]
                                if(bestclass!=1 and bestclass!=9):
                                    NoneVehicle+=1
                                    if(NoneVehicle%10==2):
                                        CarCount-=1
                                        NoneCarCount+=1

                            # else :
                            except:
                                print('this is already vehicle')

                        mainlist[e][0]=a
                        mainlist[e][1]=b
                    else:
                        mainlist[e][0]=a
                        mainlist[e][1]=b
                    newcenter.pop(i)
                    break
        # for i in range(len(newcenter)):
        #     mainlist[newcenter[i][2]][0]=newcenter[i][0]
        #     mainlist[newcenter[i][2]][1]=newcenter[i][1]
        # Display the original frame
        # cv2.imshow('Original', orig_frame)

        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

    # When everything done, release the capture
    print("this is final car count ")
    print(CarCount)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()
