import numpy as np
import cv2
import time
from ofd1 import *

def main():

    cam = cv2.VideoCapture("data/atrium.avi")
    p = int(cam.get(3))
    l = int(cam.get(4))

    ret, prev = cam.read()
   
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    cur_glitch = prev.copy()

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('result4.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30 , (p,l))
    
    while True:
        ret, img = cam.read()
        #vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 50, 20, 5, 5, 10.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray,flow))

        if show_hsv:
            gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray1, 3 , 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     
            # loop over the contours
            for c in cnts:
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centroid = (cx,cy)
                area = cv2.contourArea(c)
                #print area

                areaTH = 500
                areaTH2 = 50000

                # if the contour is too small, ignore it
                (x, y, w, h) = cv2.boundingRect(c)
                a = x + w
                b = y + h

                if (area > areaTH) and (area < areaTH2) and (w < 400) and (h < 600) :
                    cv2.rectangle(img, (x, y), (a , b ), (0, 255, 0), 2)
                    cv2.circle(img,(cx,cy),5,(0,0,255), -1)
            
            cv2.imshow('th', thresh)
            cv2.imshow('odf',draw_hsv(flow))
            cv2.imshow('Image', img)
        
            out.write(draw_flow(gray,flow))
            
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # execute main
    main()