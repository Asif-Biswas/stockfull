""" Classical Computer Vision Inference Script for Shelf Change Detection

This script will evaluate the input (local_video/video_url) and run the inference based on it.

Input Options:
--------------
  * `input_video_path`     : input - path to {video/} or Valid URL
  * `output_video_path`    : output - path to save the output

Output Options:
---------------
    - Video

Notes:
------
    None

Example Usage:
--------------
python3 main.py input.mp4 output.mp4

"""
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import sys
import os

import cv2
def draw_contours(contours, second):
    for c in contours:
        area = cv2.contourArea(c)
        if area > 20000:
            return
        if area > 1000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.putText(second,str(area),(x,y+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            cv2.rectangle(second, (x, y), (x + w, y + h), (36,255,12), 2)

def remove_glare(gray_image):
    mask = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)[1]
    res = cv2.inpaint(gray_image,mask,21,cv2.INPAINT_TELEA)
    return res

# get base-frame
def get_base_frame(vid):
    if vid.isOpened():
        ret,frame = vid.read()
        # cv2.imshow('get_base_frame',frame)
        base_frame = cv2.GaussianBlur(frame,(5,5),0)
        
        # if cv2.waitKey() & 0xFF == ord("q"):
        #     base_frame = cv2.GaussianBlur(frame,(5,5),0)
        #     # first = cv2.rotate(first,cv2.ROTATE_180)
        #     print('Base frame selected...!')
        #     cv2.destroyWindow('get_base_frame')
        #     return base_frame
        return base_frame
    else:
        print("video not opened")



def run_inference(base_frame, out, vid):
    frm_no = 0
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    frame_size = (frame_width,frame_height)
    fps = round(vid.get(cv2.CAP_PROP_FPS)) #sys.argv[2]
    vid_output = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'XVID'), int(fps), frame_size)

    while vid.isOpened():
        ret,frame_to_compare = vid.read()
        # frame_to_compare = cv2.rotate(frame_to_compare,cv2.ROTATE_180)
        if not ret:
            cv2.destroyAllWindows()
            break
        else:
            frame_to_compare_ = cv2.GaussianBlur(frame_to_compare,(5,5),0)
            if frm_no%(fps*3) != 0:
                draw_contours(contours, frame_to_compare)
                # base_frame = cv2.rotate(base_frame ,cv2.ROTATE_180)

                vid_output.write(frame_to_compare)
                
                cv2.namedWindow('frame_to_compare',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame_to_compare',700,600)
                cv2.imshow('frame_to_compare', frame_to_compare)

                ## cv2.namedWindow('base_frame',cv2.WINDOW_NORMAL)
                ## cv2.resizeWindow('base_frame',700,600)
                ## cv2.imshow('base_frame', base_frame)
                
                cv2.waitKey(1)
                

            else:
                # Convert images to grayscale
                base_frame_gray = cv2.cvtColor(base_frame.copy(), cv2.COLOR_BGR2GRAY)
                frame_to_compare_gray = cv2.cvtColor(frame_to_compare_, cv2.COLOR_BGR2GRAY)
                
                # base_frame_gray = remove_glare(base_frame_gray)
                # frame_to_compare_gray = remove_glare(frame_to_compare_gray)

                # Compute SSIM between two images
                score, diff = structural_similarity(base_frame_gray, frame_to_compare_gray, full=True)
                
                ## print("Similarity Score: {:.3f}%".format(score * 100))
                
                # The diff image contains the actual image differences between the two images
                # and is represented as a floating point data type so we must convert the array
                # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
                diff = (diff * 255).astype("uint8")
                
                # Threshold the difference image, followed by finding contours to
                # obtain the regions that differ between the two images
                # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                thresh = cv2.threshold(diff, 27, 255, cv2.THRESH_BINARY_INV )[1]
                kernel = np.ones((3,3),np.uint8)
                
                # eroded = cv2.erode(thresh, kernel, iterations=1)
                thresh = cv2.dilate(thresh, kernel, iterations=3)
                # de = cv2.erode(dilated, kernel, iterations=1)
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # exit()
                
                contours = contours[0] if len(contours) == 2 else contours[1]
                print('contours: ',len(contours))
                contours = sorted(contours,key=cv2.contourArea,reverse=True)
                if contours:
                    if cv2.contourArea(contours[0]) < 20000:
                        ###########################
                        base_frame = frame_to_compare.copy() #updating basic template
                        print('base template updated!!!')
                        ###########################
                # Highlight differences
                mask = np.zeros(base_frame.shape, dtype='uint8')
                filled = frame_to_compare.copy()
                draw_contours(contours, frame_to_compare)
                ##cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)
                ##cv2.resizeWindow('thresh',700,600)
                ##cv2.imshow('thresh',thresh)
                
                cv2.namedWindow('frame_to_compare',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame_to_compare',700,600)
                ##cv2.namedWindow('base_frame',cv2.WINDOW_NORMAL)
                ##cv2.resizeWindow('base_frame',700,600)
                
                cv2.imshow('frame_to_compare', frame_to_compare)
                ## cv2.imshow('base_frame', base_frame)
                vid_output.write(frame_to_compare)
                cv2.waitKey(1)
            frm_no +=1
    vid.release()
    vid_output.release()

if __name__ == '__main__':
    file = sys.argv[1]
    print(sys.argv)
    vid = cv2.VideoCapture(file)
    base_frame = get_base_frame(vid)
    run_inference(base_frame)

def run(inp, out):
    vid = cv2.VideoCapture(inp)
    base_frame = get_base_frame(vid)
    run_inference(base_frame, out, vid)