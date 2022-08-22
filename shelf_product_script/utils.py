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

