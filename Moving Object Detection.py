import cv2
import imutils
import time

print(" --------------   Press x to Quit  ----------------")
#  Capturing the image :
cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 420 # less the number more will be overfitted. 
while True:
    _, img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width = 600)
    gray_Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gaussian_Image = cv2.GaussianBlur(gray_Img, (21, 21), 0)
    if firstFrame is None:
        firstFrame = Gaussian_Image
        continue
    imgDiff = cv2.absdiff(firstFrame, gray_Img)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    thershImg = cv2.dilate(threshImg, None, iterations = 2)    # to remove hole from image
    contours = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # to find contours( broken edge alike will be connected) from the image.
    cnts = imutils.grab_contours(contours)   # extract the areas from contours  .. If area is less it can be skipped (below lines)
    for z in cnts:
        if cv2.contourArea(z) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(z)   # gives the exact height, width value of a object that is present apart from the background. 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)   # draw rectangel around object.
        text = "Moving Object Detected"
    #print(text)    
    cv2.putText(img, text, (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 60, 220), 2)
    cv2.imshow("Object_Detection", img)
    w_key = cv2.waitKey(1) & 0xFF   # waitKey(1) is for 32 bit and 0xFF returns 8 bit binary
    if w_key == ord('x'):   # if the binary of x matches with w_key then terminate the process.
        break
cam.release()
cv2.destroyAllWindows()
    
