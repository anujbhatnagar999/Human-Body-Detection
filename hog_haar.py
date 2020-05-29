#####FULL BODY---- GREEN
#####UPPER BODY--- BLUE
#####LOWER BODY--- RED
#####FACE     ---- YELLOW


# import the necessary packages
#'numpy' automatically installed with openCV and
#it is lib for numerical operations and array stuctures
import numpy as np
import cv2


pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profileface_cascade = cv2.CascadeClassifier('profileface.xml')
upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
lower_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')


 
# initialize the HOG descriptor/person detector

#Creates the HOG descriptor and detector with default parameters.
hog = cv2.HOGDescriptor()
#Setting SVM detector to make a decision as per HOG feature extraction for peoples.
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(1)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (300, 300))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)



######################USING HAAR CLASSIFIER FOR "FULL BODY" WITH HOG+SVM FOR MORE ACCURACY #######
    
    human = human_cascade.detectMultiScale(frame ,1.1 , 4  )
    human = np.array([[x, y, x + w, y + h] for (x, y, w, h) in human])

    for (xA, yA, xB, yB) in human:
        # display the detected boxes(green (BGR)) in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)


##################################################################################################


######################USING HAAR CLASSIFIER FOR "PEDESTRIAN"############################## #######
    
    pedestrian = pedestrian_cascade.detectMultiScale(frame ,1.1 , 4  )
    pedestrian = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrian])

    for (xA, yA, xB, yB) in pedestrian:
        # display the detected boxes(green (BGR)) in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)


##################################################################################################
        
        
######################USING HAAR CLASSIFIER FOR "FACE"############################################
    
    face = face_cascade.detectMultiScale(frame ,1.1 , 4  ) 
    face = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face])

    for (xA, yA, xB, yB) in face:
        # display the detected boxes(Yellow (BGR)) in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 255), 2)
#***profileface
        
    profileface = profileface_cascade.detectMultiScale(frame ,1.1 , 4  ) 
    profileface = np.array([[x, y, x + w, y + h] for (x, y, w, h) in profileface])

    for (xA, yA, xB, yB) in profileface:
        # display the detected boxes(Yellow (BGR)) in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 255), 2)    


##################################################################################################

######################USING HAAR CLASSIFIER FOR "UPPER BODY"######################################
    
    upper = upper_cascade.detectMultiScale(frame ,1.1 , 4  ) 
    upper = np.array([[x, y, x + w, y + h] for (x, y, w, h) in upper])

    for (xA, yA, xB, yB) in upper:
        # display the detected boxes(Blue (BGR)) in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (255, 0, 0), 2)


##################################################################################################

######################USING HAAR CLASSIFIER FOR "LOWER BODY"######################################
    
    lower = lower_cascade.detectMultiScale(frame ,1.1 , 4  ) 
    lower = np.array([[x, y, x + w, y + h] for (x, y, w, h) in lower])

    for (xA, yA, xB, yB) in lower:
        # display the detected boxes(Red (BGR)) in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 0, 255), 2)


##################################################################################################


#**************************************HOG feature extraction +SVM *******************************
        
    # detect people in the image
    # returns the bounding boxes for all the detected objects in the frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes(green (BGR)) in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
#*************************************************************************************************
        
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
