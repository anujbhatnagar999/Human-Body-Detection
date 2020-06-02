#####FULL BODY---- GREEN
#####UPPER BODY--- BLUE
#####LOWER BODY--- RED
#####FACE     ---- YELLOW



import cv2

pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
profileface_cascade = cv2.CascadeClassifier('profileface.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
lower_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

# Read the input image
img = cv2.imread('flood4.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#####COLOR ORDER (B,G,R)
pedestrian = pedestrian_cascade.detectMultiScale(gray ,1.1 , 4  )
for (x, y , w ,h) in pedestrian:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255 , 0), 3)



human = human_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y , w ,h) in human:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255 , 0), 3)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y , w ,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255 , 255), 3)

#***profileface
        
profileface = profileface_cascade.detectMultiScale(gray ,1.1 , 4  )
for (x, y , w ,h) in profileface:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255 , 255), 3)

    

upper = upper_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y , w ,h) in upper:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)

lower = lower_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y , w ,h) in lower:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0 , 255), 3) 


    

# Display the output
cv2.imshow('img', img)
cv2.waitKey(0)
       



















