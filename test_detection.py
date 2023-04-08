import cv2

image = cv2.imread('/home/pratik/jupyternotebook_programs/object_tracking/Tracking/Tracking/Images/people.jpg')

detector = cv2.CascadeClassifier('/home/pratik/jupyternotebook_programs/object_tracking/Tracking/Tracking/cascade/fullbody.xml')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('People', image_gray)

detections = detector.detectMultiScale(image)
print(detections)

for (x,y,w,h) in detections:
    cv2.rectangle(image, (x,y), (x+w,y+h),(0,255,0), 2)

cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()