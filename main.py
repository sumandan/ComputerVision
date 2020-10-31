import cv2

#colors for boxes and text
blue = (255,0,0)
green = (0,255,0)
yellow = (0,0,255)



# Loading the cascade
Person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
Face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalFace_default.xml")
Upper_Body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
Lower_Body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lowerbody.xml")



# To capture video from webcam.
cap = cv2.VideoCapture(0)

# If a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the Faces and boundaries and variables
    Person = Person_cascade.detectMultiScale(gray, 1.1, 4)
    Face = Face_cascade.detectMultiScale(gray, 1.1, 4)
    Upper_Body = Upper_Body_cascade.detectMultiScale(gray, 1.1, 4)
    Lower_Body = Lower_Body_cascade.detectMultiScale(gray,1.1,4)


    # Draw the rectangle around each Object

    #Person
    for (x, y, w, h) in Person:
        cv2.rectangle(img, (x, y), (x+w, y+h), (green), 1)
        cv2.putText(img,"Person",(x,y-5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,1,green,1)

    #Face
    for (x, y, w, h) in Face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (blue), 1)
        cv2.putText(img, "Face", (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, blue,0)

    #Upper Body
    for (x, y, w, h) in Upper_Body:
        cv2.rectangle(img, (x, y), (x + w, y + h), (yellow), 1)
        cv2.putText(img, "Upper_Body", (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, yellow,0)

    #lower Body
    for (x, y, w, h) in Lower_Body:
        cv2.rectangle(img, (x, y), (x + w, y + h), (yellow), 1)
        cv2.putText(img, "Lower_Body", (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, yellow,0)





    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
