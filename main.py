import cv2

#colors for boxes and text
blue = (255,0,0)
green = (0,255,0)

# Loading the cascade
Person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
Face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalFace_default.xml")
Car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "cars.xml")


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
    Car = Car_cascade.detectMultiScale(gray, 1.1, 4)


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

    #Car
    for (x, y, w, h) in Car:
        cv2.rectangle(img, (x, y), (x + w, y + h), (blue), 1)
        cv2.putText(img, "Lisence", (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, blue,0)



    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
