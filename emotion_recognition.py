import cv2
from deepface import DeepFace


# Use 0 for the default webcam
video_capture = cv2.VideoCapture(0)  

# Use the open-source haar_cascade classifier. 
# TOD0: use an alternative classifier 

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

while True:

    # Read a frame from the webcam
    ret, frame = video_capture.read()  
    result = DeepFace.analyze(img_path = frame, actions=['emotion'], enforce_detection=False)

    # convert video steam to gray
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_video, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    emotion = result[0]['dominant_emotion']
    txt = str(emotion)

    cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255), 3)
    
    # Display the frame with emotion
    cv2.imshow('Frame', frame)  
    
    if cv2.waitKey(1) & 0xff == ord('q'):  # Exit if 'q' is pressed
        break

video_capture.release()
cv2.destroyAllWindows()


