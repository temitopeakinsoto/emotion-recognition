import cv2 as cv

# Load the input images
img = cv.imread('assets/faces.jpg', 1)
#img2 = cv.imread('assets/maria.jpg')
#print(f'img2 is {img2}')
faces_img = cv.imread('assets/faces.jpg', 1)


# Check if the images were loaded successfully
if img is None or faces_img is None:
    print('There was an error loading the images')
    exit()

# Convert the faces image to grayscale
gray_img = cv.cvtColor(faces_img, cv.COLOR_BGR2GRAY)

# Load the Haar cascade classifier for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Detect faces in the grayscale image
face_rects = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7)

print(f'Number of faces = {len(face_rects)}')

# Draw rectangles around the detected faces in the original image
for (x, y, w, h) in face_rects:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2
                 )

# Display the image with face rectangles
cv.imshow('Person', img)

# Wait for a key press
cv.waitKey(0)

# Close all windows
cv.destroyAllWindows()
