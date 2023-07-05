import cv2 as cv

img = cv.imread('assets/maria.jpg', 1)

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()