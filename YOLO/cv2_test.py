import cv2

img = cv2.imread('./pics/GOPR0435.JPG')

print('Original Dimensions : ',img.shape)
scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dsize=dim, interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized.shape)
cv2.imshow('img',resized)
cv2.waitKey()
cv2.destroyAllWindows()
