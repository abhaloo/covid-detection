import cv2

#Function to display individual images that are being classified by the model
#It is used for debugging purposes.

def display(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
	