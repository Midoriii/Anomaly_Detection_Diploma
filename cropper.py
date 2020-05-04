import glob 
import numpy as np
import cv2


# Grab all the .jpg images
images = glob.glob('Clonky-ok/*')
#images = glob.glob('Clonky-vadne/*.jpg')

# To save the images and later convert into numpy array
images_list = []

# Crop and save each one of them
for image in images:
   img = cv2.imread(image)
   
   # Resize into 768*768 if bigger
   resized = cv2.resize(img, (768, 840)) 
   
   # Defect images are of size 768*768
   cropped_img = resized[0:768, 0:768]
   #cv2.imshow("cropped", cropped_img)
   #print(cropped_img.shape)
   
  
   
   # Make it actual grayscale 
   gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
   #cv2.imshow("gray", gray)
   #print(gray.shape)
   
   #cv2.waitKey(0)
   
   # Add the new grayscale image to the list
   images_list.append(gray)
   
   # Save the cropped grayscale image
   cv2.imwrite('Cropped\\' + image, gray)


numpy_images = np.array(images_list)
#print(numpy_images.shape)

# Finally save the numpy representation
np.save('Data\\OK.npy', numpy_images)

# Just a test
#loaded = np.load('Data\\Vadne.npy')
#print(loaded.shape)
#cv2.imshow("test", loaded[0])
#cv2.waitKey(0)