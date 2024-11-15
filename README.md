# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step-1 :
Import numpy module as np and pandas as pd.
### Step-2 : 
Assign the values to variables in the program.
### Step-3 :
Get the values from the user appropriately.
### Step-4 :
Continue the program by implementing the codes of required topics.
### Step-5 :
Thus the program is executed in jupyter notebook.
<br>

## Program:
```python
Developed By:SAILESHKUMAR A
Register Number:212222230126
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('pic2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

# 1. Translation
rows, cols, _ = image.shape
M_translate = np.float32([[1, 0, 50], [0, 1, 100]])  # Translate by (50, 100) pixels
translated_image = cv2.warpAffine(image_rgb, M_translate, (cols, rows))

plt.imshow(translated_image)
plt.title("Translated Image")
plt.axis('off')
# 2. Scaling
scaled_image = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)  # Scale by 1.5x


plt.imshow(scaled_image)
plt.title("Scaled Image")
plt.axis('off')

# 3. Shearing
M_shear = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # Shear with factor 0.5
sheared_image = cv2.warpAffine(image_rgb, M_shear, (int(cols * 1.5), int(rows * 1.5)))


plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')

# 4. Reflection (Flip)
reflected_image = cv2.flip(image_rgb, 1)  # Horizontal reflection (flip along y-axis)


plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')

# 5. Rotation
M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Rotate by 45 degrees
rotated_image = cv2.warpAffine(image_rgb, M_rotate, (cols, rows))

\
plt.imshow(rotated_image)
plt.title("Rotated Image")
plt.axis('off')

# 6. Cropping
cropped_image = image_rgb[50:300, 100:400]  # Crop a portion of the image

plt.figure(figsize=(4, 4))
plt.imshow(cropped_image)
plt.title("Cropped Image")
plt.axis('off')
plt.show()

```
## Output:
### i)Image Translation
![Screenshot 2024-10-03 153701](https://github.com/user-attachments/assets/3479c236-ec38-4b8a-a10d-56bb19ed7f35)


### ii) Image Scaling
![Screenshot 2024-10-03 153719](https://github.com/user-attachments/assets/3e45d418-f1ee-412a-9420-bb77353f9e71)



### iii)Image shearing

![Screenshot 2024-10-03 153725](https://github.com/user-attachments/assets/bcd6f918-5394-41a1-81ce-21330360d4b6)

### iv)Image Reflection
![Screenshot 2024-10-03 153733](https://github.com/user-attachments/assets/a59916d6-2c0b-4229-97b2-6d1508e11a07)



### v)Image Rotation
![Screenshot 2024-10-03 153740](https://github.com/user-attachments/assets/c3ed9b17-178f-4901-8115-de9fa3eda1fe)




### vi)Image Cropping
![Screenshot 2024-10-03 153754](https://github.com/user-attachments/assets/99cb3aff-9beb-4ede-aeaf-b0f2a663caf7)






## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
