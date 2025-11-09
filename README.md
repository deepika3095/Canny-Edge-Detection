# Canny-Edge-Detection
<H3>Name: DEEPIKA R</H3>
<H3>Register no: 212223220038</H3>
<H3>Date: 16.10.2025 </H3>

# AIM:
To implement the Canny edge detection algorithm on a sample image to identify and extract the edges, and to study the effect of different parameter settings (thresholds and smoothing) on the detected edges.

# PROGRAM:
```import cv2
import matplotlib.pyplot as plt
image = cv2.imread('download.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image path
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
edges_1 = cv2.Canny(blurred, 30, 100)   # Low thresholds: more edges, noisy
edges_2 = cv2.Canny(blurred, 50, 150)   # Balanced thresholds
edges_3 = cv2.Canny(blurred, 100, 200)  # High thresholds: only strong edges
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.title('Canny Edges (30,100)')
plt.imshow(edges_1, cmap='gray')
plt.axis('off')
plt.subplot(2,2,3)
plt.title('Canny Edges (50,150)')
plt.imshow(edges_2, cmap='gray')
plt.axis('off')
plt.subplot(2,2,4)
plt.title('Canny Edges (100,200)')
plt.imshow(edges_3, cmap='gray')
plt.axis('off')
plt.show()cv2.destroyAllWindows()

```

# OUTPUT:
## ORIGINAL IMAGE
<img width="347" height="331" alt="image" src="https://github.com/user-attachments/assets/8c96340d-a9d4-416c-bfa1-55b1205c588e" />

