import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from convolution import get_convolution 


house_image = Image.open('data\\Original photo.jpg')
house_image_np3d = np.array(house_image)
print(np.shape(house_image_np3d))


#Define vertical and horizontal filters

get_3d_kernel = lambda kernel2d: np.array([kernel2d for _ in range(3)])


#Filters for highlighting horizontal and vertical lines
vertical_sobel_filter2d = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float64)
vertical_sobel_filter3d = get_3d_kernel(vertical_sobel_filter2d)
horizontal_sobel_filter2d = vertical_sobel_filter2d.transpose()
horizontal_sobel_filter3d = get_3d_kernel(horizontal_sobel_filter2d)


#conv_img = get_convolution(house_image_np3d, horizontal_sobel_filter3d, (1, 1))
#plt.imshow(conv_img)
#plt.show()
#print(np.shape(conv_img))


#Contrast enhancement filter
contrast_enhancement = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
contrast_filter3d = get_3d_kernel(contrast_enhancement)
conv_img2 = get_convolution(house_image_np3d, contrast_filter3d)
plt.imshow(conv_img2)
plt.show()



filter3d = get_3d_kernel(np.ones((3, 3)) / 9) 
filter_conv = get_convolution(house_image_np3d, filter3d)
plt.imshow(filter_conv)
plt.show()


#Sharpness filter
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kernel_conv = get_convolution(house_image_np3d, get_3d_kernel(kernel))
plt.imshow(kernel_conv)
plt.show()

'''
gauss_blur3d = get_3d_kernel(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16)
gauss_blur_conv = get_convolution(house_image_np3d, gauss_blur3d)
plt.imshow(gauss_blur_conv)
plt.show()
'''
#Embossing filter
'''
emb_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
emb_kernel_conv = get_convolution(house_image_np3d, get_3d_kernel(emb_kernel))
plt.imshow(kernel_conv)
plt.show()
'''