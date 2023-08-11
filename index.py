from matplotlib import pyplot as plt
import cv2 as cv
# import pytesseract 
from PIL import Image
import pytesseract

version = pytesseract.pytesseract.get_tesseract_version()
if version is None:
    print("Tesseract is not installed or not accessible.")
else:
    print("Tesseract version:", version)

# pytesseract.pytesseract.tesseract_cmd = r'C:\path\to\tesseract.exe'
# image='npm.PNG'
image='page_01.jpg'
# image='Capture.PNG'

img=cv.imread(image)
# cv.imshow("cat",img)
cv.waitKey(0)



# stackover flow
def display_image_in_actual_size(im_path):

    dpi = 80
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

# display_image_in_actual_size(image)


# img_inverted=cv.bitwise_not(img)
# cv.imwrite("images/invertedimage.jpg",img_inverted)
# display_image_in_actual_size("images/invertedimage.jpg") 



thresh, im_bw=cv.threshold(img,200,230,cv.THRESH_BINARY)
cv.imwrite("images/bitwise.jpg",im_bw)
# display_image_in_actual_size("images/bitwise.jpg")

def noise_removal(image):
    import numpy as np
    kernel=np.ones((1,1),np.uint8)
    image=cv.dilate(image,kernel,iterations=1)
    kernel=np.ones((1,1),np.uint8)
    image=cv.erode(image,kernel,iterations=1)
    image=cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)
    image=cv.medianBlur(image,3)
    return(image)

no_noise=noise_removal(im_bw)
cv.imwrite("images/nonoise.jpg",no_noise)
# display_image_in_actual_size("images/nonoise.jpg")

def thin_font(image):
    import numpy as np
    image=cv.bitwise_not(img)
    kernel=np.ones((2,2),np.uint8)
    image=cv.erode(image,kernel,iterations=2)
    image=cv.bitwise_not(img)
    return image


eroded_image=thin_font(no_noise)
cv.imwrite("images/erodedimage.jpg",eroded_image)
# display_image_in_actual_size("images/erodedimage.jpg")

img1="images/bitwise.jpg"
final_image=Image.open(img1)
ocr=pytesseract.image_to_string(final_image)
with open("images/nonoise1.txt", "w", encoding="utf-8") as file:
    file.write(ocr)
print(ocr)