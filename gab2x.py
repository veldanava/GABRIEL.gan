import tensorflow as tf
import tensorflow_hub as hub
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

# based on esrgan

print("""\
  \n
 ██████╗  █████╗ ██████╗ ██████╗ ██╗███████╗██╗         ██████╗  █████╗ ███╗   ██╗
██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██║        ██╔════╝ ██╔══██╗████╗  ██║
██║  ███╗███████║██████╔╝██████╔╝██║█████╗  ██║        ██║  ███╗███████║██╔██╗ ██║
██║   ██║██╔══██║██╔══██╗██╔══██╗██║██╔══╝  ██║        ██║   ██║██╔══██║██║╚██╗██║
╚██████╔╝██║  ██║██████╔╝██║  ██║██║███████╗███████╗██╗╚██████╔╝██║  ██║██║ ╚████║
 ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝
 coded by Kiana
                                                  
                    """)

print("upscale ur image resolution \n ctrl + c to quit \n")

# get and read image to process
img = cv2.imread(input("filename: "))

image_plot = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.title(image_plot.shape)
plt.imshow(image_plot)
plt.show()

# preprocess image
def preprocessing(img):
	imageSize = (tf.convert_to_tensor(image_plot.shape[:-1]) // 4) * 4
	cropimg = tf.image.crop_to_bounding_box(
		img, 0, 0, imageSize[0], imageSize[1])
	prepro = tf.cast(cropimg, tf.float32)
	return tf.expand_dims(prepro, 0)

# add  esrgan
esrgn_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(esrgn_path)

# employ model
def srmodel(img):
	prepro = preprocessing(img) # Preprocess image
	newim = model(prepro) # Run model
	# returns original image
	return tf.squeeze(newim) / 255.0

# Plot the processed image
proimg = srmodel(image_plot)
plt.title(proimg.shape)
plt.imshow(proimg)
plt.show()
