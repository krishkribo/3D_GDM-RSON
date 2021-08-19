#!/usr/bin/env python3

import sys
import numpy as np
from sensor_msgs.msg import Image

class ImageConverter:
	def __init__(self):
		self.name_to_dtypes = {
			"rgb8":    (np.uint8,  3),
			"rgba8":   (np.uint8,  4),
			"rgb16":   (np.uint16, 3),
			"rgba16":  (np.uint16, 4),
			"bgr8":    (np.uint8,  3),
			"bgra8":   (np.uint8,  4),
			"bgr16":   (np.uint16, 3),
			"bgra16":  (np.uint16, 4),
			"mono8":   (np.uint8,  1),
			"mono16":  (np.uint16, 1),

		    # OpenCV CvMat types
			"8UC1":    (np.uint8,   1),
			"8UC2":    (np.uint8,   2),
			"8UC3":    (np.uint8,   3),
			"8UC4":    (np.uint8,   4),
			"8SC1":    (np.int8,    1),
			"8SC2":    (np.int8,    2),
			"8SC3":    (np.int8,    3),
			"8SC4":    (np.int8,    4),
			"16UC1":   (np.int16,   1),
			"16UC2":   (np.int16,   2),
			"16UC3":   (np.int16,   3),
			"16UC4":   (np.int16,   4),
			"16SC1":   (np.uint16,  1),
			"16SC2":   (np.uint16,  2),
			"16SC3":   (np.uint16,  3),
			"16SC4":   (np.uint16,  4),
			"32SC1":   (np.int32,   1),
			"32SC2":   (np.int32,   2),
			"32SC3":   (np.int32,   3),
			"32SC4":   (np.int32,   4),
			"32FC1":   (np.float32, 1),
			"32FC2":   (np.float32, 2),
			"32FC3":   (np.float32, 3),
			"32FC4":   (np.float32, 4),
			"64FC1":   (np.float64, 1),
			"64FC2":   (np.float64, 2),
			"64FC3":   (np.float64, 3),
			"64FC4":   (np.float64, 4)
		}

	def convert_to_opencv(self, msg):
		if not msg.encoding in self.name_to_dtypes:
			raise TypeError('Unrecognized encoding {}'.format(msg.encoding))

		dtype_class, channels = self.name_to_dtypes[msg.encoding]
		dtype = np.dtype(dtype_class)
		dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
		shape = (msg.height, msg.width, channels)
		data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
		data.strides = (msg.step,
						dtype.itemsize * channels,
						dtype.itemsize
						)
		if ("rgb" in msg.encoding):
			data[..., 0], data[..., 2] = data[..., 2], data[..., 0].copy() #convert from rgb to bgr

		if channels == 1:
			data = data[...,0] #convert to 2-dimensional array

		return data

	def convert_to_ros(self, arr, encoding):
		if not encoding in self.name_to_dtypes:
			raise TypeError('Unrecognized encoding {}'.format(encoding))
		
		if ("rgb" in encoding):
			arr[..., 0], arr[..., 2] = arr[..., 2], arr[..., 0].copy() #convert from bgr to bgr


		im = Image(encoding=encoding)

		dtype_class, exp_channels = self.name_to_dtypes[encoding]
		dtype = np.dtype(dtype_class)
		if len(arr.shape) == 2:
			im.height, im.width, channels = arr.shape + (1,)
		elif len(arr.shape) == 3:
			im.height, im.width, channels = arr.shape
		else:
			raise TypeError("Array must be 2 or 3 dimensional")

		if exp_channels != channels:
			raise TypeError("Array has {} channels, {} requires {}".format(channels, encoding, exp_channels))
		if dtype_class != arr.dtype.type:
			raise TypeError("Array is {}, {} requires {}".format(arr.dtype.type, encoding, dtype_class))

		contig = np.ascontiguousarray(arr)
		im.data = contig.tostring()
		im.step = contig.strides[0]
		im.is_bigendian = (
			arr.dtype.byteorder == '>' or 
			arr.dtype.byteorder == '=' and sys.byteorder == 'big')

		return im



