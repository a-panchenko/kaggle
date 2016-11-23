import cv2
import numpy as np
import os
import random


def __resize_and_crop(img, size):
    shape = np.shape(img)
    if len(shape) == 3:
        h = shape[0]
        w = shape[1]
        if h < w:
            scale = h / size
            shape = (size, int(w / scale))
        else:
            scale = w / size
            shape = (int(h / scale), size)
        img = cv2.resize(img, shape)
        h = shape[0]
        w = shape[1]
        if h < w:
            delta = (w - size) / 2
            img = img[delta:(w - delta), 0:size]
        else:
            delta = (h - size) / 2
            img = img[0:size, delta:(h - delta)]
        return img
    else:
        print("Wrong shape!")
        return None


def __randomly_rotate(img):
    degrees = [90, 180, 270]
    size = img.shape[0]
    rotation_matrix = cv2.getRotationMatrix2D((size / 2, size / 2), degrees[random.randint(0, 2)], 1.0)
    return cv2.warpAffine(img, rotation_matrix, (size, size))


def preprocess_image(root_folder, outputpath, size, distort):
    """
    Converts images from 'root_folder' into a byte array where first byte is 
    a label 0 for cats and 1 for dogs. Afterwards, writes converted images into
    a file with 'outputpath'.
    """
    with open(outputpath, 'w') as output_file:
        output_file.write('')
    counter = 0
    with open(outputpath, 'br+') as output_file:
        for root, dirs, filenames in os.walk(root_folder):
            print(len(filenames))
            for fname in filenames:
                if fname.endswith('.jpg'):
                    img_path = os.path.join(root, fname)
                    img = cv2.imread(img_path)
                    resized = __resize_and_crop(img, size)
                    label = 0 if fname.split('.')[0] == 'cat' else 1
                    images = [resized]
                    if distort:
                        images.append(__randomly_rotate(resized))
                        if random.randint(0, 1) == 0:
                            images.append(cv2.flip(resized, 1))
                    for indx, i in enumerate(images):
                        output_file.write(bytearray([label]) + bytearray(np.array(i).flatten()))
                        counter += 1
                        if counter % 1000 == 0:
                            print(counter)
    print("Total images: " + str(counter))

