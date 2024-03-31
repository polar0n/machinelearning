from os import listdir
import cv2


FOLDER = 'chapter 10/experiment/digits'


def modify_img(file_name):
    img = cv2.imread(file_name)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    img = cv2.bitwise_not(img)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # img = cv2.bitwise_not(img)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    img = cv2.resize(img, (28, 28))
    return img


images = listdir(FOLDER)
for image in images:
    result = modify_img(f'{FOLDER}/{image}')
    print(result)
    cv2.imwrite(f'{FOLDER}_exp/{image}', result)
