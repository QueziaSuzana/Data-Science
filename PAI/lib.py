import cv2

import numpy as np
import math as m

import matplotlib.pyplot as plt

import os

USER_GENERATED_FILE_PREFIX = "static/uploads/"
PREPROCESSING_GENERATED_FILE_PREFIX = "static/processed/"


def sum(l: np.ndarray) -> float:
    """
    Calculate the sum of a list of numbers.

    Args:
        l (np.ndarray): A list of numbers.

    Returns:
        float: The sum of the numbers in the list.
    """
    return l.sum()


def density(l: float, total: float) -> float:
    """
    Calculate the density of a number in a list.

    Args:
        l (float): The number to calculate the density.
        total (float): The total sum of the list.

    Returns:
        float: The density of the number in the list.
    """
    return l / total


def entropy(m: np.ndarray) -> float:
    """
    Calculate the entropy of a matrix.

    Args:
        m (np.ndarray): The matrix to calculate the entropy.
    
    Returns:
        float: The entropy of the matrix.
    """
    result = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] > 0:
                result += m[i][j] * (np.log2(m[i][j]))
    return abs(result)


def homogeneity(m) -> float:
    """
    Calculate the homogeneity of a matrix.

    Args:
        m (np.ndarray): The matrix to calculate the homogeneity.

    Returns:
        float: The homogeneity of the matrix.
    """
    result = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            result += m[i][j] / (1 + abs(i - j))
    return result


def contrast(m) -> float:
    """
    Calculate the contrast of a matrix.

    Args:
        m (np.ndarray): The matrix to calculate the contrast.

    Returns:
        float: The contrast of the matrix.
    """
    result = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            result += m[i][j] * (i - j) ** 2
    return result


def index(l, value) -> int:
    """
    Get the index of a value in a list.

    Args:
        l (list): The list to search the value.
        value (any): The value to search.
    
    Returns:
        int: The index of the value in the list.
    """
    for i in range(len(l)):
        if l[i] == value:
            return i
    return -1


def generate_histogram_of_grayscale_image(y, new_filepath, shades) -> None:
    """
    Generate a histogram of a grayscale image.

    Args:
        y (np.ndarray): The histogram.
        new_filepath (str): The path to save the histogram.
        shades (int): The number of shades of gray.

    Returns:
        None
    """
    plt.title(f"Histogram of Image with {shades} shades of gray", fontsize="xx-large")    
    plt.xlabel("Shade of gray", fontsize="xx-large")
    plt.ylabel("Density", fontsize="xx-large")

    plt.plot(y)

    plt.savefig(new_filepath)
    plt.clf()


def generate_histogram_of_hsv_image(y, new_filepath, shades) -> None:
    """
    Generate a histogram of a HSV image.
    Do not use this function. Use generate_imshow_of_hsv_image instead.

    Args:
        y (np.ndarray): The histogram.
        new_filepath (str): The path to save the histogram.
        shades (int): The number of shades of gray.
    
    Returns:
        None
    """
    
    plt.title(f"Histogram of HSV Image with {shades} shades", fontsize="xx-large")    
    plt.xlabel("Shades", fontsize="xx-large")
    plt.ylabel("Density", fontsize="xx-large")

    plt.plot(y[:,0], color="r", label="H")
    plt.plot(y[:,2], color="b", label="V")

    plt.legend()
    plt.savefig(new_filepath)
    plt.clf()


def generate_imshow_of_hsv_image(y, new_filepath, shades) -> None:
    """
    Generate an imshow of a HSV image.

    Args:
        y (np.ndarray): The image.
        new_filepath (str): The path to save the imshow.
        shades (list): The number of shades of gray for H and V.
    
    Returns:
        None
    """
    plt.title(f"Imshow of HSV Image", fontsize="xx-large")    
    plt.xlabel("V", fontsize="xx-large")
    plt.ylabel("H", fontsize="xx-large")

    plt.imshow(y, interpolation="nearest")

    plt.savefig(new_filepath)
    plt.clf()


def generate_grayscale_image_from(original_image, shades: int = 16) -> np.ndarray:
    """
    Calculate a grayscale image from a colored image.

    Args:
        original_image (np.ndarray): The colored image.
        shades (int): The number of shades of gray.

    Returns:
        np.ndarray: The grayscale image.
    """
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    result = 255 * np.floor(gray * shades + 0.5) / (shades - 1)
    return result.clip(0, 255).astype(np.uint8)


def generate_histogram_image_from(original_image, channels=[0], mask=None, bins=[16], ranges=[0, 16]) -> np.ndarray:
    """
    Calculate a histogram (ndarray) from an image.

    Args:
        original_image (np.ndarray): The image.
        channels (list): The channels to calculate the histogram.
        mask (np.ndarray | None): The mask to calculate the histogram.
        bins (list): The number of bins for each channel.
        ranges (list): The range of each channel.

    Returns:
        np.ndarray: The histogram.
    """
    return cv2.calcHist([original_image], channels, mask, bins, ranges)


def operation_gray(filename, image, shades) -> np.ndarray:
    """
    Generate and save a grayscale image from a colored image.

    Args:
        filename (str): The name of the file.
        image (np.ndarray): The colored image.
        shades (int): The number of shades of gray.

    Returns:
        np.ndarray: The grayscale image.
    """
    new_filepath = f'{PREPROCESSING_GENERATED_FILE_PREFIX}gray{shades}_{filename}'
    result = generate_grayscale_image_from(image, shades=shades)
    cv2.imwrite(new_filepath, result)
    return result


def operation_hist(filename, image, shades) -> np.ndarray:
    """
    Calculate and save a histogram of a grayscale image.

    Args:
        filename (str): The name of the file.
        image (np.ndarray): The image.
        shades (int): The number of shades of gray.
    
    Returns:
        np.ndarray: The histogram.
    """
    new_filepath = f'{PREPROCESSING_GENERATED_FILE_PREFIX}hist{shades}_{filename}'
    result = generate_histogram_image_from(image, bins=[shades], ranges=[0, shades])
    
    tmp = result[:,0]
    total = tmp.sum()

    for i in range(len(result)):
        result[i] = density(tmp[i], total)

    generate_histogram_of_grayscale_image(result, new_filepath, shades=shades)
    return result


def operation_hist_hsv(filename, image) -> np.ndarray:
    """
    Calculate and save a histogram (imshow) of a HSV image with 16 shades for H and 8 shades for V.

    Args:
        filename (str): The name of the file.
        image (np.ndarray): The image.
    
    Returns:
        np.ndarray: The histogram.
    """
    new_filepath = f'{PREPROCESSING_GENERATED_FILE_PREFIX}histHSV_{filename}'
    tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    result = generate_histogram_image_from(tmp_image, channels=[0, 2], bins=[16, 8], ranges=[0, 360, 0, 100])
    
    tmp_h = result[:,0]
    tmp_v = result[:,1]
    
    total_h = tmp_h.sum()
    total_v = tmp_v.sum()

    for i in range(len(result)):
        result[i][0] = density(tmp_h[i], total_h)
        result[i][1] = density(tmp_v[i], total_v)

    generate_imshow_of_hsv_image(result, new_filepath, shades=[16, 8])
    return result


def operation_coocurrence(filename, image) -> "list[np.ndarray]":
    """
    Calculate the coocurrence matrix of a grayscale image with 16 shades of gray.

    Args:
        filename (str): The name of the file.
        image (np.ndarray): The image.

    Returns:
        list: The coocurrence matrices.
    """
    tmp_image: np.ndarray = generate_grayscale_image_from(image)

    if not "gray16" in filename:
        cv2.imwrite(f'{PREPROCESSING_GENERATED_FILE_PREFIX}gray16_{filename}', tmp_image)

    uniques = np.sort(np.unique(tmp_image))

    result1 = np.zeros(shape=(len(uniques), len(uniques)))
    result2 = np.zeros(shape=(len(uniques), len(uniques)))
    result4 = np.zeros(shape=(len(uniques), len(uniques)))
    result8 = np.zeros(shape=(len(uniques), len(uniques)))
    result16 = np.zeros(shape=(len(uniques), len(uniques)))
    result32 = np.zeros(shape=(len(uniques), len(uniques)))

    result = [result1, result2, result4, result8, result16, result32]

    for i in range(tmp_image.shape[0]):
        for j in range(tmp_image.shape[1]):
            for c, k in enumerate(result):
                if i + m.pow(2, c) < tmp_image.shape[0] and j + m.pow(2, c) < tmp_image.shape[1]:
                    k[index(uniques, tmp_image[i, j])] \
                        [index(uniques, tmp_image[i + int(m.pow(2, c)), j + int(m.pow(2, c))])] += 1

    for i in result:
        i = i / sum(i)
    return result


def operation_hue_invariants(filename, image) -> tuple:
    """
    Calculate the Hu moments of a grayscale image and the Hu moments of the H, S and V channels of a HSV image.

    Args:
        filename (str): The name of the file.
        image (np.ndarray): The image.
    
    Returns:
        tuple: The Hu moments of the grayscale image and the Hu moments of the H, S and V channels of a HSV image.
    """
    result_grayscale = cv2.HuMoments(cv2.moments(operation_gray(filename, image, 256), binaryImage=False)).flatten()
    
    tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    result_h = cv2.HuMoments(cv2.moments(tmp_image[:, :, 0], binaryImage=False)).flatten()
    result_s = cv2.HuMoments(cv2.moments(tmp_image[:, :, 1], binaryImage=False)).flatten()
    result_v = cv2.HuMoments(cv2.moments(tmp_image[:, :, 2], binaryImage=False)).flatten()

    return (result_grayscale, (result_h, result_s, result_v))

def process_image(filename, operation) -> tuple:
    """
    Process an image with a specific operation.

    Args:
        filename (str): The name of the file.
        operation (str): The operation to perform.
    
    Returns:
        tuple: The new filepath for the processed image and the result of the operation.
    """
    image = cv2.imread(f"{USER_GENERATED_FILE_PREFIX}{filename}")
    
    new_filepath = None
    result = None

    if operation == "gray16":
        result = operation_gray(filename, image, 16)
    elif operation == "gray256":
        result = operation_gray(filename, image, 256)
    elif operation == "hist16":
        result = operation_hist(filename, image, 16)
    elif operation == "hist256":
        result = operation_hist(filename, image, 256)
    elif operation == "histHSV":
        result = operation_hist_hsv(filename, image)
    elif operation == "coocurrence":
        result = operation_coocurrence(filename, image)
    elif operation == "hueInvariants":
        result = operation_hue_invariants(filename, image)

    return (new_filepath, result)