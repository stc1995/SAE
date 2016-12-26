import math
import numpy
import time
import scipy.io
import scipy.optimize

""" Normalize the dataset provided as input """

example_num = 100

def normalizeDataset(dataset):
    """ Remove mean of dataset """
    dataset = dataset - numpy.mean(dataset)

    """ Truncate to +/-3 standard deviations and scale to -1 to 1 """
    std_dev = 3 * numpy.std(dataset)
    dataset = numpy.maximum(numpy.minimum(dataset, std_dev), -std_dev) / std_dev

    """ Rescale from [-1, 1] to [0.1, 0.9] """
    dataset = (dataset + 1) * 0.4 + 0.1

    return dataset


""" Randomly samples image patches, normalizes them and returns as dataset """


def loadDataset(num_patches, patch_side):

    # images = scipy.io.loadmat("IMAGES.mat")
    # images = images['IMAGES']
    # images = scipy.io.loadmat("d:/RecentlyUsed/Graduation Project/adData.mat")
    # images = images['adData']
    images = scipy.io.loadmat("d:/RecentlyUsed/Graduation Project/adData.mat")
    images = images['adData']

    """ Initialize dataset as array of zeros """
    dataset = numpy.zeros((patch_side*patch_side*patch_side, num_patches*example_num))

    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """
    # rand = numpy.random.RandomState(int(time.time()))
    rand = numpy.random.RandomState(1000000)

    # image_indices = rand.randint(512 - patch_side, size = (num_patches, 2))
    image_indices1 = rand.randint(81 - patch_side, size=(num_patches))
    image_indices2 = rand.randint(97 - patch_side, size=(num_patches))
    image_indices3 = rand.randint(83 - patch_side, size=(num_patches))
    # image_number = rand.randint(10, size = num_patches)
    # image_number = rand.randint(116, size=num_patches)

    """ Sample 'num_patches' random image patches """
    for j in range(example_num):
        for i in range(num_patches):
            """ Initialize indices for patch extraction """
            # index1 = image_indices[i, 0]
            # index2 = image_indices[i, 1]
            # index3 = image_number[i]
            index1 = image_indices1[i]
            index2 = image_indices2[i]
            index3 = image_indices3[i]
            # index4 = image_number[i]
            """ Extract patch and store it as a column """
            # patch = images[index1:index1+patch_side, index2:index2+patch_side, index3]
            patch = images[0, j][3][0][0][4][index1:index1 + patch_side, index2:index2 + patch_side, index3:index3+patch_side]
            patch = patch.flatten()
            dataset[:, i] = patch
    """ Normalize and return the dataset """
    dataset = normalizeDataset(dataset)
    return dataset


def sigmoid(x):
    return (1 / (1 + numpy.exp(-x)))


def getFeature():
    num_patches = 1000
    patch_side = 8
    patch_side2 = 5

    opt_W1 = numpy.loadtxt("D:/Users/stc/PycharmProjects/SAE/ad_400_W1.txt")
    opt_b1 = numpy.loadtxt("D:/Users/stc/PycharmProjects/SAE/ad_400_b1.txt")
    opt_b1 = numpy.matrix(opt_b1).transpose()
    input = loadDataset(num_patches, patch_side)
    feature = sigmoid(numpy.dot(opt_W1, input) + opt_b1)
    feature = feature.transpose()
    print('feature')
    print(type(feature))
    print(feature.shape)
    feature2 = numpy.zeros((example_num, patch_side2*patch_side2*patch_side2*num_patches))
    for i in range(example_num):
        feature2[i, :] = feature[i*num_patches:(i+1)*num_patches].reshape(1, patch_side2*patch_side2*patch_side2*num_patches)
    print('feature2')
    print(type(feature2))
    print(feature2.shape)

    numpy.savetxt("ad_400_feature.txt", feature2, fmt=['%s']*feature2.shape[1], newline='\r\n')

getFeature()
