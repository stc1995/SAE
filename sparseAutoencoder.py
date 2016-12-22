import math
import numpy
import time
import scipy.io
import matplotlib.pyplot
import scipy.optimize

""" The Sparse Autoencoder class """


class SparseAutoEncoder(object):
    """ Initialization of Autoencoder object """

    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.rho = rho
        self.lamda = lamda
        self.beta = beta

        """ Set limits for accessing 'theta' values """
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = self.limit1 + visible_size * hidden_size
        self.limit3 = self.limit2 + hidden_size
        self.limit4 = self.limit3 + visible_size

        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """
        r = math.sqrt(6) / math.sqrt(hidden_size + visible_size + 1)
        rand = numpy.random.RandomState(int(time.time()))

        W1 = numpy.asarray(rand.uniform(low=-r, high=r, size=(hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low=-r, high=r, size=(visible_size, hidden_size)))

        """ Bias values are initialized to zero """

        b1 = numpy.zeros(hidden_size)
        b2 = numpy.zeros(visible_size)

        """ Create 'theta' by unrolling W1, W2, b1, b2 """

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))

    """ Returns elementwise sigmoid output of input array """

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(x))

    """ Returns the cost of the Autoencoder and gradient at a particular 'theta' """

    def sparseAutoencoderCost(self, theta, input):
        """ Extract weights and biases from 'theta' input """

        W1 = theta[self.limit0: self.limit1].reshape(self.hidden_size, self.visible_size)  #参数没有从外界传入时，可以用类变量，记得加self.
        W2 = theta[self.limit1: self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2: self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3: self.limit4].reshape(self.visible_size, 1)

        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """
        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = self.sigmoid(numpy.dot(W2, hidden_layer) + b2)

        """ Estimate the average activation value of the hidden layers """
        rho_cap = numpy.sum(hidden_layer, axis=1) / input.shape[1]

        """ Compute intermediate difference values using Backpropagation algorithm """
        diff = output_layer - input
        outputCost = 1 / (2 * input.shape[1]) * numpy.sum(numpy.multiply(diff, diff))
        regularizationCost = self.lamda / (2 * input.shape[1]) * (numpy.sum(numpy.multiply(W1, W1)) +
                                                                  numpy.sum(numpy.multiply(W2, W2)))
        KL_divergence = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                              (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))
        cost = outputCost + regularizationCost + KL_divergence

        delta3 = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))
        deltaKL = self.beta * (-self.rho / rho_cap + (1 - self.rho) / (1 - rho_cap))
        delta2 = numpy.multiply(numpy.dot(numpy.transpose(W2), delta3) + numpy.transpose(numpy.matrix(deltaKL)),
                                numpy.multiply(hidden_layer, 1 - hidden_layer))

        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """
        W2grad = 1 / input.shape[1] * numpy.dot(delta3, numpy.transpose(hidden_layer)) + self.lamda * W2
        b2grad = 1 / input.shape[1] * numpy.sum(delta3, axis=1)
        W1grad = 1 / input.shape[1] * numpy.dot(delta2, numpy.transpose(input)) + self.lamda * W1
        b1grad = 1 / input.shape[1] * numpy.sum(delta2, axis=1)

        """ Transform numpy matrices into arrays """

        W1grad = numpy.array(W1grad)
        W2grad = numpy.array(W2grad)
        b1grad = numpy.array(b1grad)
        b2grad = numpy.array(b2grad)

        """ Unroll the gradient values and return as 'theta' gradient """

        theta_grad = numpy.concatenate((W1grad.flatten(), W2grad.flatten(), b1grad.flatten(), b2grad.flatten()))

        return [cost, theta_grad]


""" Normalize the dataset provided as input """


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

    images = scipy.io.loadmat("IMAGES.mat")
    images = images['IMAGES']

    """ Initialize dataset as array of zeros """
    dataset = numpy.zeros((patch_side*patch_side, num_patches))

    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """
    rand = numpy.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size = (num_patches, 2))
    image_number = rand.randint(10, size = num_patches)

    """ Sample 'num_patches' random image patches """
    for i in range(num_patches):
        """ Initialize indices for patch extraction """
        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]
        """ Extract patch and store it as a column """
        patch = images[index1:index1+patch_side, index2:index2+patch_side, index3]
        patch = patch.flatten()
        dataset[:, i] = patch
    """ Normalize and return the dataset """

    dataset = normalizeDataset(dataset)
    return dataset


""" Visualizes the obtained optimal W1 values as images """
def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):
    """ Add the weights as a matrix of images """
    figure, axes = matplotlib.pyplot.subplots(nrows= hid_patch_side, ncols= hid_patch_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """
        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, vis_patch_side), cmap=matplotlib.pyplot.cm.gray, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

        """ Show the obtained plot """
    matplotlib.pyplot.show()


""" Loads data, trains the Autoencoder and visualizes the learned weights """
def executeSparseAutoEncoder():
    """ Define the parameters of the Autoencoder """
    vis_patch_side = 8
    hid_patch_side = 5
    rho = 0.01
    lamda = 0.0001
    beta = 3
    num_patches = 10000
    max_iterations = 400

    visible_size = vis_patch_side * vis_patch_side
    hidden_size = hid_patch_side * hid_patch_side
    """ Load randomly sampled image patches as dataset """
    training_data = loadDataset(num_patches, vis_patch_side)
    """ Initialize the Autoencoder with the above parameters """
    encoder = SparseAutoEncoder(visible_size, hidden_size, rho, lamda, beta)

    """ Run the L-BFGS algorithm to get the optimal parameter values """
    opt_solution = scipy.optimize.minimize(encoder.sparseAutoencoderCost, encoder.theta, args=(training_data,),
                                           method="L-BFGS-B", jac=True, options={"maxiter": max_iterations})

    opt_theta = opt_solution.x
    opt_W1 = opt_theta[encoder.limit0: encoder.limit1].reshape(hidden_size, visible_size)

    """ Visualize the obtained optimal W1 weights """

    visualizeW1(opt_W1, hid_patch_side, vis_patch_side)


executeSparseAutoEncoder()



