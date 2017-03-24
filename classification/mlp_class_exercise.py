"""
Partially based on
https://triangleinequality.wordpress.com/2014/03/31/neural-networks-part-2/
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    print('Install seaborn (pip install seaborn) to get better graphs')

try:
    from tqdm import tqdm
except:
    print('Install tqdm (pip install tqdm) to get progress bars')

#activation functions
def _sigmoid(x):
    """Numerically stable sigmoid
    """
    if x > 0:
        return 1.0/(1.0+np.exp(-x))
    else:
        return np.exp(x)/(1.0 + np.exp(x))
sigmoid = np.vectorize(_sigmoid)

def sigmoid_prime(x):
    return (1-sigmoid(x))*sigmoid(x)

def identity(x):
    return x

def identity_prime(x):
    return 1.0

def tanh(x):
    return np.tanh(x)

def tanh_prime(y):
    return 1 - np.tanh(y)**2

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    relued = x > 0.0
    return relued.astype(np.float32)

def softmax(w):
    raise ValueError('please, implement me')


class MLP_Classification(object):
    def __init__(self, x_, y_, architecture, learning_rate_decay_constant = 0., enable_plot=False):
        """
        x_ = images
        y_ = labels
        architecture is a tuple containing the parameters for every layer:
        architecture = (layer_0 = (number_of_inputs,
                                    activation_function_0,
                                    derivative_activation_function_0)
                            .
                            .
                            .
                        layer_n = (number_of_units,
                                   activation_function_n,
                                   derivative_activation_function_n)
        learning_rate_decay_constant = constant k decreasing the initial learning lr_0
                                       rate as lr_(num_epochs) = lr_0 / (1 - k * num_epochs)
        enable_plot is a boolean (True for plotting the results during training)
        REMARK: input and output layers have size matching the dimensionality of
        the data and the number of classes, respectively
        """
        self.X = x_
        self.y = y_
        self.lr_decay_constant = learning_rate_decay_constant
        self.enable_plot = enable_plot
        self.n_layers = len(architecture) #the length of the tuple corresponds to the number of layers
        self.layers_size = [layer[0] for layer in architecture] #number of neurons per layer (excluding biases)
        self.activation_functions = [layer[1] for layer in architecture] #activation functions per layer
        self.fprimes = [layer[2] for layer in architecture] #derivative of each activation function
        self.build_net() #build the architecture described in architecture



    def build_net(self):
        ### We create lists to store the matrices needed by the chosen architecture
        self.inputs = [] #list of inputs (it contains the input to each layer of the network)
        self.outputs = [] #list of outputs (idem)
        self.W = [] #list of weight matrices (per layer)
        self.b_ = [] #list of bias vectors (per layer)
        self.errors = [] #list of error vectors (per layer)

        ### Initialise weights and biases with values sampled from a normal
        # distribution, while 0s are used to initialise the inputs, outputs, and
        # errors arrays. This operation is done, again, per layer.
        # The last layer is treated separately
        for layer in range(self.n_layers - 1):
            n = self.layers_size[layer]
            m = self.layers_size[layer + 1]
            self.W.append(np.random.normal(0, .1, (m, n)))
            self.b_.append(np.random.normal(0, .1, (m, 1)))
            self.inputs.append(np.zeros((n, 1)))
            self.outputs.append(np.zeros((n, 1)))
            self.errors.append(np.zeros((n, 1)))

        n = self.layers_size[-1]
        self.inputs.append(np.zeros((n, 1)))
        self.outputs.append(np.zeros((n, 1)))
        self.errors.append(np.zeros((n, 1)))

    def feedforward(self, x):
        """
        x is propagated through the architecture
        """
        x = np.expand_dims(x,axis=1)
        self.inputs[0] = x
        self.outputs[0] = x

        for i in range(1, self.n_layers):
            self.inputs[i] = self.W[i - 1].dot(self.outputs[i - 1]) + self.b_[i - 1]
            self.outputs[i] = self.activation_functions[i](self.inputs[i])

        return self.outputs[-1]

    def backprop(self,x,y, iteration_number):
        #Weigth matrices and biases are updated based on a single input x and its target y
        # learning rate decay in time (really slow step decay is desirable)
        self.learning_rate = np.true_divide(self.learning_rate,  (1 + self.lr_decay_constant * iteration_number))
        self.out_deltas = (x - y) # -(y - x)
        self.errors[-1] = self.outputs[-1] * self.out_deltas
        n = self.n_layers - 2
        #Again, we will treat the last layer separately
        for i in range(n, 0, -1):
            self.errors[i] = self.fprimes[i](self.inputs[i]) * self.W[i].T.dot(self.errors[i + 1])
            self.W[i] = self.W[i] - self.learning_rate * np.outer(self.errors[i + 1],self.outputs[i])
            self.b_[i] = self.b_[i] - self.learning_rate * self.errors[i + 1]

        self.W[0] = self.W[0] - self.learning_rate * np.outer(self.errors[1],self.outputs[0])
        self.b_[0] = self.b_[0] - self.learning_rate * self.errors[1]

    def compute_accuracy(self, x_, y_):
        # subtract labels to targets
        raise ValueError('please, implement me')


    def train(self,n_iter, learning_rate = 1):
        self.learning_rate = learning_rate
        n = self.X.shape[0]

        if self.enable_plot:
            self.initialise_figure()

        errors_plot = []
        accuracy_plot = []
        learning_rate_plot = []

        for repeat in tqdm(range(n_iter)):
            predictions = []
            targets = []
            errors = []
            out_xs = []
            #We shuffle the order in which we go through the inputs on each iter.
            index = list(range(n)) #get an array of the size of inputs (batch)
            # np.random.shuffle(index) #shuffle it

            for row in index:
                x = self.X[row] #input
                y = self.y[row] #target
                y = np.expand_dims(y, axis=1)
                out_x = self.feedforward(x) #forward pass
                # print("{} -> {}".format(np.argmax(np.squeeze(out_x)), np.argmax(y)))

                self.backprop(out_x,y, repeat) #update weights
                errors.append(self.out_deltas)
                predictions.append(np.argmax(out_x))
                targets.append(np.argmax(y))

            errors_plot.append(np.mean(np.abs(errors)))
            accuracy_plot.append(self.compute_accuracy(np.asarray(predictions), np.asarray(targets)))
            learning_rate_plot.append(self.learning_rate)

            if repeat % 10 == 0 and repeat != 0:
                if self.lr_decay_constant != 0:
                    print('the learnign rate is: ' + str(self.learning_rate))

                accuracy = self.compute_accuracy(np.asarray(predictions), np.asarray(targets))
                print ('accuracy at step ' + str(repeat) + ': ', str(accuracy))
                print ('error at step ' + str(repeat) + ': ', str(np.mean(np.abs(errors))))

            if self.enable_plot and repeat % 1 == 0: #if plot is enabled
                self.plot_fit_during_train(errors_plot, accuracy_plot, learning_rate_plot) #plot

    def initialise_figure(self):
        self.figure, self.ax_arr  = plt.subplots(1,3)
        self.ax1 = self.ax_arr[0]
        self.ax2 = self.ax_arr[1]
        self.ax3 = self.ax_arr[2]
        plt.ion() #enable interactive plotting
        self.ax1.set_title('Error')
        self.ax2.set_title('Accuracy')
        self.ax3.set_title('Learning rate')

        self.epsilon = 1e-10 # matplotlib needs to pause for a little while in order to display the update of the weights

    def plot_fit_during_train(self, errors_plot, accuracy_plot, learning_rate_plot):
        plt.cla()
        epochs = range(len(errors_plot))
        self.ax1.plot(epochs, errors_plot, linewidth=2, color='black')
        self.ax2.plot(epochs, accuracy_plot, linewidth=2, color='black')
        self.ax3.plot(epochs, learning_rate_plot, linewidth=2, color='black')
        plt.pause(self.epsilon)
        plt.show()


"""
Data acquisition and preprocessing
"""
def acquire_data(path):
    raise ValueError('please, implement me')

def standardise_images(images):
    raise ValueError('please, implement me')

def dense2one_hot(labels):
    raise ValueError('please, implement me')

def test_classification(plots=True):
    #Load data
    path = 'data/digits.csv'
    images, labels = acquire_data(path)
    images_std = standardise_images(images)
    labels_oh = dense2one_hot(labels)
    image_size = images_std.shape[1]
    num_classes = labels_oh.shape[1]
    print ('image shape: ', images_std.shape)
    print ('num classes: ', num_classes)
    #set the architecture
    param = ()
    #Set learning rate.
    lrs = []

    for learning_rate in lrs:
        print('training with learning rate ', lrs)
        Network = MLP_Classification(images_std,labels_oh,param, learning_rate_decay_constant = 1e-15, enable_plot=True)
        Network.train(500, learning_rate=learning_rate)

test_classification()
