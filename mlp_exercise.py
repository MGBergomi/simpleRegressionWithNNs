"""
Based on
https://triangleinequality.wordpress.com/2014/03/31/neural-networks-part-2/
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#activation functions
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_prime(x):
    ex=np.exp(-x)
    return ex/(1+ex)**2

def identity(x):
    return x

def identity_prime(x):
    return 1

def tanh(x):
    raise ValueError('Please, implement me!')

def tanh_prime(y):
    raise ValueError('Please, implement me!')

def relu(x):
    raise ValueError('Please, implement me!')

def relu_prime(x):
    raise ValueError('Please, implement me!')

class MLP_Regression(object):
    def __init__(self, x_, y_, architecture, enable_plot=False):
        """
        x_ = input data
        y_ = targets
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
        enable_plot is a boolean (True for plotting the results during training)
        """
        self.X = x_
        self.y = y_
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
            self.W.append(np.random.normal(0, 1, (m, n)))
            self.b_.append(np.random.normal(0, 1, (m, 1)))
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

    def backprop(self,x,y):
        #Weigth matrices and biases are updated based on a single input x and its target y
        self.errors[-1] = self.fprimes[-1](self.outputs[-1]) * (x - y)
        n = self.n_layers - 2
        #Again, we will treat the last layer separately
        for i in xrange(n, 0, -1):
            self.errors[i] = self.fprimes[i](self.inputs[i]) * self.W[i].T.dot(self.errors[i + 1])
            self.W[i] = self.W[i] - self.learning_rate * np.outer(self.errors[i + 1],self.outputs[i])
            self.b_[i] = self.b_[i] - self.learning_rate * self.errors[i + 1]

        self.W[0] = self.W[0] - self.learning_rate * np.outer(self.errors[1],self.outputs[0])
        self.b_[0] = self.b_[0] - self.learning_rate * self.errors[1]

    def train(self,n_iter, learning_rate = 1):
        #Updates the weights after comparing each input in X with y
        #repeats this process n_iter times.

        self.learning_rate = learning_rate
        n = self.X.shape[0]

        if self.enable_plot:
            self.initialise_figure()

        for repeat in range(n_iter):
            out_xs = []
            #We shuffle the order in which we go through the inputs on each iter.
            index = list(range(n)) #get an array of the size of inputs (batch)
            # np.random.shuffle(index) #shuffle it

            for row in index:
                x = self.X[row] #input
                y = self.y[row] #target
                out_x = self.feedforward(x) #forward pass
                print 'outputs ', out_x
                if np.isnan(np.squeeze(out_x)):
                    raise ValueError('The model diverged')

                self.backprop(out_x,y) #update weights
                out_xs.append(out_x)

            if self.enable_plot and repeat % 100 == 0: #if plot is enabled
                self.plot_fit_during_train(out_xs) #plot

    def initialise_figure(self):
        self.figure, self.ax  = plt.subplots(1,1)
        plt.ion() #enable interactive plotting
        self.ax.plot(self.X, self.y, label='Sine', linewidth=2, color='black')
        self.ax.plot(self.X, np.zeros_like(self.X), label="Learning Rate: "+str(self.learning_rate))
        self.epsilon = 1e-10 # matplotlib needs to pause for a little while in order to display the update of the weights

    def plot_fit_during_train(self,out_xs):
        plt.cla()
        out_xs = np.reshape(out_xs, [-1,1])
        self.ax.plot(self.X, self.y, label='Sine', linewidth=2, color='black')
        self.ax.plot(self.X, out_xs, label="Learning Rate: "+str(self.learning_rate))
        plt.pause(self.epsilon)
        plt.show()

    def predict(self, X):
        n = len(X)
        m = self.layers_size[-1]
        ret = np.ones((n,m))

        for i in range(len(X)):
            ret[i,:] = self.feedforward(X[i])

        return ret

def test_regression(plots=True):
    #Create data
    n = 20
    X = np.linspace(0,3*np.pi,num=n)
    X.shape = (n, 1)
    y = np.sin(X)
    #set the architecture
    param = ((1,0,0),(20, sigmoid, sigmoid_prime),(1,identity, identity_prime))
    #Set learning rate.
    lrs = [1.0]
    predictions = []
    for learning_rate in lrs:
        Network = MLP_Regression(X,y,param, True)
        Network.train(3000, learning_rate=learning_rate)
        pred = Network.predict(X)
        print 'pred = ', pred
        predictions.append([learning_rate, pred])


if __name__ == "__main__":
    test_regression()
