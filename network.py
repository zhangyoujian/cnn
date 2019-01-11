import numpy as np
import sys
import timeit
from itertools import chain
from scipy import misc
import matplotlib.pyplot as plt
import scipy.signal


class sigmoid:
    def __init__(self):
        pass

    def forward(self, X):
        self.cache = X
        self.feature_map = 1.0 / (1.0 + np.exp(-X))
        return self.feature_map, 0

    def backward(self, delta):
        self.delta_X = delta * (self.feature_map) * (1 - self.feature_map)
        return self.delta_X

class tanh:
    def forward(self, X):
        self.cache = X
        self.feature_map = np.tanh(X)
        return self.feature_map, 0

    def backward(self, delta):
        self.delta_X = delta * (1 - self.feature_map * self.feature_map)
        return self.delta_X


class ReLu:
    def __init__(self):
        pass

    def forward(self,X):
        self.cache = X
        self.feature_map = np.maximum(X, 0)
        return self.feature_map, 0

    def backward(self, delta):
        self.delta_X = delta * (self.cache >= 0)
        return self.delta_X



class ConvolutionLayer(object):
    def __init__(self,layer_size, kernel_size, fan, stride,padding):

        self.depth,self.height,self.width = layer_size

        self.stride = stride
        self.padding = padding


        f = np.sqrt(6) / np.sqrt(fan[0] + fan[1])
        epsilon = 1e-6

        self.weight = np.random.uniform(-f, f + epsilon, kernel_size)
        self.bias = np.random.uniform(-f, f + epsilon, kernel_size[0])

        self.gradient_history = np.zeros(kernel_size)
        self.bias_history = np.zeros(kernel_size[0])
        self.m_kernel = np.zeros(kernel_size)
        self.m_bias = np.zeros(kernel_size[0])
        self.v_kernel = np.zeros(kernel_size)
        self.v_bias = np.zeros(kernel_size[0])
        self.timestamp = 0

    def forward(self,X):
        padding = self.padding
        stride = self.stride
        N,D,H,W = X.shape
        fm,fc,fh,fw = self.weight.shape

        conv_h = (H - fh + 2 * padding)//stride +1
        conv_w = (W - fw + 2 * padding) // stride + 1
        self.feature_map = np.zeros([N,fm,conv_h,conv_w])
        x_padded = np.pad(X,((0,0),(0,0),(padding,padding),(padding,padding)),'constant')

        if stride==1:
            weight_180 = np.rot90(self.weight, 2, (2, 3))
            for img in range(N):
                for conv_depth in range(fm):
                    for inp_depth in range(D):
                        self.feature_map[img, conv_depth] += scipy.signal.convolve2d(x_padded[img, inp_depth],
                                                                                     weight_180[conv_depth, inp_depth],
                                                                                     mode='valid')
                    self.feature_map[img, conv_depth] += self.bias[conv_depth]
        else:
            for img in range(N):
                for conv_depth in range(fm):
                    for h in range(0, H + 2 * padding - fh + 1, stride):
                        for w in range(0, W + 2 * padding - fw + 1, stride):
                            self.feature_map[img, conv_depth, h // stride, w // stride] = \
                                np.sum(x_padded[img, :, h:h + fh, w:w + fw] * self.weight[conv_depth, :, :, :]) + \
                                self.bias[conv_depth]

        self.cache = X
        return self.feature_map, np.sum(np.square(self.weight))

    def backward(self,delta):
        X = self.cache
        padding = self.padding
        stride = self.stride

        N, D, H, W = X.shape
        fm, fc, fh, fw = self.weight.shape
        conv_h = (H - fh + 2 * padding) // stride + 1
        conv_w = (W - fw + 2 * padding) // stride + 1
        x_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
        delta_X_padded = np.zeros(x_padded.shape)
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

        for img in range(N):
            for conv_depth in range(fm):
                for h in range(0, H + 2 * padding - fh + 1, stride):
                    for w in range(0, W + 2 * padding - fw + 1, stride):
                        delta_X_padded[img, :, h:h + fh, w:w + fw] += delta[img, conv_depth, h // stride, w // stride] * \
                                                                        self.weight[conv_depth]


        if padding > 0:
            self.delta_X = delta_X_padded[:, :, padding:-padding, padding:-padding]
        else:
            self.delta_X = delta_X_padded[:]

            # as self.delta_X.shape == X.shapesert
            # Delta kernel
        for img in range(N):
            for kernel_num in range(fm):
                for h in range(conv_h):
                    for w in range(conv_w):
                        self.delta_w[kernel_num, :, :, :] += delta[img, kernel_num, h, w] * x_padded[img, :,
                                                                                            h * stride:h * stride + fh,
                                                                                            w * stride:w * stride + fw]

            # Delta Bias
        self.delta_b = np.sum(delta, (0, 2, 3))
        return self.delta_X

    def update_kernel(self,parames):
        method = parames.get("method","")
        alpha = parames.get("alpha",0.001)
        zeta = parames.get("zeta",0.01)
        batch_size = parames.get("batch",1)
        beta1 = parames.get("beta1",0.9)
        beta2 =parames.get("beta2",0.999)
        fudge_factor = 1e-8

        if method == "adagrad":
            self.gradient_history += np.square(self.delta_w + (zeta * self.weight / batch_size))
            self.bias_history += np.square(self.delta_b)
            self.weight -= alpha * (self.delta_w + (zeta * self.weight / batch_size)) / (
                        np.sqrt(self.gradient_history) + fudge_factor)
            self.bias -= alpha * self.delta_b / (np.sqrt(self.bias_history) + fudge_factor)
        elif method == "adam":
            self.timestamp += 1
            alpha = alpha * np.sqrt(1 - np.power(beta2, self.timestamp)) / (1 - np.power(beta1, self.timestamp))
            self.m_kernel = beta1 * self.m_kernel + (1 - beta1) * (self.delta_w + (zeta * self.weight / batch_size))
            self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.delta_b
            self.v_kernel = beta2 * self.v_kernel + (1 - beta2) * np.square(
                (self.delta_w + (zeta * self.weight / batch_size)))
            self.v_bias = beta2 * self.v_bias + (1 - beta2) * np.square(self.delta_b)

            self.weight -= np.divide(alpha * self.m_kernel, (np.sqrt(self.v_kernel) + fudge_factor))
            self.bias -= np.divide(alpha * self.m_bias, (np.sqrt(self.v_bias) + fudge_factor))
        else:
            self.weight -= alpha * (self.delta_w + zeta * self.weight / batch_size)
            self.bias -= alpha * self.delta_b
        pass



class MaxpoolLayer(object):
    def __init__(self,params):
        self.factor=params.get("stride",2)

    def forward(self,X):
        factor = self.factor
        N,D,H,W = X.shape
        self.cache = [X,factor]
        self.feature_map = X.reshape(N, D, H // factor, factor, W // factor, factor).max(axis=(3, 5))
        return self.feature_map, 0

    def backward(self, delta):

        X, factor = self.cache
        if len(delta.shape) != 4:  # then it must be 2
            # assert delta.shape[0] == X.shape[0]
            delta = delta.reshape(self.feature_map.shape)

        fmap = np.repeat(np.repeat(self.feature_map, factor, axis=2), factor, axis=3)
        dmap = np.repeat(np.repeat(delta, factor, axis=2), factor, axis=3)

        self.delta_X = np.zeros(X.shape)
        self.delta_X = (fmap == X) * dmap

        return self.delta_X



class Fullconnectlayer(object):
    def __init__(self, layer_size, kernel_size):

        self.nodes = layer_size
        f = np.sqrt(6) / np.sqrt(kernel_size[0] + kernel_size[1])
        epsilon = 1e-6
        self.weight = np.random.uniform(-f, f + epsilon, kernel_size)
        self.bias = np.random.uniform(-f, f + epsilon, kernel_size[1])
        self.gradient_history = np.zeros(kernel_size)
        self.bias_history = np.zeros(kernel_size[1])
        self.m_kernel = np.zeros(kernel_size)
        self.m_bias = np.zeros(kernel_size[1])
        self.v_kernel = np.zeros(kernel_size)
        self.v_bias = np.zeros(kernel_size[1])
        self.timestamp = 0
        pass



    def forward(self,X):
        weight,bias = self.weight,self.bias
        self.cache = (X,weight,bias)
        self.activations = np.dot(X,weight)+bias

        return self.activations, np.sum(np.square(self.weight))

    def backward(self, delta):

        X, weight, bias = self.cache
        self.delta_X = np.dot(delta, weight.T)
        self.delta_w = np.dot(X.T, delta)

        self.delta_b = np.sum(delta, axis=0)
        return self.delta_X


    def update_kernel(self, parames):
        method = parames.get("method", "")
        alpha = parames.get("alpha", 0.001)
        mu = parames.get("mu",0.9)
        zeta = parames.get("zeta", 0.01)
        batch_size = parames.get("batch", 1)
        beta1 = parames.get("beta1", 0.9)
        beta2 = parames.get("beta2", 0.999)
        fudge_factor = 1e-8

        if method == "adagrad":
            self.gradient_history += np.square(self.delta_w + (zeta * self.weight / batch_size))
            self.bias_history += np.square(self.delta_b)
            # print("\n\n\n\n\n\n")
            # print(self.gradient_history[0])
            # sys.exit()
            # print(alpha*(self.delta_K + (zeta*self.kernel/batch_size))/(np.sqrt(self.gradient_history) + fudge_factor))
            self.weight -= np.divide(alpha * (self.delta_w + (zeta * self.weight / batch_size)),
                                     (np.sqrt(self.gradient_history) + fudge_factor))
            self.bias -= np.divide(alpha * self.delta_b, (np.sqrt(self.bias_history) + fudge_factor))
        elif method == "gd_momentum":
            new_delta_K = alpha * (self.delta_w + (zeta * self.weight / batch_size)) + mu * self.gradient_history
            new_delta_b = alpha * self.delta_b + mu * self.bias_history
            self.weight -= new_delta_K
            self.bias -= new_delta_b
            self.gradient_history = self.delta_w + (zeta * self.weight / batch_size)
            self.bias_history = self.delta_b
        elif method == "adam":
            self.timestamp += 1
            alpha = alpha * np.sqrt(1 - np.power(beta2, self.timestamp)) / (1 - np.power(beta1, self.timestamp))
            self.m_kernel = beta1 * self.m_kernel + (1 - beta1) * (self.delta_w + (zeta * self.weight / batch_size))
            self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.delta_b
            self.v_kernel = beta2 * self.v_kernel + (1 - beta2) * np.square(
                (self.delta_w + (zeta * self.weight / batch_size)))
            self.v_bias = beta2 * self.v_bias + (1 - beta2) * np.square(self.delta_b)

            self.weight -= np.divide(alpha * self.m_kernel, (np.sqrt(self.v_kernel) + fudge_factor))
            self.bias -= np.divide(alpha * self.m_bias, (np.sqrt(self.v_bias) + fudge_factor))
        else:
            self.weight -= alpha * (self.delta_w + zeta * self.weight / batch_size)
            self.bias -= alpha * self.delta_b
        pass


class softmax(object):
    def __init__(self):
        pass

    def forward(self,X):
        self.cache = X
        dummy = np.exp(X)
        self.Y = dummy / np.sum(dummy, axis=1, keepdims=True)
        return self.Y, 0

    def backward(self, output):

        self.delta_X = (self.Y - output) / self.Y.shape[0]
        return self.delta_X

    def softmax_loss(self, Y, output):

        assert Y.shape == output.shape
        epsilon = 1e-10
        self.loss = (-output * np.log(Y + epsilon)).sum() / Y.shape[0]
        pass



class cnn(object):
    def __init__(self, t_input, t_output, v_input, v_output,x_test,y_test):
       parames = {}
       conv1 = ConvolutionLayer((6,28,28),(6,1,5,5),(784,4704),1,2)
       relu1 = ReLu()
       parames.setdefault('stride',2)
       pool2 = MaxpoolLayer(parames)

       conv3 = ConvolutionLayer((16,10,10),(16,6,5,5),(1176,1600),1,0)
       relu3 = ReLu()
       pool4 = MaxpoolLayer(parames)

       fc5 = Fullconnectlayer(120,(400,120))
       sigmoid5 = sigmoid()

       fc6 = Fullconnectlayer(84,[120,84])
       sigmoid6 = sigmoid()

       output = Fullconnectlayer(10,[84,10])
       softmaxOut = softmax()
       self.layers = [conv1, relu1, pool2, conv3, relu3, pool4, fc5, sigmoid5, fc6, sigmoid6, output, softmaxOut]

       self.X = t_input
       self.Y = t_output
       self.Xv = v_input
       self.Yv = v_output
       self.x_test = x_test
       self.y_test = y_test

    @staticmethod
    def one_image_time(X, layers):
       inp = X
       conv_time = 0.0
       fc_time = 0.0
       layer_time = []

       for layer in layers:
           start = timeit.default_timer()
           if isinstance(layer, Fullconnectlayer) and len(inp.shape) == 4:
               inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2] * inp.shape[3]))
           else:
               inp, ws = layer.forward(inp)
           stop = timeit.default_timer()
           layer_time += [stop - start]
           if isinstance(layer, (Fullconnectlayer, sigmoid, softmax)):
               fc_time += stop - start
           if isinstance(layer, (ConvolutionLayer, ReLu)):
               conv_time += stop - start
       return conv_time, fc_time, layer_time

    @staticmethod
    def feedForward(X, layers):

       inp = X
       wsum = 0
       for layer in layers:
           if isinstance(layer, Fullconnectlayer) and len(inp.shape) == 4:
               inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2] * inp.shape[3]))
           else:
               inp, ws = layer.forward(inp)
           wsum += ws
       return inp, wsum

    @staticmethod
    def backpropagation(Y, layers):
        delta = Y
        for layer in layers[::-1]:
            delta = layer.backward(delta)

    @staticmethod
    def update_parameters(layers,batch_size,a,z,m):
       params={"batch":batch_size,"alpha":a,"zeta":z,"method":m}
       for layer in layers:
           if isinstance(layer, (ConvolutionLayer, Fullconnectlayer)):
               layer.update_kernel(params)

    @staticmethod
    def loss_function(pred, t, params):
        w_sum = params.get("wsum",0)
        z = params.get("zeta",0)
        epsilon = 1e-10
        return ((-t * np.log(pred + epsilon)).sum() + (z / 2) * w_sum) / pred.shape[0]

    @staticmethod
    def plots(x, y, z, steps):
        try:
            plt.figure(1)
            plt.plot(x, '-bo', label="Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss', fontsize=18)
            plt.title('Training Error rate vs Number of iterations')
            plt.savefig("Loss_function_vs_iter.jpeg")
        except:
            pass

        try:
            plt.figure(2)
            plt.plot(steps, y, '-bo', label="Training Loss")
            plt.plot(steps, z, '-ro', label="Validation Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss Value', fontsize=18)
            plt.title('Training and Validation error rates vs number of iterations')
            plt.legend(loc='upper right')
            plt.savefig("error_rates.jpeg")
        except:
            pass
        pass



    def lenet_train(self,params):

        batch = params.get("batch",50)
        alpha = params.get("alpha",0.01)
        zeta =params.get("zeta",0)
        method = params.get("method","adam")
        epochs = params.get("epoch",4)
        print("Training on params: batch=", batch, " learning rate=", alpha, " L2 regularization=", zeta, " method=",
              method, " epochs=", epochs)
        self.loss_history = []
        self.gradient_history = []
        self.valid_loss_history = []
        self.step_loss = []
        print(method)
        X_train = self.X
        Y_train = self.Y
        assert X_train.shape[0] == Y_train.shape[0]
        num_batches = int(np.ceil(X_train.shape[0] / batch))
        step = 0
        steps = []
        X_batches = zip(np.array_split(X_train, num_batches, axis=0), np.array_split(Y_train, num_batches, axis=0))

        for ep in range(epochs):
            print("Epoch: ", ep, "===============================================")
            for x, y in X_batches:
                predictions, weight_sum = cnn.feedForward(x, self.layers)
                par = {"wsum":weight_sum,"zeta":zeta}
                loss = cnn.loss_function(predictions, y, par)
                self.loss_history += [loss]
                cnn.backpropagation(y, self.layers)  # check this gradient
                cnn.update_parameters(self.layers,x.shape[0], alpha, zeta, method)
                print("Step: ", step, ":: Loss: ", loss, "weight_sum: ", weight_sum)
                if step % 100 == 0:
                    pred, w = cnn.feedForward(self.Xv, self.layers)
                    par['wsum'] = w
                    par['zeta'] = zeta
                    v_loss = cnn.loss_function(pred, self.Yv, par)
                    print("Validation error: ", v_loss)
                    self.lenet_predictions(self.x_test, self.y_test)
                    steps += [step]
                    self.valid_loss_history += [v_loss]
                    self.step_loss += [loss]
                step += 1

            XY = list(zip(X_train, Y_train))
            np.random.shuffle(XY)
            new_X, new_Y = zip(*XY)
            assert len(new_X) == X_train.shape[0] and len(new_Y) == len(new_X)
            X_batches = zip(np.array_split(new_X, num_batches, axis=0), np.array_split(new_Y, num_batches, axis=0))
        cnn.plots(self.loss_history, self.step_loss, self.valid_loss_history, steps)
        pass


    def lenet_predictions(self,X, Y):
        """
        Predicts the ouput and computes the accuracy on the dataset provided.
        Input:
            X: Input of shape (Num, depth, height, width)
            Y: True output of shape (Num, Classes)
        """
        start = timeit.default_timer()
        predictions, weight_sum = cnn.feedForward(X, self.layers)
        stop = timeit.default_timer()

      #  loss = cnn.loss_function(predictions, Y, weight_sum, 0)
        y_true = np.argmax(Y, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        count = 0
        for i in range(len(Y)):
            if y_true[i]==y_pred[i]:
                count = count+1

        print("Dataset accuracy: ", count/len(Y) * 100)
        print("FeedForward time:", stop - start)
        pass





















