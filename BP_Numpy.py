import numpy as np
import random
import os


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NerualNetwork:
    def __init__(self, sizes):
        """
        :param sizes:[784,128,64,10]
        """
        # size : [784,128,64,10]
        # w:[ch_in,ch_out]
        # b:[ch_out]
        self.sizes = sizes
        self.num_layers = len(sizes) - 1
        self.weights = [
            np.random.randn(ch1, ch2) for ch1, ch2 in zip(sizes[:-1], sizes[1:])
        ]  # [784,30],[30,10]
        # z = xw + b [30,1]
        self.biases = [np.random.randn(1, ch) for ch in sizes[1:]]

    def forward(self, x):
        """
        :param x: [batch_size,784]
        :return: [batch_size,10]
        """
        # [batch_size,784]*[784,128]=>[batch_size,128]+[batch_size,128]=>[batch_size,128]
        # [batch_size,128]*[128,64]=>[batch_size,64]+[batch_size,64]=>[batch_size,64]
        # [batch_size,64]*[64,10]=>[batch_size,10]+[batch_size,10]=>[batch_size,10]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(x, w) + b
            x = sigmoid(z)
        return x

    def backwprop(self, x, y):
        """
        :param x: [batch_size,784]
        :param y: [batch_size,10]  one_hot encoding
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # 1. forward
        # save activation for every layer
        activations = [x]
        # save z for every layer
        zs = []
        activation = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
        loss = np.power(activations[-1] - y, 2).sum()

        # 2. backward
        # 2.1 compute gradient on output layer
        delta = activations[-1] * (1 - activations[-1]) * (activations[-1] - y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].T, delta)

        # 2.2 compute hidden gradient
        for l in range(2, self.num_layers + 1):
            l = -l
            z = zs[l]
            a = activations[l]
            delta = np.dot(delta, self.weights[l + 1].T) * a * (1 - a)
            nabla_b[l] = delta
            nabla_w[l] = np.dot(activations[l - 1].T, delta)
        return nabla_w, nabla_b, loss

    def train(self, training_data, epochs, batchsz, lr, test_data):
        """
        :param training_data: list of (x,y)
        :param epochs: 1000
        :param batchsz: 10
        :param lr: 0.01
        :param test_data: list of (x,y)
        :return:
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + batchsz] for k in range(0, n, batchsz)
            ]
            # for every batch in current batch
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, lr)

            if test_data:
                print((
                    "Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test),
                    loss,
                ))
            else:
                print(("Epoch {0} complete".format(j)))

    def update_mini_batch(self, batch, lr):
        """
        :param batch:  list of (x,y)
        :param lr: 0.01
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        loss = 0

        # for every sample in current batch
        for x, y in batch:
            # list of every w/b gradient
            nabla_w_, nabla_b_, loss_ = self.backwprop(x.T, y.T)
            nabla_w = [accu + cur for accu, cur in zip(nabla_w, nabla_w_)]
            nabla_b = [accu + cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_


        nabla_w = [w / len(batch) for w in nabla_w]
        nabla_b = [b / len(batch) for b in nabla_b]
        loss = loss / len(batch)

        # w = w - lr * nabla_w
        self.weights = [w - lr * nabla for w, nabla in zip(self.weights, nabla_w)]
        self.biases = [b - lr * nabla for b, nabla in zip(self.biases, nabla_b)]

        return loss

    def evaluate(self, test_data):
        """
        y is not one-hot enconding
        :param test_data: list of (x,y)
        :return:
        """
        result = [(np.argmax(self.forward(x.T)), y) for x, y in test_data]
        correct = sum(int(pred == y) for pred, y in result)
        return correct


def main():
    import mnist_loader

    # loading the MNIST data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # Set up a Network with 30 hidden neurons
    net = NerualNetwork([784, 128, 64, 10])
    # Use stochastic gradient descent to learn from the MNIST training data over
    # 30 epoches, with a mini-batch size of 10, and a learning rate of n=3.0
    net.train(training_data, 1000, 10, 0.1, test_data=test_data)


if __name__ == "__main__":
    main()
