import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline

input_num = 10
feature_num = 1000

arr = [3, 7, 9.8, 4, 6, 2, 4.7, 8.1, 7.3, 2]
print(len(arr))
true_w = np.array(arr, dtype=np.float32).reshape(-1, 1)
print("true_w, shape", true_w.shape)
true_b = 7.4

features = np.random.normal(loc=0, scale=0.1, size=(feature_num, input_num)).astype(np.float32)
logits = features @ true_w + true_b

noise = np.random.normal(loc=0, scale=0.01, size=(feature_num, 1)).astype(np.float32)
logits += noise

labels = (logits > 0).astype(np.float32)

print("labels shape", labels.shape)

param_w = np.random.normal(loc=0, scale=0.01, size=(input_num, 1)).astype(np.float32)
param_b = np.zeros((1,)).astype(np.float32)


def data_iter(x, y, batch_size):
    length = len(x)
    indices = np.arange(length)
    np.random.shuffle(indices)
    for i in range(0, length, batch_size):
        indices_bs = indices[i:min(i + batch_size, length)]
        x_bs = x[indices_bs]
        y_bs = y[indices_bs]
        yield x_bs, y_bs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear_regression(x, w, b):
    # return np.matmul(x, w) + b
    return x @ w + b


def sgd(w, b, lr, bs, grad_w, grad_b):
    w -= lr * grad_w / bs
    b -= lr * grad_b / bs
    return w, b


def bce_loss(y_true, y_hat, ):
    y_true = y_true.reshape(-1, 1)
    epsilon = 1e-7
    y_hat = y_hat.clip(y_hat, epsilon, 1.0 - epsilon)
    return -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))


lr = 0.01
batch_size = 20
epochs = 30

losses = []
for epoch in range(epochs):
    for x, y in data_iter(features, labels, batch_size):
        y_hat = sigmoid(linear_regression(x, param_w, param_b))
        loss = bce_loss(y, y_hat)
        err = y_hat - y.reshape(-1, 1)
        grad_w = x.T @ err
        grad_b = err.sum(axis=0)
        param_w, param_b = sgd(param_w, param_b, lr=lr, bs=batch_size, grad_w=grad_w, grad_b=grad_b)

    train_hat = sigmoid(linear_regression(features, param_w, param_b))
    train_l = bce_loss(labels, train_hat).mean()
    losses.append(train_l)
    # IDE Can not infer, it's correct here.
    acc = np.mean((train_l > 0.5) == labels)
    print("epoch %d, train loss: %f, acc %f" % (epoch, train_l, acc))
