import matplotlib.pyplot as plt
import numpy as np

input_num = 10
feature_num = 1000

arr = [3, 7, 9.8, 4, 6, 2, 4.7, 8.1, 7.3, 2]
print(len(arr))
true_w = np.array(arr, dtype=np.float32).reshape(-1, 1)
print("true_w, shape", true_w.shape)
# true_b = 7.4
# For balance feature points better.
true_b = 0.0

features = np.random.normal(loc=0, scale=1, size=(feature_num, input_num)).astype(np.float32)
logits = features @ true_w + true_b

noise = np.random.normal(loc=0, scale=0.01, size=(feature_num, 1)).astype(np.float32)
logits += noise

labels = (logits > 0).astype(np.float32)

print("labels shape", labels.shape)

projection = features @ true_w

plt.figure()
plt.hist(projection[labels[:, 0] == 0], bins=50, alpha=0.6, label="Class 0")
plt.hist(projection[labels[:, 0] == 1], bins=50, alpha=0.6, label="Class 1")
plt.axvline(x=- true_b, color="red", linestyle="--", label="Decision boundary")
plt.legend()
plt.title("Projection on true_w direction")
plt.show()

# sys.exit()

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


def linear(x, w, b):
    # return np.matmul(x, w) + b
    return x @ w + b


def sgd(w, b, lr, grad_w, grad_b):
    w -= lr * grad_w
    b -= lr * grad_b
    return w, b


def bce_loss(y_true, y_hat, ):
    y_true = y_true.reshape(-1, 1)
    epsilon = 1e-7
    y_hat = y_hat.clip(epsilon, 1.0 - epsilon)
    return -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))


lr = 0.1
batch_size = 20
epochs = 30

losses = []
accs = []
for epoch in range(epochs):
    for x, y in data_iter(features, labels, batch_size):
        y_hat = sigmoid(linear(x, param_w, param_b))
        err = y_hat - y.reshape(-1, 1)
        grad_w = x.T @ err / x.shape[0]
        grad_b = err.mean()
        param_w, param_b = sgd(param_w, param_b, lr=lr, grad_w=grad_w, grad_b=grad_b)

    train_hat = sigmoid(linear(features, param_w, param_b))
    train_l = bce_loss(labels, train_hat)
    train_l_mean = train_l.mean()
    losses.append(float(train_l_mean))
    acc = np.mean((train_hat > 0.5) == labels).astype(np.float32)
    accs.append(float(acc))
    # print("epoch %d, train loss: %f, acc %f" % (epoch+1, train_l, acc))
    print(f"epoch {epoch + 1}, train loss {train_l_mean:.4f}, train acc {acc:.4f}")

print("true_w", true_w, " learned w:", param_w)
print("true_b", true_b, "learned b:", param_b)

plt.plot(losses, label="Train BCE Loss")
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.title("Logistic Regression (Numpy) - Loss")
plt.legend()
plt.show()

plt.plot(accs, label="Train Accuracy")
plt.xlabel("Epoches")
plt.ylabel("Train Accuracy")
plt.title("Logistic Regression (Numpy) - Train Accuracy")
plt.legend()
plt.show()
