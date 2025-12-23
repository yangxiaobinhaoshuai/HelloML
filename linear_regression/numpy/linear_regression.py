import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np

#  1. Data generation
num_input = 2
num_examples = 1000

true_w = np.array([2.0, -3.4], dtype=np.float32)
true_b = 4.2

features = np.random.normal(loc=0, scale=1, size=(num_examples, num_input)).astype(np.float32)
labels = (true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b).astype(np.float32)

print("features shape: ", features.shape)
print("labels shape: ", labels.shape)

noise = np.random.normal(loc=0, scale=0.01, size=labels.shape).astype(np.float32)

print("noise shape: ", noise.shape)

labels += noise

def use_svg_display():
    # 设置 matplotlib 的导出格式为 svg
    matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

def set_figsize(figsize=(6, 4)):
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


# 2. Draw scatter points

# Only one dimension，does NOT show the full linear relationship.
set_figsize()
plt.scatter(x=features[:, 0], y=labels, s=10, label="features")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# 3. data iterations
def data_iter(batch_size, features, labels):
    nums = len(features)
    indices = list(range(nums))
    np.random.shuffle(indices)
    for i in range(0, nums, batch_size):
        batch_idx = indices[i:min(i + batch_size, nums)]
        features_batch = features[batch_idx]
        labels_batch = labels[batch_idx]
        yield features_batch, labels_batch


# 4. Params initialization
w = np.random.normal(loc=0, scale=0.01, size=(num_input, 1)).astype(np.float32)
b = np.zeros((1,), dtype=np.float32)


# 5. model
def linear_reg(X, w, b):
    return np.dot(X, w) + b


# 6. loss
def squared_loss(y_true, y_hat):
    y_true = y_true.reshape(-1, 1)
    return (y_true - y_hat) ** 2 / 2


# 7. optimizer
def sgd(w, b, grad_w, grad_b, lr, batch_size):
    w -= lr * grad_w / batch_size
    b -= lr * grad_b / batch_size
    return w, b


# 8.training
lr = 0.01
batch_size = 20
epochs = 30

losses = []
for epoch in range(epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linear_reg(X, w, b)
        loss = squared_loss(y, y_hat)
        err = y_hat - y.reshape(-1, 1)
        grad_w = X.T @ err
        grad_b = err.sum(axis=0)

        # For last batch
        cur_bs = len(X)
        w, b = sgd(w, b, grad_w, grad_b, lr, cur_bs)
    train_l = squared_loss(labels, linear_reg(features, w, b)).mean()
    losses.append(train_l)
    print("epoch:", epoch + 1, f"train_l: {train_l:.6f}")

print("train_w", true_w, " learn w:", w)
print("train_b", true_b, " learn b:", b)


# 8. Draw loss curve
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss over Epochs (Numpy)")
plt.legend()
plt.show()
