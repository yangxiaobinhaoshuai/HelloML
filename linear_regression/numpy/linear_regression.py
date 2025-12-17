import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np

num_input = 2
# num_examples = 1000
num_examples = 100

true_w = np.array([2.0, -3.4], dtype=np.float32)
true_b = 4.2

features = np.random.normal(loc=0, scale=1, size=(num_examples, num_input)).astype(np.float32)
labels = (true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b).astype(np.float32)

print("features shape: ", features.shape)
print("labels shape: ", labels.shape)

noise = np.random.normal(loc=0, scale=0.01, size=labels.shape).astype(np.float32)

print("noise shape: ", noise.shape)

# labels += noise


def use_svg_display():
    # 设置 matplotlib 的导出格式为 svg
    matplotlib_inline.backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(6, 4)):
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


# Only one dimension
set_figsize()
plt.scatter(x=features[:, 0], y=labels, s=10, label="features")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

