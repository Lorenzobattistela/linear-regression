from helpers import *
import pandas as pd
import matplotlib.pyplot as plt

test_data = pd.read_csv("./test.csv")
test_data.dropna(inplace=True)
y = list(test_data["y"])
x = list(test_data["x"])

# testing calculating linear regression model 
# by hand

xs = [1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83]
ys =[52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46]

assert len(xs) == len(ys), "Length of vectors should be equal"

n = len(xs)
sxy = sum_xs_ys(xs, ys)
sx = sum_xs(xs)
sy = sum_ys(ys)
sx_squared = sum_of_xs_squared(xs)

# This is least squares fit function done by hand
beta = ((n * sxy) - (sx * sy)) / ((n * sx_squared) - (sx ** 2))

assert beta == 61.272186542107434, "beta value dont match"

alpha = (sy / n) - (beta * (sx / n))

assert alpha == -39.06195591883866, "alpha values dont match"

alpha_i, beta_i = least_squares_fit(xs, ys)


assert -39.07 < alpha_i < 39.05, "alpha value should be -39.06"
assert 61.26 < beta_i < 61.28, "beta value should be 61.27"

# the equation ends up being: yi = -39.06 + 61.27 * x + error

alpha, beta = least_squares_fit(x, y)
predictions = [predict(alpha, beta, x_i) for x_i in x]

ax = plt.axes()
ax.scatter(x, y, c="#fc5a8d")
ax.plot(x, predictions)
ax.set_xlabel("x")
ax.axis("tight")
plt.show()
