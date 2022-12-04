# Linear regression 

This repository aims to understand the math behind linear regression and build all the formulas needed to calculate it.

First of all, I import some test data provided by [this kaggle dataset.](https://www.kaggle.com/datasets/andonians/random-linear-regression?select=train.csv)

Then, I utilize some numbers provided by Wikipedia on Simple linear regretion article [(see references)](#references) and do the calculations by hand, and check the results.

On `helpers.py`, math functions built by hand too are used to do this process in the entire dataset, and results are compared to the calculation done by hand.

Then, I utilize my functions to make predictions about the kaggle dataset and plot a graph where the line represents our model predictions and the dots represent the correct data.

This model utilizes Linear Least Squares for fitting and sum of squared errors.

# Notes

- Linear approach for modelling the relationship between a scalar response and one or more explanatory variables (dependent and independent).

- Scalar is an element used to define a ************************vector space************************. One explanatory variable = simple linear regression, more = multiple linear regression

- Relationship is modeled using linear predictor functions, model parameters are estimated form data. (linear models).

- Linear regression focuses on conditional probability distribution (given two jointly distributed random variable X and Y, conditional probability distribution of Y given X is the probability distribution of Y when X is known its value.) of the response given the values of the predictors.

- Given a dataset, a lineal regression model assumes that the relationship between the dependent variable y and the p-vector (A ***k*-vector** is such a linear combination that is *homogeneous* of degree *k* (all terms are *k*-blades for the same *k - ex: if k=1, all vectors would be simple. If k=2, all vectors would be bivector. If k=0, it would be a scalar number*) of regressors x is linear.

- The relationship is modeled through a disturbance term or error variable - a random variable that adds “noise” to the linear relationship between dependent variable and regressors.



# References

https://en.wikipedia.org/wiki/Simple_linear_regression

https://en.wikipedia.org/wiki/Linear_regression

https://en.wikipedia.org/wiki/Linear_least_squares

https://www.nature.com/articles/nmeth.3627