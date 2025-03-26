---
title: Introduction to Linear Regression - mathematics and application with Python
published: true
mathjax: true
---

<iframe src="https://github.com/sponsors/pabloinsente/card" title="Sponsor pabloinsente" height="225" width="600" style="border: 0;"></iframe>

Linear regression is among the most widely used tools in machine learning. Linear models are *linear* simply because the outputs are modeled as *linear combinations* of input vectors. Hence, we want to learn a function $f$ that describes with as little error as possible, the linear relationship between inputs and outputs. 

## Model definition

Consider a matrix of inputs $\textit{X} \in \mathbb{R}^{m\times n}$, a vector of weights $\bf{w} \in \mathbb{R}^n$, and output vector $\bf{y} \in \mathbb{R}$. We predict $\bf{y}$ given $\textit{X}$ as: 

$$
\hat{y} = \hat{w}_0 + \sum_{j=1}^nx_j\hat{w}_j = \hat{w}_0 + \hat{w}_1x_1 + \cdots + \hat{w}_nx_n
$$

Where $\hat{w_0}$ is the *bias* or *intercept*. Note we add a "hat" to the unknown estimated parameters to distinguish them from known given values. To express a linear regression in matrix notation, we can incorporate a constant vector $x_i=1$ to $\textit{X}$, and the bias $\hat{w_0}$ into the vector of weights $\hat{w}$, to obtain:

$$
\hat{y} = X^T\bf{\hat{w}}
$$

<img src="/assets/post-11/b-lin-regression.svg">

Note that in the absence of the bias term, our solution will be forced to pass through the origin of the coordinate space, forming a subspace. Adding the bias term allows our solution to be detached of such constrain, forming an affine set.

## Cost function

The goal of our model is to find a set of weights that minimizes some measure of error or cost function. The most popular measure of error is the *Sum of Squared Errors* (SSE), sometimes referred to as *Residual Sum of Squares* (RSS). The expression describing the SSE is: 

$$
SSE(w) = \sum_{i=1}^n(y_i - x_i^Tw)^2
$$

In words: we take the squared difference between the target and predicted value for the $i$ row of $\textit{X}$ and sum up the result. Again, we can express this in matrix notation as:

$$
SSE(w)= (y-\textit{X}w)^T(y-\textit{X}b)
$$

In machine learning is common to take the mean of the SSE to obtain the Mean Squared Error (MSE) as:

$$
MSE(w) = \frac{1}{n} \sum_{i=1}^n(y_i - x_i^Tw)^2
$$

## Model training

Given that the SSE is a quadratic function of the weights, the error surface is convex, hence always has a minimum. There are multiple ways to adjust the weights to minimize our measure of error. One way, know as *closed-form solution* or *normal equations*, is to solve for where the derivatives with respect to the weights are $0$:

$$
\Delta_wSSE=0
$$

Using the matrix notation, we solve for $w$ as:

$$
\begin{align}
(y-\textit{X}w)^T(y-\textit{X}b) &= 0 \\
(y^T-\textit{X}^Tw^T)(y-\textit{X}b) &= 0 \\
(w^T \textit{X}^T \textit{X}w - 2w^T\textit{X}y + y^Ty) &= 0 \\
2\textit{X}^T \textit{X} w- 2w^T\textit{X}y &= 0 \\
(\textit{X}^T \textit{X})^{-1} \textit{X}^T y &= w \\
\end{align}
$$

Note that the solution $(\textit{X}^T \textit{X})^{-1} \textit{X}^T y = w$ works only if $\textit{X}$ is *nonsingular* or *invertible* (see [here](https://en.wikipedia.org/wiki/Invertible_matrix)). Geometrically, this means that each vector in $\textit{X}$ is independent of each other. Otherwise, we can compute the *minimum norm solution* for the singular case (see [here](https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf)). If you are familiar with iterative methods, you can think of the closed-form solution as a one-step solution.

A different approach to solve a linear model with iterative methods like *gradient descent*. This method is preferred when the matrix is large, as inverting large matrices is computationally expensive. I won't describe gradient descent in this section (which you can review [here](https://www.youtube.com/watch?v=IHZwWFHWa-w)), to maintain our focus in the linear regression problem.

## Simple linear regression example

Let's try out a simple linear regression example with Python and sklearn. By simple, I mean one feature and one output. In the next section will do a multivariable or multi-feature regression.

We will load the *Boston house prices dataset* from sklearn. This dataset contains 506 rows (houses), 13 columns (houses features). The targets (house prices) range from 5 to 50. Our goal is just to show how to run a linear regression with sklearn, so we won't do an exploratory data analysis this time. A detailed description of the dataset attributes can be found [here](https://scikit-learn.org/stable/datasets/index.html#boston-dataset). 


```python
# Libraries for this section
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import altair as alt
import pandas as pd
alt.themes.enable('dark')
```




    ThemeRegistry.enable('dark')



We first load the dataset using sklearn API


```python
X, y = load_boston(return_X_y=True)
```


```python
print(f'Dataset shape: {X.shape}')
```

    Dataset shape: (506, 13)


We split our data into training and testing sets, using a 80/20 split.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
```


```python
print(f'Training data shape: {X_train.shape}')
print(f'Testing data shape: {X_test.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Testing labels shape: {y_test.shape}')
```

    Training data shape: (404, 13)
    Testing data shape: (102, 13)
    Training labels shape: (404,)
    Testing labels shape: (102,)


Since we want to run a regression with a single feature predictor, let's compute the correlation coefficients for each feature and the target (house prices). 


```python
corr = np.corrcoef(np.column_stack((X,y)).T)[:,13]
max_abs_corr = np.argmax(np.absolute(corr[0:13]))
print(f'correlation coefficient features and house prices: {np.round(corr, 2)}')
print(f'feature with max absolute correlation with house prices: {max_abs_corr}')
```

    correlation coefficient features and house prices: [-0.39  0.36 -0.48  0.18 -0.43  0.7  -0.38  0.25 -0.38 -0.47 -0.51  0.33
     -0.74  1.  ]
    feature with max absolute correlation with house prices: 12


Feature number 12 has the maximum absolute correlation with house prices, $-0.74$. According to the documentation, this is the  **% lower status of the population**: the more low-status people around, the lower the house price.

Now we fit the model using the training set.


```python
reg = linear_model.LinearRegression()
reg.fit(X_train[:, 12].reshape(-1, 1), y_train)  # pick all the rows for the 12 variable
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



The model has learned the coefficients or weights $w$ that best fit the data, which we can use to make predictions on the testing set.


```python
y_pred = reg.predict(X_test[:, 12].reshape(-1, 1)) # pick all the rows for the 12 variable
```

We evaluate the overall performance by computing the SSE, MSE, and the $R^2$ coefficient of determination. 


```python
SSE = ((y_test - y_pred) ** 2).sum() 
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
```


```python
print(f'Sum of Squared Errors: {np.round(SSE,2)}')
print(f'Mean of Squared Errors: {np.round(MSE,2)}')
print(f'R2 coefficient of determination: {np.round(R2,2)}')
```

    Sum of Squared Errors: 4726.3
    Mean of Squared Errors: 46.34
    R2 coefficient of determination: 0.43


Based on a single feature, we obtain a $SSE\approx4726.3$, a $MSE\approx46.34$, and a $R^2\approx0.43$.

Finally, let's visualize the regression line for this pair of variables.


```python
df = pd.DataFrame({'low-status-pop': X[:, 12], 'house-prices': y})

chart = alt.Chart(df).mark_point(color='fuchsia').encode(
    x='low-status-pop',
    y='house-prices')

chart + chart.transform_regression('low-status-pop', 'house-prices').mark_line(color='yellow')
```





<div id="altair-viz-9e02787248644f9eae4df3e50966b314"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-9e02787248644f9eae4df3e50966b314") {
      outputDiv = document.getElementById("altair-viz-9e02787248644f9eae4df3e50966b314");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"usermeta": {"embedOptions": {"theme": "dark"}}, "config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": {"type": "point", "color": "fuchsia"}, "encoding": {"x": {"type": "quantitative", "field": "low-status-pop"}, "y": {"type": "quantitative", "field": "house-prices"}}}, {"mark": {"type": "line", "color": "yellow"}, "encoding": {"x": {"type": "quantitative", "field": "low-status-pop"}, "y": {"type": "quantitative", "field": "house-prices"}}, "transform": [{"on": "low-status-pop", "regression": "house-prices"}]}], "data": {"name": "data-3c42feedf3ed9a0d516ec43eb701840b"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-3c42feedf3ed9a0d516ec43eb701840b": [{"low-status-pop": 4.98, "house-prices": 24.0}, {"low-status-pop": 9.14, "house-prices": 21.6}, {"low-status-pop": 4.03, "house-prices": 34.7}, {"low-status-pop": 2.94, "house-prices": 33.4}, {"low-status-pop": 5.33, "house-prices": 36.2}, {"low-status-pop": 5.21, "house-prices": 28.7}, {"low-status-pop": 12.43, "house-prices": 22.9}, {"low-status-pop": 19.15, "house-prices": 27.1}, {"low-status-pop": 29.93, "house-prices": 16.5}, {"low-status-pop": 17.1, "house-prices": 18.9}, {"low-status-pop": 20.45, "house-prices": 15.0}, {"low-status-pop": 13.27, "house-prices": 18.9}, {"low-status-pop": 15.71, "house-prices": 21.7}, {"low-status-pop": 8.26, "house-prices": 20.4}, {"low-status-pop": 10.26, "house-prices": 18.2}, {"low-status-pop": 8.47, "house-prices": 19.9}, {"low-status-pop": 6.58, "house-prices": 23.1}, {"low-status-pop": 14.67, "house-prices": 17.5}, {"low-status-pop": 11.69, "house-prices": 20.2}, {"low-status-pop": 11.28, "house-prices": 18.2}, {"low-status-pop": 21.02, "house-prices": 13.6}, {"low-status-pop": 13.83, "house-prices": 19.6}, {"low-status-pop": 18.72, "house-prices": 15.2}, {"low-status-pop": 19.88, "house-prices": 14.5}, {"low-status-pop": 16.3, "house-prices": 15.6}, {"low-status-pop": 16.51, "house-prices": 13.9}, {"low-status-pop": 14.81, "house-prices": 16.6}, {"low-status-pop": 17.28, "house-prices": 14.8}, {"low-status-pop": 12.8, "house-prices": 18.4}, {"low-status-pop": 11.98, "house-prices": 21.0}, {"low-status-pop": 22.6, "house-prices": 12.7}, {"low-status-pop": 13.04, "house-prices": 14.5}, {"low-status-pop": 27.71, "house-prices": 13.2}, {"low-status-pop": 18.35, "house-prices": 13.1}, {"low-status-pop": 20.34, "house-prices": 13.5}, {"low-status-pop": 9.68, "house-prices": 18.9}, {"low-status-pop": 11.41, "house-prices": 20.0}, {"low-status-pop": 8.77, "house-prices": 21.0}, {"low-status-pop": 10.13, "house-prices": 24.7}, {"low-status-pop": 4.32, "house-prices": 30.8}, {"low-status-pop": 1.98, "house-prices": 34.9}, {"low-status-pop": 4.84, "house-prices": 26.6}, {"low-status-pop": 5.81, "house-prices": 25.3}, {"low-status-pop": 7.44, "house-prices": 24.7}, {"low-status-pop": 9.55, "house-prices": 21.2}, {"low-status-pop": 10.21, "house-prices": 19.3}, {"low-status-pop": 14.15, "house-prices": 20.0}, {"low-status-pop": 18.8, "house-prices": 16.6}, {"low-status-pop": 30.81, "house-prices": 14.4}, {"low-status-pop": 16.2, "house-prices": 19.4}, {"low-status-pop": 13.45, "house-prices": 19.7}, {"low-status-pop": 9.43, "house-prices": 20.5}, {"low-status-pop": 5.28, "house-prices": 25.0}, {"low-status-pop": 8.43, "house-prices": 23.4}, {"low-status-pop": 14.8, "house-prices": 18.9}, {"low-status-pop": 4.81, "house-prices": 35.4}, {"low-status-pop": 5.77, "house-prices": 24.7}, {"low-status-pop": 3.95, "house-prices": 31.6}, {"low-status-pop": 6.86, "house-prices": 23.3}, {"low-status-pop": 9.22, "house-prices": 19.6}, {"low-status-pop": 13.15, "house-prices": 18.7}, {"low-status-pop": 14.44, "house-prices": 16.0}, {"low-status-pop": 6.73, "house-prices": 22.2}, {"low-status-pop": 9.5, "house-prices": 25.0}, {"low-status-pop": 8.05, "house-prices": 33.0}, {"low-status-pop": 4.67, "house-prices": 23.5}, {"low-status-pop": 10.24, "house-prices": 19.4}, {"low-status-pop": 8.1, "house-prices": 22.0}, {"low-status-pop": 13.09, "house-prices": 17.4}, {"low-status-pop": 8.79, "house-prices": 20.9}, {"low-status-pop": 6.72, "house-prices": 24.2}, {"low-status-pop": 9.88, "house-prices": 21.7}, {"low-status-pop": 5.52, "house-prices": 22.8}, {"low-status-pop": 7.54, "house-prices": 23.4}, {"low-status-pop": 6.78, "house-prices": 24.1}, {"low-status-pop": 8.94, "house-prices": 21.4}, {"low-status-pop": 11.97, "house-prices": 20.0}, {"low-status-pop": 10.27, "house-prices": 20.8}, {"low-status-pop": 12.34, "house-prices": 21.2}, {"low-status-pop": 9.1, "house-prices": 20.3}, {"low-status-pop": 5.29, "house-prices": 28.0}, {"low-status-pop": 7.22, "house-prices": 23.9}, {"low-status-pop": 6.72, "house-prices": 24.8}, {"low-status-pop": 7.51, "house-prices": 22.9}, {"low-status-pop": 9.62, "house-prices": 23.9}, {"low-status-pop": 6.53, "house-prices": 26.6}, {"low-status-pop": 12.86, "house-prices": 22.5}, {"low-status-pop": 8.44, "house-prices": 22.2}, {"low-status-pop": 5.5, "house-prices": 23.6}, {"low-status-pop": 5.7, "house-prices": 28.7}, {"low-status-pop": 8.81, "house-prices": 22.6}, {"low-status-pop": 8.2, "house-prices": 22.0}, {"low-status-pop": 8.16, "house-prices": 22.9}, {"low-status-pop": 6.21, "house-prices": 25.0}, {"low-status-pop": 10.59, "house-prices": 20.6}, {"low-status-pop": 6.65, "house-prices": 28.4}, {"low-status-pop": 11.34, "house-prices": 21.4}, {"low-status-pop": 4.21, "house-prices": 38.7}, {"low-status-pop": 3.57, "house-prices": 43.8}, {"low-status-pop": 6.19, "house-prices": 33.2}, {"low-status-pop": 9.42, "house-prices": 27.5}, {"low-status-pop": 7.67, "house-prices": 26.5}, {"low-status-pop": 10.63, "house-prices": 18.6}, {"low-status-pop": 13.44, "house-prices": 19.3}, {"low-status-pop": 12.33, "house-prices": 20.1}, {"low-status-pop": 16.47, "house-prices": 19.5}, {"low-status-pop": 18.66, "house-prices": 19.5}, {"low-status-pop": 14.09, "house-prices": 20.4}, {"low-status-pop": 12.27, "house-prices": 19.8}, {"low-status-pop": 15.55, "house-prices": 19.4}, {"low-status-pop": 13.0, "house-prices": 21.7}, {"low-status-pop": 10.16, "house-prices": 22.8}, {"low-status-pop": 16.21, "house-prices": 18.8}, {"low-status-pop": 17.09, "house-prices": 18.7}, {"low-status-pop": 10.45, "house-prices": 18.5}, {"low-status-pop": 15.76, "house-prices": 18.3}, {"low-status-pop": 12.04, "house-prices": 21.2}, {"low-status-pop": 10.3, "house-prices": 19.2}, {"low-status-pop": 15.37, "house-prices": 20.4}, {"low-status-pop": 13.61, "house-prices": 19.3}, {"low-status-pop": 14.37, "house-prices": 22.0}, {"low-status-pop": 14.27, "house-prices": 20.3}, {"low-status-pop": 17.93, "house-prices": 20.5}, {"low-status-pop": 25.41, "house-prices": 17.3}, {"low-status-pop": 17.58, "house-prices": 18.8}, {"low-status-pop": 14.81, "house-prices": 21.4}, {"low-status-pop": 27.26, "house-prices": 15.7}, {"low-status-pop": 17.19, "house-prices": 16.2}, {"low-status-pop": 15.39, "house-prices": 18.0}, {"low-status-pop": 18.34, "house-prices": 14.3}, {"low-status-pop": 12.6, "house-prices": 19.2}, {"low-status-pop": 12.26, "house-prices": 19.6}, {"low-status-pop": 11.12, "house-prices": 23.0}, {"low-status-pop": 15.03, "house-prices": 18.4}, {"low-status-pop": 17.31, "house-prices": 15.6}, {"low-status-pop": 16.96, "house-prices": 18.1}, {"low-status-pop": 16.9, "house-prices": 17.4}, {"low-status-pop": 14.59, "house-prices": 17.1}, {"low-status-pop": 21.32, "house-prices": 13.3}, {"low-status-pop": 18.46, "house-prices": 17.8}, {"low-status-pop": 24.16, "house-prices": 14.0}, {"low-status-pop": 34.41, "house-prices": 14.4}, {"low-status-pop": 26.82, "house-prices": 13.4}, {"low-status-pop": 26.42, "house-prices": 15.6}, {"low-status-pop": 29.29, "house-prices": 11.8}, {"low-status-pop": 27.8, "house-prices": 13.8}, {"low-status-pop": 16.65, "house-prices": 15.6}, {"low-status-pop": 29.53, "house-prices": 14.6}, {"low-status-pop": 28.32, "house-prices": 17.8}, {"low-status-pop": 21.45, "house-prices": 15.4}, {"low-status-pop": 14.1, "house-prices": 21.5}, {"low-status-pop": 13.28, "house-prices": 19.6}, {"low-status-pop": 12.12, "house-prices": 15.3}, {"low-status-pop": 15.79, "house-prices": 19.4}, {"low-status-pop": 15.12, "house-prices": 17.0}, {"low-status-pop": 15.02, "house-prices": 15.6}, {"low-status-pop": 16.14, "house-prices": 13.1}, {"low-status-pop": 4.59, "house-prices": 41.3}, {"low-status-pop": 6.43, "house-prices": 24.3}, {"low-status-pop": 7.39, "house-prices": 23.3}, {"low-status-pop": 5.5, "house-prices": 27.0}, {"low-status-pop": 1.73, "house-prices": 50.0}, {"low-status-pop": 1.92, "house-prices": 50.0}, {"low-status-pop": 3.32, "house-prices": 50.0}, {"low-status-pop": 11.64, "house-prices": 22.7}, {"low-status-pop": 9.81, "house-prices": 25.0}, {"low-status-pop": 3.7, "house-prices": 50.0}, {"low-status-pop": 12.14, "house-prices": 23.8}, {"low-status-pop": 11.1, "house-prices": 23.8}, {"low-status-pop": 11.32, "house-prices": 22.3}, {"low-status-pop": 14.43, "house-prices": 17.4}, {"low-status-pop": 12.03, "house-prices": 19.1}, {"low-status-pop": 14.69, "house-prices": 23.1}, {"low-status-pop": 9.04, "house-prices": 23.6}, {"low-status-pop": 9.64, "house-prices": 22.6}, {"low-status-pop": 5.33, "house-prices": 29.4}, {"low-status-pop": 10.11, "house-prices": 23.2}, {"low-status-pop": 6.29, "house-prices": 24.6}, {"low-status-pop": 6.92, "house-prices": 29.9}, {"low-status-pop": 5.04, "house-prices": 37.2}, {"low-status-pop": 7.56, "house-prices": 39.8}, {"low-status-pop": 9.45, "house-prices": 36.2}, {"low-status-pop": 4.82, "house-prices": 37.9}, {"low-status-pop": 5.68, "house-prices": 32.5}, {"low-status-pop": 13.98, "house-prices": 26.4}, {"low-status-pop": 13.15, "house-prices": 29.6}, {"low-status-pop": 4.45, "house-prices": 50.0}, {"low-status-pop": 6.68, "house-prices": 32.0}, {"low-status-pop": 4.56, "house-prices": 29.8}, {"low-status-pop": 5.39, "house-prices": 34.9}, {"low-status-pop": 5.1, "house-prices": 37.0}, {"low-status-pop": 4.69, "house-prices": 30.5}, {"low-status-pop": 2.87, "house-prices": 36.4}, {"low-status-pop": 5.03, "house-prices": 31.1}, {"low-status-pop": 4.38, "house-prices": 29.1}, {"low-status-pop": 2.97, "house-prices": 50.0}, {"low-status-pop": 4.08, "house-prices": 33.3}, {"low-status-pop": 8.61, "house-prices": 30.3}, {"low-status-pop": 6.62, "house-prices": 34.6}, {"low-status-pop": 4.56, "house-prices": 34.9}, {"low-status-pop": 4.45, "house-prices": 32.9}, {"low-status-pop": 7.43, "house-prices": 24.1}, {"low-status-pop": 3.11, "house-prices": 42.3}, {"low-status-pop": 3.81, "house-prices": 48.5}, {"low-status-pop": 2.88, "house-prices": 50.0}, {"low-status-pop": 10.87, "house-prices": 22.6}, {"low-status-pop": 10.97, "house-prices": 24.4}, {"low-status-pop": 18.06, "house-prices": 22.5}, {"low-status-pop": 14.66, "house-prices": 24.4}, {"low-status-pop": 23.09, "house-prices": 20.0}, {"low-status-pop": 17.27, "house-prices": 21.7}, {"low-status-pop": 23.98, "house-prices": 19.3}, {"low-status-pop": 16.03, "house-prices": 22.4}, {"low-status-pop": 9.38, "house-prices": 28.1}, {"low-status-pop": 29.55, "house-prices": 23.7}, {"low-status-pop": 9.47, "house-prices": 25.0}, {"low-status-pop": 13.51, "house-prices": 23.3}, {"low-status-pop": 9.69, "house-prices": 28.7}, {"low-status-pop": 17.92, "house-prices": 21.5}, {"low-status-pop": 10.5, "house-prices": 23.0}, {"low-status-pop": 9.71, "house-prices": 26.7}, {"low-status-pop": 21.46, "house-prices": 21.7}, {"low-status-pop": 9.93, "house-prices": 27.5}, {"low-status-pop": 7.6, "house-prices": 30.1}, {"low-status-pop": 4.14, "house-prices": 44.8}, {"low-status-pop": 4.63, "house-prices": 50.0}, {"low-status-pop": 3.13, "house-prices": 37.6}, {"low-status-pop": 6.36, "house-prices": 31.6}, {"low-status-pop": 3.92, "house-prices": 46.7}, {"low-status-pop": 3.76, "house-prices": 31.5}, {"low-status-pop": 11.65, "house-prices": 24.3}, {"low-status-pop": 5.25, "house-prices": 31.7}, {"low-status-pop": 2.47, "house-prices": 41.7}, {"low-status-pop": 3.95, "house-prices": 48.3}, {"low-status-pop": 8.05, "house-prices": 29.0}, {"low-status-pop": 10.88, "house-prices": 24.0}, {"low-status-pop": 9.54, "house-prices": 25.1}, {"low-status-pop": 4.73, "house-prices": 31.5}, {"low-status-pop": 6.36, "house-prices": 23.7}, {"low-status-pop": 7.37, "house-prices": 23.3}, {"low-status-pop": 11.38, "house-prices": 22.0}, {"low-status-pop": 12.4, "house-prices": 20.1}, {"low-status-pop": 11.22, "house-prices": 22.2}, {"low-status-pop": 5.19, "house-prices": 23.7}, {"low-status-pop": 12.5, "house-prices": 17.6}, {"low-status-pop": 18.46, "house-prices": 18.5}, {"low-status-pop": 9.16, "house-prices": 24.3}, {"low-status-pop": 10.15, "house-prices": 20.5}, {"low-status-pop": 9.52, "house-prices": 24.5}, {"low-status-pop": 6.56, "house-prices": 26.2}, {"low-status-pop": 5.9, "house-prices": 24.4}, {"low-status-pop": 3.59, "house-prices": 24.8}, {"low-status-pop": 3.53, "house-prices": 29.6}, {"low-status-pop": 3.54, "house-prices": 42.8}, {"low-status-pop": 6.57, "house-prices": 21.9}, {"low-status-pop": 9.25, "house-prices": 20.9}, {"low-status-pop": 3.11, "house-prices": 44.0}, {"low-status-pop": 5.12, "house-prices": 50.0}, {"low-status-pop": 7.79, "house-prices": 36.0}, {"low-status-pop": 6.9, "house-prices": 30.1}, {"low-status-pop": 9.59, "house-prices": 33.8}, {"low-status-pop": 7.26, "house-prices": 43.1}, {"low-status-pop": 5.91, "house-prices": 48.8}, {"low-status-pop": 11.25, "house-prices": 31.0}, {"low-status-pop": 8.1, "house-prices": 36.5}, {"low-status-pop": 10.45, "house-prices": 22.8}, {"low-status-pop": 14.79, "house-prices": 30.7}, {"low-status-pop": 7.44, "house-prices": 50.0}, {"low-status-pop": 3.16, "house-prices": 43.5}, {"low-status-pop": 13.65, "house-prices": 20.7}, {"low-status-pop": 13.0, "house-prices": 21.1}, {"low-status-pop": 6.59, "house-prices": 25.2}, {"low-status-pop": 7.73, "house-prices": 24.4}, {"low-status-pop": 6.58, "house-prices": 35.2}, {"low-status-pop": 3.53, "house-prices": 32.4}, {"low-status-pop": 2.98, "house-prices": 32.0}, {"low-status-pop": 6.05, "house-prices": 33.2}, {"low-status-pop": 4.16, "house-prices": 33.1}, {"low-status-pop": 7.19, "house-prices": 29.1}, {"low-status-pop": 4.85, "house-prices": 35.1}, {"low-status-pop": 3.76, "house-prices": 45.4}, {"low-status-pop": 4.59, "house-prices": 35.4}, {"low-status-pop": 3.01, "house-prices": 46.0}, {"low-status-pop": 3.16, "house-prices": 50.0}, {"low-status-pop": 7.85, "house-prices": 32.2}, {"low-status-pop": 8.23, "house-prices": 22.0}, {"low-status-pop": 12.93, "house-prices": 20.1}, {"low-status-pop": 7.14, "house-prices": 23.2}, {"low-status-pop": 7.6, "house-prices": 22.3}, {"low-status-pop": 9.51, "house-prices": 24.8}, {"low-status-pop": 3.33, "house-prices": 28.5}, {"low-status-pop": 3.56, "house-prices": 37.3}, {"low-status-pop": 4.7, "house-prices": 27.9}, {"low-status-pop": 8.58, "house-prices": 23.9}, {"low-status-pop": 10.4, "house-prices": 21.7}, {"low-status-pop": 6.27, "house-prices": 28.6}, {"low-status-pop": 7.39, "house-prices": 27.1}, {"low-status-pop": 15.84, "house-prices": 20.3}, {"low-status-pop": 4.97, "house-prices": 22.5}, {"low-status-pop": 4.74, "house-prices": 29.0}, {"low-status-pop": 6.07, "house-prices": 24.8}, {"low-status-pop": 9.5, "house-prices": 22.0}, {"low-status-pop": 8.67, "house-prices": 26.4}, {"low-status-pop": 4.86, "house-prices": 33.1}, {"low-status-pop": 6.93, "house-prices": 36.1}, {"low-status-pop": 8.93, "house-prices": 28.4}, {"low-status-pop": 6.47, "house-prices": 33.4}, {"low-status-pop": 7.53, "house-prices": 28.2}, {"low-status-pop": 4.54, "house-prices": 22.8}, {"low-status-pop": 9.97, "house-prices": 20.3}, {"low-status-pop": 12.64, "house-prices": 16.1}, {"low-status-pop": 5.98, "house-prices": 22.1}, {"low-status-pop": 11.72, "house-prices": 19.4}, {"low-status-pop": 7.9, "house-prices": 21.6}, {"low-status-pop": 9.28, "house-prices": 23.8}, {"low-status-pop": 11.5, "house-prices": 16.2}, {"low-status-pop": 18.33, "house-prices": 17.8}, {"low-status-pop": 15.94, "house-prices": 19.8}, {"low-status-pop": 10.36, "house-prices": 23.1}, {"low-status-pop": 12.73, "house-prices": 21.0}, {"low-status-pop": 7.2, "house-prices": 23.8}, {"low-status-pop": 6.87, "house-prices": 23.1}, {"low-status-pop": 7.7, "house-prices": 20.4}, {"low-status-pop": 11.74, "house-prices": 18.5}, {"low-status-pop": 6.12, "house-prices": 25.0}, {"low-status-pop": 5.08, "house-prices": 24.6}, {"low-status-pop": 6.15, "house-prices": 23.0}, {"low-status-pop": 12.79, "house-prices": 22.2}, {"low-status-pop": 9.97, "house-prices": 19.3}, {"low-status-pop": 7.34, "house-prices": 22.6}, {"low-status-pop": 9.09, "house-prices": 19.8}, {"low-status-pop": 12.43, "house-prices": 17.1}, {"low-status-pop": 7.83, "house-prices": 19.4}, {"low-status-pop": 5.68, "house-prices": 22.2}, {"low-status-pop": 6.75, "house-prices": 20.7}, {"low-status-pop": 8.01, "house-prices": 21.1}, {"low-status-pop": 9.8, "house-prices": 19.5}, {"low-status-pop": 10.56, "house-prices": 18.5}, {"low-status-pop": 8.51, "house-prices": 20.6}, {"low-status-pop": 9.74, "house-prices": 19.0}, {"low-status-pop": 9.29, "house-prices": 18.7}, {"low-status-pop": 5.49, "house-prices": 32.7}, {"low-status-pop": 8.65, "house-prices": 16.5}, {"low-status-pop": 7.18, "house-prices": 23.9}, {"low-status-pop": 4.61, "house-prices": 31.2}, {"low-status-pop": 10.53, "house-prices": 17.5}, {"low-status-pop": 12.67, "house-prices": 17.2}, {"low-status-pop": 6.36, "house-prices": 23.1}, {"low-status-pop": 5.99, "house-prices": 24.5}, {"low-status-pop": 5.89, "house-prices": 26.6}, {"low-status-pop": 5.98, "house-prices": 22.9}, {"low-status-pop": 5.49, "house-prices": 24.1}, {"low-status-pop": 7.79, "house-prices": 18.6}, {"low-status-pop": 4.5, "house-prices": 30.1}, {"low-status-pop": 8.05, "house-prices": 18.2}, {"low-status-pop": 5.57, "house-prices": 20.6}, {"low-status-pop": 17.6, "house-prices": 17.8}, {"low-status-pop": 13.27, "house-prices": 21.7}, {"low-status-pop": 11.48, "house-prices": 22.7}, {"low-status-pop": 12.67, "house-prices": 22.6}, {"low-status-pop": 7.79, "house-prices": 25.0}, {"low-status-pop": 14.19, "house-prices": 19.9}, {"low-status-pop": 10.19, "house-prices": 20.8}, {"low-status-pop": 14.64, "house-prices": 16.8}, {"low-status-pop": 5.29, "house-prices": 21.9}, {"low-status-pop": 7.12, "house-prices": 27.5}, {"low-status-pop": 14.0, "house-prices": 21.9}, {"low-status-pop": 13.33, "house-prices": 23.1}, {"low-status-pop": 3.26, "house-prices": 50.0}, {"low-status-pop": 3.73, "house-prices": 50.0}, {"low-status-pop": 2.96, "house-prices": 50.0}, {"low-status-pop": 9.53, "house-prices": 50.0}, {"low-status-pop": 8.88, "house-prices": 50.0}, {"low-status-pop": 34.77, "house-prices": 13.8}, {"low-status-pop": 37.97, "house-prices": 13.8}, {"low-status-pop": 13.44, "house-prices": 15.0}, {"low-status-pop": 23.24, "house-prices": 13.9}, {"low-status-pop": 21.24, "house-prices": 13.3}, {"low-status-pop": 23.69, "house-prices": 13.1}, {"low-status-pop": 21.78, "house-prices": 10.2}, {"low-status-pop": 17.21, "house-prices": 10.4}, {"low-status-pop": 21.08, "house-prices": 10.9}, {"low-status-pop": 23.6, "house-prices": 11.3}, {"low-status-pop": 24.56, "house-prices": 12.3}, {"low-status-pop": 30.63, "house-prices": 8.8}, {"low-status-pop": 30.81, "house-prices": 7.2}, {"low-status-pop": 28.28, "house-prices": 10.5}, {"low-status-pop": 31.99, "house-prices": 7.4}, {"low-status-pop": 30.62, "house-prices": 10.2}, {"low-status-pop": 20.85, "house-prices": 11.5}, {"low-status-pop": 17.11, "house-prices": 15.1}, {"low-status-pop": 18.76, "house-prices": 23.2}, {"low-status-pop": 25.68, "house-prices": 9.7}, {"low-status-pop": 15.17, "house-prices": 13.8}, {"low-status-pop": 16.35, "house-prices": 12.7}, {"low-status-pop": 17.12, "house-prices": 13.1}, {"low-status-pop": 19.37, "house-prices": 12.5}, {"low-status-pop": 19.92, "house-prices": 8.5}, {"low-status-pop": 30.59, "house-prices": 5.0}, {"low-status-pop": 29.97, "house-prices": 6.3}, {"low-status-pop": 26.77, "house-prices": 5.6}, {"low-status-pop": 20.32, "house-prices": 7.2}, {"low-status-pop": 20.31, "house-prices": 12.1}, {"low-status-pop": 19.77, "house-prices": 8.3}, {"low-status-pop": 27.38, "house-prices": 8.5}, {"low-status-pop": 22.98, "house-prices": 5.0}, {"low-status-pop": 23.34, "house-prices": 11.9}, {"low-status-pop": 12.13, "house-prices": 27.9}, {"low-status-pop": 26.4, "house-prices": 17.2}, {"low-status-pop": 19.78, "house-prices": 27.5}, {"low-status-pop": 10.11, "house-prices": 15.0}, {"low-status-pop": 21.22, "house-prices": 17.2}, {"low-status-pop": 34.37, "house-prices": 17.9}, {"low-status-pop": 20.08, "house-prices": 16.3}, {"low-status-pop": 36.98, "house-prices": 7.0}, {"low-status-pop": 29.05, "house-prices": 7.2}, {"low-status-pop": 25.79, "house-prices": 7.5}, {"low-status-pop": 26.64, "house-prices": 10.4}, {"low-status-pop": 20.62, "house-prices": 8.8}, {"low-status-pop": 22.74, "house-prices": 8.4}, {"low-status-pop": 15.02, "house-prices": 16.7}, {"low-status-pop": 15.7, "house-prices": 14.2}, {"low-status-pop": 14.1, "house-prices": 20.8}, {"low-status-pop": 23.29, "house-prices": 13.4}, {"low-status-pop": 17.16, "house-prices": 11.7}, {"low-status-pop": 24.39, "house-prices": 8.3}, {"low-status-pop": 15.69, "house-prices": 10.2}, {"low-status-pop": 14.52, "house-prices": 10.9}, {"low-status-pop": 21.52, "house-prices": 11.0}, {"low-status-pop": 24.08, "house-prices": 9.5}, {"low-status-pop": 17.64, "house-prices": 14.5}, {"low-status-pop": 19.69, "house-prices": 14.1}, {"low-status-pop": 12.03, "house-prices": 16.1}, {"low-status-pop": 16.22, "house-prices": 14.3}, {"low-status-pop": 15.17, "house-prices": 11.7}, {"low-status-pop": 23.27, "house-prices": 13.4}, {"low-status-pop": 18.05, "house-prices": 9.6}, {"low-status-pop": 26.45, "house-prices": 8.7}, {"low-status-pop": 34.02, "house-prices": 8.4}, {"low-status-pop": 22.88, "house-prices": 12.8}, {"low-status-pop": 22.11, "house-prices": 10.5}, {"low-status-pop": 19.52, "house-prices": 17.1}, {"low-status-pop": 16.59, "house-prices": 18.4}, {"low-status-pop": 18.85, "house-prices": 15.4}, {"low-status-pop": 23.79, "house-prices": 10.8}, {"low-status-pop": 23.98, "house-prices": 11.8}, {"low-status-pop": 17.79, "house-prices": 14.9}, {"low-status-pop": 16.44, "house-prices": 12.6}, {"low-status-pop": 18.13, "house-prices": 14.1}, {"low-status-pop": 19.31, "house-prices": 13.0}, {"low-status-pop": 17.44, "house-prices": 13.4}, {"low-status-pop": 17.73, "house-prices": 15.2}, {"low-status-pop": 17.27, "house-prices": 16.1}, {"low-status-pop": 16.74, "house-prices": 17.8}, {"low-status-pop": 18.71, "house-prices": 14.9}, {"low-status-pop": 18.13, "house-prices": 14.1}, {"low-status-pop": 19.01, "house-prices": 12.7}, {"low-status-pop": 16.94, "house-prices": 13.5}, {"low-status-pop": 16.23, "house-prices": 14.9}, {"low-status-pop": 14.7, "house-prices": 20.0}, {"low-status-pop": 16.42, "house-prices": 16.4}, {"low-status-pop": 14.65, "house-prices": 17.7}, {"low-status-pop": 13.99, "house-prices": 19.5}, {"low-status-pop": 10.29, "house-prices": 20.2}, {"low-status-pop": 13.22, "house-prices": 21.4}, {"low-status-pop": 14.13, "house-prices": 19.9}, {"low-status-pop": 17.15, "house-prices": 19.0}, {"low-status-pop": 21.32, "house-prices": 19.1}, {"low-status-pop": 18.13, "house-prices": 19.1}, {"low-status-pop": 14.76, "house-prices": 20.1}, {"low-status-pop": 16.29, "house-prices": 19.9}, {"low-status-pop": 12.87, "house-prices": 19.6}, {"low-status-pop": 14.36, "house-prices": 23.2}, {"low-status-pop": 11.66, "house-prices": 29.8}, {"low-status-pop": 18.14, "house-prices": 13.8}, {"low-status-pop": 24.1, "house-prices": 13.3}, {"low-status-pop": 18.68, "house-prices": 16.7}, {"low-status-pop": 24.91, "house-prices": 12.0}, {"low-status-pop": 18.03, "house-prices": 14.6}, {"low-status-pop": 13.11, "house-prices": 21.4}, {"low-status-pop": 10.74, "house-prices": 23.0}, {"low-status-pop": 7.74, "house-prices": 23.7}, {"low-status-pop": 7.01, "house-prices": 25.0}, {"low-status-pop": 10.42, "house-prices": 21.8}, {"low-status-pop": 13.34, "house-prices": 20.6}, {"low-status-pop": 10.58, "house-prices": 21.2}, {"low-status-pop": 14.98, "house-prices": 19.1}, {"low-status-pop": 11.45, "house-prices": 20.6}, {"low-status-pop": 18.06, "house-prices": 15.2}, {"low-status-pop": 23.97, "house-prices": 7.0}, {"low-status-pop": 29.68, "house-prices": 8.1}, {"low-status-pop": 18.07, "house-prices": 13.6}, {"low-status-pop": 13.35, "house-prices": 20.1}, {"low-status-pop": 12.01, "house-prices": 21.8}, {"low-status-pop": 13.59, "house-prices": 24.5}, {"low-status-pop": 17.6, "house-prices": 23.1}, {"low-status-pop": 21.14, "house-prices": 19.7}, {"low-status-pop": 14.1, "house-prices": 18.3}, {"low-status-pop": 12.92, "house-prices": 21.2}, {"low-status-pop": 15.1, "house-prices": 17.5}, {"low-status-pop": 14.33, "house-prices": 16.8}, {"low-status-pop": 9.67, "house-prices": 22.4}, {"low-status-pop": 9.08, "house-prices": 20.6}, {"low-status-pop": 5.64, "house-prices": 23.9}, {"low-status-pop": 6.48, "house-prices": 22.0}, {"low-status-pop": 7.88, "house-prices": 11.9}]}}, {"mode": "vega-lite"});
</script>



## Multivariable linear regression example

Now let's fit a model with all 13 features as predictors. For this we just need to remove `[:, 12].reshape(-1, 1)` from the `fit` and `predict` methods.


```python
multi_reg = linear_model.LinearRegression()
multi_reg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
y_pred_multi = multi_reg.predict(X_test) 
```

Again, let's evaluate the overall performance by computing the SSE, MSE, and the $R^2$ coefficient of determination. 


```python
SSE_multi = ((y_test - y_pred_multi) ** 2).sum() 
MSE_multi = mean_squared_error(y_test, y_pred_multi)
R2_multi = r2_score(y_test, y_pred_multi)
```


```python
print(f'Sum of Squared Errors multivariable regression: {np.round(SSE_multi,2)}')
print(f'Mean of Squared Errors multivariable regression: {np.round(MSE_multi,2)}')
print(f'R2 coefficient of determination multivariable regression: {np.round(R2_multi,2)}')
```

    Sum of Squared Errors multivariable regression: 3411.8
    Mean of Squared Errors multivariable regression: 33.45
    R2 coefficient of determination multivariable regression: 0.59


Based on the 13 features, we obtain a $SSE\approx3411.8$, a $MSE\approx33.45$, and a $R^2\approx0.59$. As expected, the error measures went down and the association measure went up, as more features provide more information for prediction.

Visualization is not possible for a 13 features regression, but you can make your best effort by imaging a 3D space and thinking "13!" with all your might.

## References

- Bishop, C. M. (2006). 3. Linear models for regression. Pattern recognition and machine learning. springer.

- Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020) 9. Linear Regression. In Mathematics for machine learning. Cambridge University Press.

- Friedman, J., Hastie, T., & Tibshirani, R. (2009). 3. Linear Methods for Regression. In The elements of statistical learning (Vol. 1, No. 10). New York: Springer series in statistics.