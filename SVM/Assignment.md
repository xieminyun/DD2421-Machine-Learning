#### Assignment 1

**Move the clusters around and change their sizes to make it easier or harder for the classifier to find a decent boundary. Pay attention to when the optimizer (minimize function) is not able to find a solution at all.**

```python
ret = minimize(objective, start, bounds=B, constraints=XC)
if(not ret['success']):
    raise ValueError("Can't find a soluition")
    
# Use the following dataset
classA = np.concatenate((
    np.random.randn(10, 2) * 0.3 + [1.4, 0],
    np.random.randn(10, 2) * 0.3 + [-1.4, 0],
 ))
classB = np.random.randn(20, 2) * 0.5 + [0.0, -0.5]
```

Sometimes

<img src="D:\DD2421-Machine-Learning\SVM\assets\1676808368457.png" alt="1676808368457" style="zoom:67%;" />

Sometimes

![image-20230219124941625](C:\Users\Lynn\AppData\Roaming\Typora\typora-user-images\image-20230219124941625.png)

We can see in the first figure, the red liner boundary can mainly classify the red points correctly, and I think if it can't, then the optimizer (minimize function) is not able to find a solution at all.

If I change 1.4 to bigger numbers like 1.6, which means the cluster is more centralized, it's easier to find the boundary. If I change 1.4 to smaller numbers like 1.2, which means the cluster is more scattered, it's harder to find the boundary.

#### Assignment 2

Linear Kernel

![1677156426515](D:\DD2421-Machine-Learning\SVM\assets\1677156426515.png)

Polynomial kernels

![1677156585856](D:\DD2421-Machine-Learning\SVM\assets\1677156585856.png)

Radial Basis Function (RBF) kernels

![1677156647170](D:\DD2421-Machine-Learning\SVM\assets\1677156647170.png)

**Implement the two non-linear kernels. You should be able to classify very hard data sets with these.**

```python
def linear_kernel(x1, x2):
	return np.dot(x1, x2)
    
def polynomial_kernel(x1, x2, p = 5):
	return np.power((np.dot(x1, x2) + 1), p)

def RBF_kernel(x1, x2, sigma = 2):
	diff = np.subtract(x1, x2)
	return np.exp(- np.dot(diff, diff) / (2 * (sigma ** 2)))

classA = np.concatenate((
    np.random.randn(10, 2) * 0.4 + [1.4, 0],
    np.random.randn(5, 2) * 0.4 + [-1.4, 0],
    np.random.randn(5, 2) * 0.3 + [0, -2],
 ))
classB = np.random.randn(20, 2) * 0.5 + [0.0, -0.5]
```

Use polynomial kernel

<img src="D:\DD2421-Machine-Learning\SVM\assets\1676810521894.png" alt="1676810521894" style="zoom:67%;" />

Use RBF kernel

<img src="D:\DD2421-Machine-Learning\SVM\assets\image-20230219134331892.png" alt="image-20230219134331892" style="zoom:67%;" />

#### Assignment 3

**The non-linear kernels have parameters; explore how they influence the decision boundary. Reason about this in terms of the bias variance trade-off.**

```python
classA = np.concatenate((
    np.random.randn(10, 2) * 0.1 + [-0.5, 0],
    np.random.randn(10, 2) * 0.1 + [0.5, 0],
 ))
classB = np.random.randn(20, 2) * 0.1 + [0.0, -0.8]
```

For polynomial kernel

Initial p = 2

<img src="D:\DD2421-Machine-Learning\SVM\assets\1676817142147.png" alt="1676817142147" style="zoom:67%;" />

p = 6

<img src="D:\DD2421-Machine-Learning\SVM\assets\1676817224367.png" alt="1676817224367" style="zoom:67%;" />

As p increases, complexity will increase with a higher-order regression polynomial (p). Thus, variance increases but bias decreases, and this makes the boundary steeper.

For RBF kernel

sigma = 2

<img src="D:\DD2421-Machine-Learning\SVM\assets\image-20230219153649897.png" alt="image-20230219153649897" style="zoom:67%;" />

sigma = 5

<img src="D:\DD2421-Machine-Learning\SVM\assets\1676817442669.png" alt="1676817442669" style="zoom: 67%;" />

As sigma increases, the exponent is smaller, which means the complexity of the model decreases,  the variance decreases and the bias increases. And this makes the boundary smoother.

#### Assignment 4

**Explore the role of the slack parameter C. What happens for very large/small values?**

```python
classA = np.concatenate((
    np.random.randn(10, 2) * 0.3 + [1.5, 0],
    np.random.randn(10, 2) * 0.3 + [-1.5, 0],
 ))
classB = np.random.randn(20, 2) * 0.3 + [0.0, -0.5]
```

C = 10

<img src="D:\DD2421-Machine-Learning\SVM\assets\image-20230219154329090.png" alt="image-20230219154329090" style="zoom:67%;" />

C = 10000

<img src="D:\DD2421-Machine-Learning\SVM\assets\image-20230219154409250.png" alt="image-20230219154409250" style="zoom:67%;" />

Large C means low slack, and there's less tolerance for outliers. And less error is allowed.

Small C means high slack, and there's more tolerance for outliers. And more error is allowed.

This is because the penalty term of the cost function

<img src="D:\DD2421-Machine-Learning\SVM\assets\1676818091019.png" alt="1676818091019" style="zoom:67%;" />

The smaller the C is  , the larger the epsilon is.

<img src="D:\DD2421-Machine-Learning\SVM\assets\image-20230219155141174.png" alt="image-20230219155141174" style="zoom:67%;" />

And for large epsilon, it's easier to satisfy the expression and that means more tolerance.

To prevent overfitting and ensure the accuracy of the model, we need to debate the balance between small C and large C.

#### Assignment 5

**Imagine that you are given data that is not easily separable. When should you opt for more slack rather than going for a more complex model (kernel) and vice versa?**

In general, the choice between more slack or a more complex model depends on the balance between model complexity and model performance. 

If the data is easily separable and a simple linear model with a small slack variable is not able to achieve satisfactory performance, then increasing the slack variable may be a better choice. This allows for more flexibility in the decision boundary and can improve performance on the training data and potentially on new data. However, increasing the slack variable too much can lead to a model that is too simple and underfits the data.

On the other hand, if the data is complex and more features or a higher degree kernel are necessary to improve the model's ability to capture the underlying structure of the data and potentially improve performance, then a more complex model may be a better choice. However, it is important to monitor the model's performance on a validation set and avoid overfitting.