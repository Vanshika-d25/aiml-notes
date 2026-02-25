Regression
Predicting a continuous numerical value like house price, salary, temperature.
Types:
a. Linear Family
1. Simple Linear Regression
2. Multiple linear regression
3. Polynomial Regression
b. Regularized Regression
4. Ridge Regression- used to overcome Overfitting, Multicollinearity
5. Lasso Regression- used to overcome overfitting and feature selection
6. Elastic net
c. Regression based on error handling
   7. Ordinary least square
   8. Robust regression
d. based on output distribution
9. Poisson Regression
10. Negative binomials


## ***Regression*** 

### Linear Regression
-ML technique to model the relationship between independent(X) and dependent variable(Y), assumed to be on a straight line.
-Y=mX+ b + e
Where:
- **m → Slope (coefficient)** rate of change
- **b → Intercept** baseline value
- **X → Input variable** 
- **Y → Predicted output**
- **e-> epsilon (error/noise)**
#### *-Cost Function:*

We need a way to measure total error.
Linear regression uses:
<img width="428" height="109" alt="image" src="https://github.com/user-attachments/assets/5d95597d-5e54-410b-8fa2-72c7049f1945" />

Why squared?
✔ Removes negatives
-Least Squared Method
The **Least Squares Method** is the mathematical principle used by linear regression to find the _best possible line_.
-We do not just add the errors as it may cancel out positive and negative errors instead of this we square the error before adding it.
-It helps us to find the values of m and b
-How do we actually find m, b
1.we look for values which minimizes J(m,b)
![[Pasted image 20260223102306.png]]2. Gradient Descent (ML Approach)
-Start with random m and b
- Measure error
- adjust the parameters gradually using learning rate and derivatives(gradient descent)
![[Pasted image 20260223102512.png]]

### *Multivariable Regression* 
Multiple inputs → One output
Y=b+m1​X1​+m2​X2​+m3​X3​+⋯+mn​Xn​
-Example can be taken of a House price prediction, where it depends on multiple factors like Size, Bedrooms, Location, Area....
-Y predicted=b+m1X1+m2X2+....
-Cost function is J=1/n(sigma(yactual-ypredicted))
-m1, m2....=coeffients

### Polynomial Regression

When the relationship between X and Y is not linear, so instead of plotting the best fit line, we plot a curve
![[Pasted image 20260223110225.png]]


### Ridge Regression
Linear regression + L2 regularization (penalty)

L2 regularization (also called **Ridge regularization** or **weight decay**) is a technique used in machine learning to reduce **overfitting** by discouraging large parameter values.

Instead of only minimizing the training error, we add a penalty for large weights.
- Ridge Regression solves two problems-> overfitting(when noise is also fitted instead of true patterns)
- Multicollinearity (when features are strongly correlated)
![[Pasted image 20260223111305.png]]
### Lasso Regression

Least Absolute Shrinkage and Selection Operator

Linear Regression + L1 Regularization
L1 regularisation (also called **Lasso regularisation**) is a technique used to reduce **overfitting** by adding a penalty based on the **absolute values of weights**.
![[Pasted image 20260223112916.png]]
### Gradient Descent and Cost function

Cost function=> total prediction error of the model
Gradient descent=> Method to minimize the cost functionn

### Logistic regression
- It is used for binary classification where output is 0/1
- instead of predicting y, we predict the probability, we map the real numbers with probability using sigmoid function
- Sigmoid Function:
- ![[Pasted image 20260223114450.png]]
- output of a sigmoid function is either 0 or 1
Steps are:
1. Calculate z
2. Find P(z)
3. check range of this (classification rule)
P(z)>0.5 YES
P(z)<=0.5 NO
Cost Function
- Linear regression used squared error but it is not used here as there are probability values, and it may lead to non-convex loss function
- Logistic regression uses Maximum Likelihood Estimation(MLE)
Logistic regression predicts:

![[Pasted image 20260223115441.png]]

For one data point:
Case 1: If Y=1
Correct probability = P

Case 2: If Y=0
Correct probability = 1−P
Combined Expression
Both cases can be written compactly:

![[Pasted image 20260223115525.png]]
This works because:

- If Y=1→ gives P
    
- If Y=0→ gives 1−P
    
Likelihood for the Dataset

For all data points:

![[Pasted image 20260223115559.png]]

This represents how probable the observed data is under the model.

Goal: Maximize likelihood.

Log Transformation

Products are difficult to optimize, so we take logarithms.

![[Pasted image 20260223115620.png]]

Log converts product → sum, simplifying optimization.

Cost Function Definition

Maximizing log-likelihood is equivalent to minimizing negative log-likelihood.

![[Pasted image 20260223115638.png]]

This is called:

- Log Loss
- Cross-Entropy Loss
- Binary Cross-Entropy
Decision Boundary: 
- In order to classify whether probability is 0 or 1, we need to consider a threshold, although it is not fixed, but a default value of 0.5 is considered
- Why? because 
![[Pasted image 20260223121104.png]]

### k-Fold Cross Validation
* while training a model, we need to consider how well will it perform on unseen data
* K- fold cross validation divides the dataset into k equal parts(folds) and perform multiple training-testing cycles
* Steps:
1. Split data into K folds
2. Train on K-1 folds
3. Test on remaining fold
4. Repeat K times
5. make record of accuracy per iteration
6. then average accuracy is the final accuracy

### KNN K nearest neighbors
Similar data points tend to have similar output.
- We chose a value K 
- Pick how many neighbors you want to consider
- k=1 look at only the closest point
- k=5 look at the 5 closest points

- Measure distance between the new point, and every training point using Euclidean Distance.
- find nearest neighbors
### Naive Bayes
It is a probabilistic classifier

Bayes Theorem
![[Pasted image 20260224210207.png]]
In plain language:
Posterior = Likelihood × Prior / Evidence
Where:

✔ **Posterior** → What we want  
✔ **Likelihood** → How likely data is under a class  
✔ **Prior** → Initial belief about class  
✔ **Evidence** → Normalization factor
