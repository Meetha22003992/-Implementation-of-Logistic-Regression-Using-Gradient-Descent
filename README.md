# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import numpy as np.
2. Import matplot.lip.pyplot.
3. Print array of X value.
4. Print array of Y values.
5. By plotting the x and y values get the Sigmoid function grap.
6. Print the grad values of X and Y.
7. Plot the decision boundary of the given data.
8. Obtain the probability value.
9. Get the prediction value of mean and print it.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Meetha Prabhu 
RegisterNumber:  212222240065
 **
 import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
**

**
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
**

**
print("Array value of x:")
X[:5]
**

**
print("Array value of y:")
y[:5]
**

**
plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1 - Score graph:")
plt.show()
**

**
def sigmoid(z):
  return 1/(1+np.exp(-z))
**

**
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
print("Sigmoid fucntion graph:")
plt.show()
**

**
def costfunction(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J, grad
 **
 
 **
 X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costfunction(theta,X_train,y)
print("X_train_grad value:")
print(J)
print(grad)
**

**
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costfunction(theta,X_train,y)
print("Y_train_grad value:")
print(J)
print(grad)
**

**
def cost (theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return J
**

**
def gradient(theta,X,y):
   h=sigmoid(np.dot(X,theta))
   grad=np.dot(X.T,h-y)/X.shape[0]
   return grad
**

**
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print("Res.x:")
print(res.x)
**

**
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
**

**
print("Decision Boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,y)
**

**
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print("Probability Value:")
print(prob)
**

**
def predict(theta,X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
 **
 
 **
 print("Prediction value of mean:")
np.mean(predict(res.x,X)==y)
**
*/
```

## Output:
![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/d2391f61-7de5-4afb-ac0d-0dbfa8fda81f)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/96ef80c5-8f51-4dfe-8321-e4d32f45c6d1)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/f2406341-6755-41a3-a308-f6600673fa0b)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/1f38d8ee-5cd3-4899-b931-ce5a7bc23920)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/0972a2f3-12cc-4e24-9899-1d5823648f0a)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/0d50459b-b8bc-4886-bd70-4135ca0090b5)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/0a5197f6-6940-4144-95fe-cc4ac01b13ed)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/a87ec5ec-56ad-47a2-b68a-1aaabeb77052)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/1e4e1975-b935-4bfe-9c80-7ce0f2e26315)

![image](https://github.com/Meetha22003992/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119401038/5436e5ac-c497-4543-a966-83ff4065a5fb)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

