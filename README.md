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
![image](https://user-images.githubusercontent.com/119401038/233590840-cb0c926c-dfcf-401d-8543-2729b530b886.png)
![image](https://user-images.githubusercontent.com/119401038/233591041-94ba05fd-dbf9-4b07-9e20-099e3923fb6f.png)
![image](https://user-images.githubusercontent.com/119401038/233591126-2aa99939-4403-4ec4-ba11-2a03074fb6a4.png)
![image](https://user-images.githubusercontent.com/119401038/233591338-ee424e4c-3ec0-461c-8e59-060924f6ef1c.png)
![image](https://user-images.githubusercontent.com/119401038/233591573-92479396-0247-4d3e-8f81-ff72a86fb9e0.png)
![image](https://user-images.githubusercontent.com/119401038/233591638-31be73c4-2388-4a4d-b237-7d72e4beb1de.png)
![image](https://user-images.githubusercontent.com/119401038/233591884-f5877f70-2e0a-4a8e-982d-c1860c7b95dc.png)
![image](https://user-images.githubusercontent.com/119401038/233591937-6946fe00-17b4-4cf4-9c6e-56783369990a.png)
![image](https://user-images.githubusercontent.com/119401038/233591987-fcce2982-4afc-476d-923b-98d81d8d1731.png)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

