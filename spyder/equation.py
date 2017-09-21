import numpy as np

def LinearSolverI():
    A = np.mat('1,3,5; 2,5,1; 2,3,8')
    print(A)
    b = np.array([10,8,3])
    print(b)
    x = np.linalg.solve(A, b)
    print('Solution: ', x)

def LinearSolverI1():
    A = np.array([[1,3,5], [2,5,1], [2,3,8]])
    print(A)
    b = np.array([10,8,3])
    print(b)
    x = np.linalg.solve(A, b)
    print('Solution: ', x)
    
def LinearSolverII(A, b):
    x = np.linalg.solve(A, b)
    return x

from scipy.optimize import fsolve
def f(x):                                  #定义函数
    x1,x2=x[0],x[1]
    return [x1+x2-4,  x1**2 + x2**2 - 8]   # 返回误差

def NoLinearSolverI():
    result=fsolve(f,[1,1])                     # 调用 fsolve 函数                     
    print ('the result is',result)
    print ('the error is',f(result))


    
if __name__ == '__main__':
#    A = np.array([[1,3,5], [2,5,1], [2,3,8]])
#    print(A)
#    b = np.array([10,8,3])
#    print(b)
#    x = LinearSolverII(A,b)
#    print(x)
    LinearSolverI1()
    pass
	
#  test for git commit.