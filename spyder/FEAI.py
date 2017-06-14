# 《有限元分析及应用》 例4.1（1） 1D阶梯杆结构的有限元分析
import numpy as np
from equation import LinearSolverII
def EX040101(): 
    E1 = 2.0e7
    E2 = 2.0e7
    A1 = 2.0e-4
    A2 = 1.0e-4
    l1 = 10.0e-2
    l2 = 10.0e-2
    F3 = 10.0
    
    print("E1", E1)
    print("E2", E2)
    print("A1", A1)
    print("A2", A2)
    print("l1", l1)
    print("l2", l2)
    print("F3", F3)
    
    
    print("应变-位移矩阵")
    B1 = np.array([-1.0/l1, 1.0/l1])
    B2 = np.array([-1.0/l2, 1.0/l2])
    print(B1)
    print(B2)
    
    print("应力-位移矩阵")
    S1 = np.array([-E1/l1, E1/l1])
    S2 = np.array([-E2/l2, E2/l2])
    print(S1)
    print(S2)
    
    print("刚度矩阵：")
    SM = np.array([[E1*A1/l1+E2*A2/l2,-E2*A2/l2],[-E2*A2/l2,E2*A2/l2]])
    print(SM)
    print("载荷向量：")
    F = np.array([0.0,10.0])
    print(F)
    print("求解刚度方程：")
    u = LinearSolverII(SM, F)
    print(u)
    
    print("单元位移：")
    u1 = np.array([0.0, u[0]])
    u2 = u
    print(u1)
    print(u2)
    p = np.array([0.0, u[0], u[1]])
    print(p)

    print("单元应变：")    
    epsilon1 = np.dot(B1, u1)
    print(epsilon1)
    epsilon2 = np.dot(B2, u2)
    print(epsilon2)
    
    print("单元应力：")
    sigma1 = np.dot(S1, u1)
    sigma2 = np.dot(S2, u2)
    print(sigma1)
    print(sigma2)
    
    print("支反力")
    K1 = E1*A1/l1*np.array([[1,-1],[-1, 1]])
    print(K1)
    print(u1)
    print(np.dot(K1,u1))
    
def EX040301(): 
    E1 = 3.0e7
    E2 = 3.0e7
    A1 = 5.25
    A2 = 3.75
    l1 = 12.0
    l2 = 12.0
    
    F2 = 100.0    
    PW1 = 8.9334
    PW2 = 15.3144
    PW3 = 6.381
    
    print("E1", E1)
    print("E2", E2)
    print("A1", A1)
    print("A2", A2)
    print("l1", l1)
    print("l2", l2)
    
    print("F2", F2)
    print("PW1", PW1)
    print("PW2", PW2)
    print("PW3", PW3)
    
    
    print("应变-位移矩阵")
    B1 = np.array([-1.0/l1, 1.0/l1])
    B2 = np.array([-1.0/l2, 1.0/l2])
    print(B1)
    print(B2)
    
    print("应力-位移矩阵")
    S1 = np.array([-E1/l1, E1/l1])
    S2 = np.array([-E2/l2, E2/l2])
    print(S1)
    print(S2)
    
    print("刚度矩阵：")
    SM = np.array([[E1*A1/l1+E2*A2/l2,-E2*A2/l2],[-E2*A2/l2,E2*A2/l2]])
    print(SM)
    
    print("载荷向量：")
    F = np.array([F2+PW2,PW3])
    print(F)
    
    print("求解刚度方程：")
    u = LinearSolverII(SM, F)
    print(u)
    
    print("单元位移：")
    u1 = np.array([0.0, u[0]])
    u2 = u
    print(u1)
    print(u2)
    p = np.array([0.0, u[0], u[1]])
    print(p)

    print("单元应变：")    
    epsilon1 = np.dot(B1, u1)
    print(epsilon1)
    epsilon2 = np.dot(B2, u2)
    print(epsilon2)
    
    print("单元应力：")
    sigma1 = np.dot(S1, u1)
    sigma2 = np.dot(S2, u2)
    print(sigma1)
    print(sigma2)
    
    print("支反力")
    K1 = E1*A1/l1*np.array([[1,-1],[-1, 1]])
    print(K1)
    print(u1)
    P1 = np.dot(K1,u1)
    print("R1", P1[0]-PW1)
 
    
def EX040302(): 
    E = 20.0e3
    A = 250.0
    l = 150.0
    F = 60.0e3
    
    print("单元1的刚度矩阵：")
    K1 = E*A/l*np.array([[1, -1], [-1, 1]])
    print(K1)
    print("单元1的刚度矩阵：")
    K2 = E*A/l*np.array([[1, -1], [-1, 1]])
    print(K2)
    print("整体刚度矩阵：")
    print("初始的3*3的零矩阵:")
    K = np.zeros((3,3))
    print(K)
    print("单元刚度矩阵的组装")
    K[0:2,0:2] = K[0:2,0:2] + K1
    K[1:3,1:3] = K[1:3,1:3] + K2
    print(K)
    print("(1)判断接触是否发生：")
    print("  假设边界条件 u1 = 0, R3 = 0：")
    print("刚度矩阵变为：")
    K_Assumption1 = K[1:3,1:3]
    print(K_Assumption1)
    print("载荷向量")
    P_Assumption1 = np.array([60.0e3, 0.0])
    print(P_Assumption1)
    u_Assumption1 = np.linalg.solve(K_Assumption1, P_Assumption1)
    print(u_Assumption1)
    print("判断结果，接触发生")
    print("（2）接触发生时的分析：")
    print("修改边界条件：u1=0,u3=1.2")
    k = E*A/l
    u2 = (F+1.2*k)/(2*k)
    print(u2)
    u = np.array([0.0, u2, 1.2])
    P = np.dot(K,u)
    print("载荷向量：")
    print(P)
    print("(3)求发生接触的临界条件：")
    print("修改边界条件：u1=0,u3=1.2, R3=0, 求Fcr")
    u2cr = u3cr = 1.2
    print(u2cr)
    ucr = np.array([0.0, u2cr, u3cr])
    Pcr = np.dot(K,ucr)
    print("临界载荷：")
    print(Pcr[1])
    print("单元应力计算")
    S1 = S2 = E/l*np.array([-1,1])
    sigma1 = np.dot(S1,u[0:2])
    sigma2 = np.dot(S2,u[1:3])
    print("单元1应力：")
    print(sigma1)
    print("单元2应力：")
    print(sigma2)


import math as math    
def EX040303():
    E = 29.5e4
    A = 100
    L1 = 400
    L2 = 300
    L3 = 500
    L4 = 400
    
    Fu2 =  20000.0
    Fv3 = -25000.0
    
    print("单元节点关系：（1） 1， 2 ")
    print("单元节点关系：（2） 2， 3 ")
    print("单元节点关系：（3） 1， 3 ")
    print("单元节点关系：（4） 3， 4 ")
    
    alpha1 = 0.0
    alpha2 = np.pi/2
    alpha3 = math.atan(L2/L1)
    alpha4 = np.pi
    
    print("(1)号单元：")
    print("刚度矩阵系数")
    k1 = E*A/L1
    print(k1)
    print("整体坐标系下刚度矩阵")
    K1 = k1*GetKe(alpha1)
    print(K1)

    print("(2)号单元：")
    print("刚度矩阵系数")
    k2 = E*A/L2
    print(k2)
    print("整体坐标系下刚度矩阵")
    K2 = k2*GetKe(alpha2)
    print(K2)
    
    print("(3)号单元：")
    print("刚度矩阵系数")
    k3 = E*A/L3
    print(k3)
    print("整体坐标系下刚度矩阵")
    K3 = k3*GetKe(alpha3)
    print(K3)
    
    print("(4)号单元：")
    print("刚度矩阵系数")
    k4 = E*A/L4
    print(k4)
    print("整体坐标系下刚度矩阵")
    K4 = k4*GetKe(alpha4)
    print(K4)
    
    print("组装整体坐标系下的整体刚度矩阵：")
    K = np.zeros((8,8))
    print(K)
    
    K[0:4,0:4] = K[0:4,0:4] + K1
    K[2:6,2:6] = K[2:6,2:6] + K2
    
    K[0:2,0:2] = K[0:2,0:2] + K3[0:2,0:2]
    K[0:2,4:6] = K[0:2,4:6] + K3[0:2,2:4]
    K[4:6,0:2] = K[4:6,0:2] + K3[2:4,0:2]
    K[4:6,4:6] = K[4:6,4:6] + K3[2:4,2:4]

    K[4:8,4:8] = K[4:8,4:8] + K4

#    K = K*6000/29.5e6
    
    print(K)
    
    K_ = np.zeros((3,3))
    print (K_)
    
    K_[0,0] = K[2,2]
    K_[0,1] = K[2,4]
    K_[0,2] = K[2,5]
    K_[1,0] = K[4,2]
    K_[2,0] = K[5,2]
    K_[1:3,1:3] = K[4:6,4:6]

    print(K_)
    
    print("载荷向量：")
    F = np.array([Fu2, 0.0, Fv3])
    print (F)
    
    print("求解刚度方程：")
    u = np.linalg.solve(K_,F)
    print(u)
    
    
def GetKe(a):
#    坐标变换矩阵
    Te = np.array([[np.cos(a), np.sin(a), 0,         0        ],
                   [0,         0,         np.cos(a), np.sin(a)]])
#    刚度系数矩阵
    K = np.array([ [  1, -1],
                   [ -1,  1] ])
#    print(K)
#    print(Te)
#    刚度系数转置
    TeT = Te.transpose()
#    print(TeT)
#    print(np.dot(np.dot(TeT,K),Te))
#   根据公式 （4.90）计算整体坐标系下的刚度矩阵
    return np.dot(np.dot(TeT,K),Te)
    
#    x1 = np.arange(9.0).reshape((3, 3))
#    print(x1)
#    x2 = np.arange(3.0)
#    print(x2)
#    print(np.multiply(x1,x2))
    
if __name__ == '__main__':
    EX040303()
#    print(np.sin(0.0))
#    print(np.cos(np.pi/2))
#    print(np.cos(np.pi/2)*np.cos(np.pi/2))
#    a = np.arange(0,60,10)
#    print(a)
#    print(a.shape)
#    a = np.arange(0,60,10).reshape(-1,1)
#    print(a)
#    print(a.shape)
#    b = np.arange(0,6)
#    print(b)
#    c = a+b
#    print(c)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass