import numpy as np

def EX050201():
    print("单元节点对应关系：")
    print("1号单元：2,3,4")
    print("2号单元：3,2,1")
    print("1号单元刚度矩阵：")
    K1 = GetKe(2.0,0.0, 0.0,1.0, 0.0,0.0, 1, 1.0/3.0)
    print(K1)
    print("2号单元刚度矩阵：")
    K2 = GetKe(0.0,1.0, 2.0,0.0, 2.0,1.0, 1, 1.0/3.0)
    print(K2)
    
    K = np.zeros((8,8))
    
    K[2:8,2:7] = K[2:8,2:7]+K1
    
    

    print(K)
    
    pass

def GetKe(x1,y1,x2,y2,x3,y3, E, Mu):
#    print("A")
    A_ = np.array([[1,1,1],[x1,x2,x3],[y1,y2,y3]])
#    print(A_)
    A = np.linalg.det(A_)*0.5
#    print(A)
    a1 = x2*y3-x3*y2
    b1 = y2-y3
    c1 = -x2+x3
    a2 = x3*y1-x1*y3
    b2 = y3-y1
    c2 = -x3+x1
    a3 = x1*y2-x2*y1
    b3 = y1-y2
    c3 = -x1+x2
#    print((a1+a2+a3)/2.0)
#    print((b1*c2-b2*c1)/2.0)
    B = np.array([[b1,  0.0, b2,  0.0, b3,  0.0],
                  [0.0, c1,  0.0, c2,  0.0, c3 ],
                  [c1,  b1,  c2,  b2,  c3,  b3 ]])*(1.0/(2*A))
#    print(B)
    
    D = np.array([[1.0,  Mu,   0.0         ],
                  [Mu,   1.0,  0.0         ],
                  [0.0,  0.0,  (1.0-Mu)/2.0],])*(E/(1-Mu*Mu))
    
#    print(D)
    
    Ke = np.dot(np.dot(B.transpose(),D),B)*A
    return Ke
#    print(Ke)
    
    pass

if __name__ == '__main__':
#    GetKe(2.0,0.0, 0.0,1.0, 0.0,0.0, 1, 1.0/3.0)
    EX050201()
    pass