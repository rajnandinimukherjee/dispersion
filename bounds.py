import numpy as np

#m_1 = 
#m_2 = 
#t_plus = (m_1 + m_2)**2
#t_minus = (m_1 - m_2)**2

#test values
t_plus = 2.5
t_minus = 1.8846

def z(t):
    num = np.sqrt((t_plus-t)/(t_plus-t_minus)) - 1
    den = np.sqrt((t_plus-t)/(t_plus-t_minus)) + 1
    return num/den

known_ts = np.array([1.3461, 1.6154])#, 1.8846])
known_ffs = np.array([1.102, 1.208])#, 1.336])
ffs_err = np.array([0.038, 0.0443])#, 0.054])

def phi(t):
    return 1

X = 0.0043

def phi_ff(idx):
    return phi(known_ts[idx])*known_ffs[idx]

def gg(t1,t2):
    return 1/(z(t1)*z(t2))

def g_matrix(t_values):
    return np.array([[gg(t1,t2) for t2 in t_values] for t1 in t_values])

def del_row_col(matrix, row, col):
    matrix = np.delete(matrix, row, axis=0)
    matrix = np.delete(matrix, col, axis=1)
    return matrix

def G_M_1_1(unknown_t, known_ts, known_ffs, X):
    N = len(known_ffs)
    all_ts = np.hstack([unknown_t, known_ts])
    G = g_matrix(all_ts)

    M_1_1 = np.array([np.hstack([phi_ff(i),del_row_col(G,0,0)[i,:]])
                    for i in range(N)])
    top_line = np.hstack([X, np.array([phi_ff(i) for i in range(N)])])
    M_1_1 = np.vstack([top_line, M_1_1])
    #print(M_1_1)

    unitarity_satisfied = True if (det(G)>=0 and det(M_1_1)>=0)else False
    return G, M_1_1, unitarity_satisfied

from numpy.linalg import det
def bounds(unknown_t, known_ts, known_ffs, X):
    N = len(known_ffs)

    G, M_11, unitarity_satisfied = G_M_1_1(unknown_t, known_ts, known_ffs, X)

    alpha = det(del_row_col(G,0,0))
    beta = np.sum(np.array([((-1)**j)*phi_ff(j)*det(del_row_col(G,0,j))
                    for j in range(N)]))
    gamma_mtx = np.array([[((-1)**(i+j))*phi_ff(i)*phi_ff(j)*det(del_row_col(G,i,j))
                        for j in range(N)] for i in range(N)])
    gamma = X*det(G) - np.sum(gamma_mtx)

    discriminant = (beta**2) + alpha*gamma

    upper_bound = (-beta + np.sqrt(discriminant))/(alpha*phi(unknown_t))
    lower_bound = (-beta - np.sqrt(discriminant))/(alpha*phi(unknown_t))

    #return (lower_bound, upper_bound, unitarity_satisfied)
    return (-lower_bound, -upper_bound)

def bootstrap_ffs(known_ffs, ffs_err, K=100, **kwargs):
    N = len(known_ffs)
    np.random.seed(1)
    samples = np.array([np.random.normal(known_ffs[i], ffs_err[i], K)
                        for i in range(N)])
    return samples

K=1000
unknown_t = 0.5385
samples = bootstrap_ffs(known_ffs, ffs_err, K=K)
accepted_idx = []
for k in range(K):
    G, M_1_1, us = G_M_1_1(unknown_t,known_ts,samples[:,k],X)
    if us==True:
        accepted_idx.append[k]
        
accepted_samples = samples[:,accepted_idx]





























