import numpy as np
import matplotlib.pyplot as plt

# D -> Kpi
eta = 2

m_1 = 1.869 # D
m_2 = 0.493 # K
t_plus = (m_1 + m_2)**2
t_minus = 1.8846 #(m_1 - m_2)**2

# susceptibility at zero
X = 0.0043

def z(t):
    num = np.sqrt((t_plus-t)/(t_plus-t_minus)) - 1
    den = np.sqrt((t_plus-t)/(t_plus-t_minus)) + 1
    return num/den


def rho(t):
    return np.sqrt((t_plus-t)/(t_plus-t_minus))


def phi(t, ff='+', **kwargs):
    if ff=='+':
        factor1 = np.sqrt(2*eta/(3*np.pi*(t_plus-t_minus)))
        factor2 = ((1+z(t))**2)/((1-z(t))**(9/2))
        factor3 = ((rho(0)+rho(t))**(-5))
    elif ff=='0':
        factor1 = np.sqrt(eta*t_plus*t_minus/(2*np.pi))
        factor2 = (1+z(t))/((t_plus-t_minus)*((1-z(t))**(5/2))) 
        factor3 = ((rho(0)+rho(t))**(-4))
    else:
        factor1, factor2, factor3 = 1, 1, 1
    return factor1*factor2*factor3

def phi_p(phi, t, t_p, **kwargs):
    if type(t_p)==list:
        for p in t_p:
            phi = phi*(z(t) - z(p))/(1-(z(p)*z(t)))
        return phi
    else:
        return phi*(z(t) - z(t_p))/(1-(z(t_p)*z(t)))


def phi_ff(idx, ffs, t_p=None, **kwargs):
    if t_p==None:
        return phi(known_ts[idx])*ffs[idx]
    else:
        PHI = phi_p(phi(known_ts[idx]), known_ts[idx], t_p)
        return PHI*ffs[idx]

def gg(t1,t2):
    return 1/(1 - z(t1)*z(t2))

def g_matrix(t_values):
    return np.array([[gg(t1,t2) for t2 in t_values] for t1 in t_values])

def del_row_col(matrix, row, col):
    matrix = np.delete(matrix, row, axis=0)
    matrix = np.delete(matrix, col, axis=1)
    return matrix

def G_M_1_1(unknown_t, known_ts, known_ffs, t_p=None, X=X, **kwargs):
    N = len(known_ffs)
    all_ts = np.hstack([unknown_t, known_ts])
    G = g_matrix(all_ts)

    M_1_1 = np.array([np.hstack([phi_ff(i,known_ffs,t_p),
                    del_row_col(G,0,0)[i,:]]) for i in range(N)])
    top_line = np.hstack([X, np.array([phi_ff(i,known_ffs,t_p)
                        for i in range(N)])])
    M_1_1 = np.vstack([top_line, M_1_1])
    #print(M_1_1)

    unitarity_satisfied = True if (det(G)>=0 and det(M_1_1)>=0) else False
    return G, M_1_1, unitarity_satisfied

from numpy.linalg import det
def bounds(unknown_t, known_ts, known_ffs, t_p=None, X=X, **kwargs):
    N = len(known_ffs)

    G, M_11, unitarity_satisfied = G_M_1_1(unknown_t, known_ts, known_ffs, t_p)

    alpha = det(del_row_col(G,0,0))
    beta = np.sum(np.array([((-1)**j)*phi_ff(j,known_ffs,
                      t_p)*det(del_row_col(G,0,j)) for j in range(N)]))
    gamma_mtx = np.array([[((-1)**(i+j))*phi_ff(i,known_ffs,t_p)*phi_ff(j,
                        known_ffs,t_p)*det(del_row_col(G,i,j))
                        for j in range(N)] for i in range(N)])
    gamma = X*det(G) - np.sum(gamma_mtx)

    discriminant = (beta**2) + alpha*gamma

    upper_bound = (-beta + np.sqrt(discriminant))/(alpha*phi(unknown_t))
    lower_bound = (-beta - np.sqrt(discriminant))/(alpha*phi(unknown_t))

    #return (lower_bound, upper_bound, unitarity_satisfied)
    return [lower_bound, upper_bound]

def bootstrap_ffs(known_ffs, ffs_cov, K=100, **kwargs):
    N = len(known_ffs)
    np.random.seed(1)
    samples = np.random.multivariate_normal(known_ffs, ffs_cov, K)
    return samples

known_ts = np.array([1.3461, 1.6154, 1.8846])

# for f+
known_ffs_plus = np.array([1.102, 1.208, 1.336])
ffs_plus_err = np.array([0.038, 0.0443, 0.054])
pole_t = 2.1122

N_boot = 1000
unknown_t = 0.2692
samples = bootstrap_ffs(known_ffs_plus, np.diag(ffs_plus_err), K=N_boot)
accepted_idx = []
for k in range(N_boot):
    G, M_1_1, us = G_M_1_1(unknown_t,known_ts,samples[k,:], pole_t)
    if us==True:
        accepted_idx.append(k)
        
accepted_samples = samples[accepted_idx,:]
N_acc = accepted_samples.shape[0] 
bounds_dist = np.array([bounds(unknown_t, known_ts, accepted_samples[n,:], pole_t)
                        for n in range(N_acc)])




























