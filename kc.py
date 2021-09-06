import numpy as np
import matplotlib.pyplot as plt

# D -> K
eta = 2

m_1 = 1.869 # D
m_2 = 0.493 # K
t_plus = (m_1 + m_2)**2
t_minus = 1.8846 #(m_1 - m_2)**2

Q_sq = 0

def z(t):
    a = np.sqrt(t_plus-t)
    b = np.sqrt(t_plus-t_minus)
    return (a-b)/(a+b)

def rho(t):
    return np.sqrt((t_plus-t)/(t_plus-t_minus))

def phi(t, t_p=None, Q_sq=0, ff='+',  **kwargs):
    if ff=='0':
        f1 = np.sqrt(eta*t_plus*t_minus/(2*np.pi))/(t_plus-t_minus)
        f2 = (1+z(t))/((1-z(t))**(5/2))
        f3 = ((rho(0)+rho(t))*(rho(Q_sq)+rho(t)))**(-2)
    elif ff=='+':
        f1 = np.sqrt(2*eta/(3*np.pi*(t_plus-t_minus)))
        f2 = ((1+z(t))**2)/((1-z(t))**(9/2))
        f3 = ((rho(0)+rho(t))**(-2))*((rho(Q_sq)+rho(t))**(-3))
    else:
        f1, f2, f3 = 1, 1, 1

    phi = f1*f2*f3

    if t_p!=None:
        if type(t_p)==list:
            for p in t_p:
                phi = phi*(z(t)-z(p))/(1-z(t)*z(p))
        else:
            phi = phi*(z(t)-z(t_p))/(1-z(t)*z(t_p))
    
    return phi

def G(t_vec, **kwargs):
    G = np.array([[1/(1-z(t1)*z(t2)) for t2 in t_vec] for t1 in t_vec])
    return G

def del_RC(matrix, row, col):
    matrix = np.delete(matrix, row, axis=0)
    matrix = np.delete(matrix, col, axis=1)
    return matrix

def M11(known_ts, known_ffs, X, **kwargs):
    N = len(known_ts)
    phi_ff_vec = phi(np.array(known_ts),**kwargs)*known_ffs
    #phi_ff_vec = np.array([phi(known_ts[i],**kwargs)*known_ffs[i]
    #                       for i in range(N)])

    top_line = np.hstack((X, phi_ff_vec))
    g00 = G(known_ts)
    bottom = np.array([np.hstack((phi_ff_vec[i], g00[i,:])) for i in range(N)])

    M11 = np.vstack((top_line, bottom))
    return M11

from numpy.linalg import det
def bounds(unknown_t, known_ts, known_ffs, X, prnt=False, **kwargs):
    N = len(known_ts)
    g = G(np.hstack((unknown_t, known_ts)))

    alpha = det(del_RC(g,0,0))
    beta = sum([((-1)**(j+1))*phi(known_ts[j],**kwargs)*known_ffs[j]*det(del_RC(g,0,j+1))
                for j in range(N)])
    gamma_mtx = np.array([[((-1)**(i+j+2))*phi(known_ts[i],**kwargs)*phi(known_ts[j],
                        **kwargs)*known_ffs[i]*known_ffs[j]*det(del_RC(g,i+1,j+1)) 
                            for j in range(N)] for i in range(N)])
    gamma = X*det(g)-np.sum(gamma_mtx)

    disc = ((beta**2)+(alpha*gamma))
    upper_bound = (-beta+(disc**0.5))/(alpha*phi(unknown_t,**kwargs))
    lower_bound = (-beta-(disc**0.5))/(alpha*phi(unknown_t,**kwargs))

    if prnt:
        print((lower_bound, upper_bound), disc)
    return [lower_bound, upper_bound]

def bootstrap(var, cov, K=100, **kwargs):
    np.random.seed(1)
    samples = np.random.multivariate_normal(var, cov, K)
    return samples


known_ts = [1.3461, 1.6154, 1.8846]
g00 = G(known_ts)
print('g00 is pos def') if det(g00)>0 else print('g00 is not pos def')

unknown_ts = [0.0, 0.2692, 0.5385, 0.8077, 1.0769]
list_t = np.hstack((unknown_ts,known_ts))

ffs_zero = [0.911, 0.944, 0.979]
ffs_plus = [1.102, 1.208, 1.336]
COV = np.loadtxt('cov.txt')
COV_input = np.block([[COV[5:8,5:8],COV[5:8,12:15]],
                      [COV[12:15,5:8],COV[12:15,12:15]]])

#known_vals = {'known_ts':known_ts, 'ffs_zero':ffs_zero,
#              'ffs_plus':ffs_plus, 'Q_sq':0, 'N_boot':1000,
#              'X_zero':0.0043, 'X_zero_err':0.0013,
#              'X_plus':0.00419, 'X_plus_err': 0.00036,
#              't_p_zero':2.3178, 't_p_plus':2.1122}

known_ffs = np.hstack((ffs_zero,ffs_plus))
X_zero, X_plus = 0.0043, 0.00419
X_zero_err, X_plus_err = 0.0013, 0.00036
X, X_err = np.array([X_zero,X_plus]), np.array([X_zero_err, X_plus_err])

dict_zero = {'t_p':2.3178, 'ff':'0'}
dict_plus = {'t_p':2.1122, 'ff':'+'}

t_range = np.arange(0,t_minus,0.01)+0.01
N_boot = 100
N_0 = 10
samples = bootstrap(known_ffs, COV_input, K=N_boot)
samples_X = bootstrap(X,np.diag(X_err)**2,K=N_boot)

zero_dist = {str(round(t,2)):{'up':[], 'lo':[]} for t in t_range}
plus_dist = {str(round(t,2)):{'up':[], 'lo':[]} for t in t_range} 
accepted_idx = []

import time
t1 = time.time()
for k in range(N_boot):
    m11_zero = M11(known_ts, samples[k,:3], samples_X[k,0], **dict_zero)
    m11_plus = M11(known_ts, samples[k,3:], samples_X[k,1], **dict_plus)
    if det(m11_zero)>0 and det(m11_plus)>0:
        [zero_low, zero_up] = bounds(0, known_ts, samples[k,:3], samples_X[k,0],
                                     **dict_zero)
        [plus_low, plus_up] = bounds(0, known_ts, samples[k,3:], samples_X[k,1],
                                     **dict_plus)
        if zero_up>plus_low and plus_up>zero_low: # kinematical constraint
            accepted_idx.append(k)
            np.random.seed(1)
            f0s = np.random.uniform(max(zero_low,plus_low), min(zero_up,plus_up), N_0)
            known_ts_0 = np.hstack((known_ts,0))
            
            for t in t_range:
                zero_bounds = np.array([bounds(t,known_ts_0,
                                np.hstack((samples[k,:3],f0s[n])),
                                samples_X[k,0],**dict_zero)
                                for n in range(N_0)])

                plus_bounds = np.array([bounds(t,known_ts_0,
                                np.hstack((samples[k,3:],f0s[n])),
                                samples_X[k,1],**dict_plus)
                                for n in range(N_0)])

                zero_dist[str(t)]['lo'].append(np.min(zero_bounds[:,0]))
                zero_dist[str(t)]['up'].append(np.min(zero_bounds[:,1]))
                plus_dist[str(t)]['lo'].append(np.min(plus_bounds[:,0]))
                plus_dist[str(t)]['up'].append(np.min(plus_bounds[:,1]))

print('Time taken:',time.time()-t1)
































#N_acc = len(accepted_idx)
#accepted_samples = samples[accepted_idx,:]
#accepted_X = samples_X[accepted_idx,:]
#
#def constrained_bounds(unknown_t, known_ts, known_ffs, X, **kwargs):
#    N_0 = 100
#    f_dist = []
#    for k in range(N_acc):
#        f0s = np.random.uniform(accepted_bounds[k][0], accepted_bounds[k][1], N_0)
#        bnds= []
#        for n in range(N_0):
#            known_ts_0 = np.hstack((known_ts,0))
#            known_ffs_0 = np.hstack((known_ffs,f0s[n]))
#            bnds.append(bounds(unknown_t, known_ts_0, known_ts_0, X,
#                                        **kwargs))
#        bnds = np.array(bnds)
#        f_dist.append([min(bnds[:,0]), max(bnds[:,1])])
#    
#    return f_dist
#        
#def bounds_with_errors(t_range, **kwargs):
#    t_dist = []
#    for t in t_range:
#        t_dist.append(constrained_bounds(unknown_t=t,**kwargs))
#
#    return np.array(t_dist)
#
#t_dist_zero = bounds_with_errors(t_range, known_ts=known_ts, known_ffs=ffs_zero,
#                                X=X_zero, **dict_zero)



        









































