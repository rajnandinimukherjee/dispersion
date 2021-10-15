import numpy as np
import matplotlib.pyplot as plt
import pdb
prec = np.float64

# Bs -> K

eta = 1

m_Bs = 5.366820
m_pi = 0.050704
t_cut = (m_Bs + m_pi)**2
t_plus = 29.349571
t_minus = 26.434463
t_0 = 16.505107

Q_sq = 0

def z(t):
    a = np.sqrt(t_cut-t)
    b = np.sqrt(t_cut-t_0)
    return (a-b)/(a+b)

def rho(t):
    return np.sqrt((t_cut-t)/(t_cut-t_0))

def phi(t, t_p=None, Q_sq=0, ff='+',  **kwargs):
    if ff=='0':
        f1 = np.sqrt(eta*t_cut*t_0/(2*np.pi))/(t_cut-t_0)
        f2 = (1+z(t))/((1-z(t))**(5/2))
        f3 = ((rho(0)+rho(t))*(rho(Q_sq)+rho(t)))**(-2)
    elif ff=='+':
        f1 = np.sqrt(2*eta/(3*np.pi*(t_cut-t_0)))
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
    G = np.array([[1/(1-z(t1)*z(t2)) for t2 in t_vec] for t1 in t_vec], 
            dtype=prec)
    return G

def del_RC(matrix, row, col):
    matrix = np.delete(matrix, row, axis=0)
    matrix = np.delete(matrix, col, axis=1)
    return matrix

def M11(known_ts, known_ffs, X, **kwargs):
    N = len(known_ts)
    phi_ff_vec = np.array(phi(np.array(known_ts),**kwargs)*known_ffs,
                        dtype=prec)

    top_line = np.hstack((X, phi_ff_vec))
    g00 = G(known_ts)
    bottom = np.array([np.hstack((phi_ff_vec[i], g00[i,:])) for i in range(N)],
                        dtype=prec)

    M11 = np.vstack((top_line, bottom))
    return M11

from numpy.linalg import det
def bounds(unknown_t, known_ts, known_ffs, X, prnt=False, **kwargs):
    N = len(known_ts)
    g = G(np.hstack((unknown_t, known_ts)))

    alpha = det(G(known_ts))
    beta = np.sum(np.array([((-1)**(j+1))*phi(known_ts[j],**kwargs
                            )*known_ffs[j]*det(del_RC(g,0,j+1))
                            for j in range(N)], dtype=prec))
    gamma_mtx = np.array([[((-1)**(i+j))*phi(known_ts[i],**kwargs)*phi(known_ts[j],
                        **kwargs)*known_ffs[i]*known_ffs[j]*det(del_RC(g,i+1,j+1)) 
                            for j in range(N)] for i in range(N)], dtype=prec)
    gamma = X*det(g)-np.sum(gamma_mtx)
    disc = det(M11(known_ts, known_ffs, X, **kwargs))*det(g)

    #disc = ((beta**2)+(alpha*gamma))
    if disc<0:
        disc=0
    upper_bound = (-beta+(disc**0.5))/(alpha*phi(unknown_t,**kwargs))
    lower_bound = (-beta-(disc**0.5))/(alpha*phi(unknown_t,**kwargs))

    if prnt:
        print((lower_bound, upper_bound),'\ndiscriminant:'+str(disc))
    return [lower_bound, upper_bound]

def bootstrap(var, cov, K=100, **kwargs):
    np.random.seed(1)
    samples = np.random.multivariate_normal(var, cov, K)
    return samples



nplus=2 # number of input values for f+
nzero=3 # number of input values for f0
import h5py as h5
path='/home/rm/PhD/disp/BstoK_data/'
with h5.File(path+'zfit_data_BstoK.h5', 'r') as f:
    gp=f.get('BstoK_refdata_qsqmin_17.50_Np{:d}_Nz{:d}'.format(nplus,nzero))
    known_ts=np.array(gp['qsqref'])
    known_ffs=np.array(gp['central'])
    COV_input=np.array(gp['tot_cov'])
import pickle
pickle.dump([known_ts, known_ffs, COV_input], open(f'BstoK_Data_{nplus}x{nzero}.p','wb'))
#import pickle
#[known_ts, known_ffs, COV_input] = pickle.load(open('BstoK_Data_3x3.p','rb'))

g00 = G(known_ts[:nplus])
print('g00 is pos def') if det(g00)>0 else print('g00 is not pos def')

X_zero, X_plus = 1.48e-2, 6.03e-4 
X_zero_err, X_plus_err = X_zero/20, X_plus/20 # made 5% error 
X, X_err = np.array([X_zero,X_plus]), np.array([X_zero_err, X_plus_err])

dict_zero = {'ff':'0'}
dict_plus = {'t_p':5.3247**2, 'ff':'+'}

t_range = np.arange(0,known_ts[0]+0.1,0.1)
N_boot = 100
N_0 = 10
samples = bootstrap(known_ffs, COV_input, K=N_boot)
samples_X = bootstrap(X,np.diag(X_err)**2,K=N_boot)

zero_dist = {str(t):{'up':[], 'lo':[]} for t in t_range}
plus_dist = {str(t):{'up':[], 'lo':[]} for t in t_range} 
accepted_idx = []

import time
t1 = time.time()
from tqdm import tqdm
for k in tqdm(range(N_boot)):
    m11_plus = M11(known_ts[:nplus], samples[k,:nplus], samples_X[k,1], **dict_plus)
    m11_zero = M11(known_ts[nplus:], samples[k,nplus:], samples_X[k,0], **dict_zero)
    #print(det(m11_plus), det(m11_zero))
    if det(m11_zero)>0 and det(m11_plus)>0:
        [zero_low, zero_up] = bounds(0, known_ts[nplus:], samples[k,nplus:], samples_X[k,0],
                                     **dict_zero)
        [plus_low, plus_up] = bounds(0, known_ts[:nplus], samples[k,:nplus], samples_X[k,1],
                                     **dict_plus)
        if zero_up>plus_low and plus_up>zero_low: # kinematical constraint
            accepted_idx.append(k)
            np.random.seed(1)
            f0s = np.random.uniform(max(zero_low,plus_low), min(zero_up,plus_up), N_0)
            known_ts_0_plus = np.hstack((known_ts[:nplus],0))
            known_ts_0_zero = np.hstack((known_ts[nplus:],0))
            
            for t in t_range:
                #for n in range(N_0):
                #    m11 = M11(known_ts_0,np.hstack((samples[k,:3],f0s[n])),
                #            samples_X[k,0], **dict_zero)
                #    g = G(np.hstack((t,known_ts_0)))
                #    print('\nbtsp:'+str(k), 't:'+str(t), 'n_0:'+str(n),
                #            'f0:'+str(f0s[n]),
                #            '\ndetM11*detG:'+str(det(m11)*det(g)),
                #                '\neigvals:'+str(np.hstack((np.linalg.eigvals(m11),
                #                                    np.linalg.eigvals(g)))),
                #            '\nbounds:')
                #    bounds(t,known_ts_0, np.hstack((samples[k,:3],f0s[n])),
                #            samples_X[k,0], prnt=True, **dict_zero)
                zero_bounds = np.array([bounds(t,known_ts_0_zero,
                    np.hstack((samples[k,nplus:],f0s[n])),
                                samples_X[k,0],**dict_zero)
                                for n in range(N_0)])

                plus_bounds = np.array([bounds(t,known_ts_0_plus,
                                np.hstack((samples[k,:nplus],f0s[n])),
                                samples_X[k,1],**dict_plus)
                                for n in range(N_0)])

                zero_dist[str(t)]['lo'].append(np.min(zero_bounds[:,0]))
                zero_dist[str(t)]['up'].append(np.max(zero_bounds[:,1]))
                plus_dist[str(t)]['lo'].append(np.min(plus_bounds[:,0]))
                plus_dist[str(t)]['up'].append(np.max(plus_bounds[:,1]))
print('Time taken:',time.time()-t1)

for t in t_range:
    if np.isnan(zero_dist[str(t)]['up']).any():
        del zero_dist[str(t)]
        del plus_dist[str(t)]
#import pickle
#zero_dist = pickle.load(open('zero_dist_20000x50x0.1.p','rb'))
#plus_dist = pickle.load(open('plus_dist_20000x50x0.1.p','rb'))

def st_dev(data, mean=None, **kwargs):                                                                                                                                                                
    '''standard deviation function - finds stdev around data mean or mean
    provided as input'''
    n = len(data)
    if mean.any()==None:
        mean = np.mean(data)
    return (((data-mean).dot(data-mean))/n)**0.5

accepted_ts = np.array(list(zero_dist.keys())).astype(float)
bnds = np.zeros(shape=(len(accepted_ts),2))
bnds_err = np.zeros(shape=(len(accepted_ts),2))
def final_bounds(dist):
    f = np.zeros(shape=(len(accepted_ts)))
    errs = np.zeros(shape=(len(accepted_ts)))
    for i in range(len(accepted_ts)):
        t = accepted_ts[i]
        lows = list(dist[str(t)]['lo'])
        ups = list(dist[str(t)]['up'])
        f_lo = np.mean(lows)
        f_lo_err = st_dev(lows, mean=f_lo)
        f_up = np.mean(ups)
        f_up_err = st_dev(ups, mean=f_up)

        bnds[i,:] = [f_lo, f_up]
        bnds_err[i,:] = [f_lo_err, f_up_err]
        
        N = len(lows)
        rho = np.sum(np.array([(lows[i]-f_lo)*(ups[i]-f_up) for i in range(N)]))/(N-1)

        f_t = (f_lo + f_up)/2
        var_t = ((f_up-f_lo)**2)/12
        var_t = var_t + ((f_lo_err**2)+(f_up_err**2)+(rho))/3

        f[i] = f_t
        errs[i] = var_t**0.5
    return f, errs

f_zero, f_zero_errs = final_bounds(zero_dist)
f_plus, f_plus_errs = final_bounds(plus_dist)

plt.figure()
plt.plot(t_range, f_zero, c='b')
plt.plot(t_range, f_plus, c='g')
plt.fill_between(accepted_ts, f_zero+f_zero_errs, f_zero-f_zero_errs,
                alpha=0.2, color='b')
plt.fill_between(accepted_ts, f_plus+f_plus_errs, f_plus-f_plus_errs,
                alpha=0.2, color='g')
plt.errorbar(known_ts[nplus:], known_ffs[nplus:], yerr=np.diag(COV_input)[nplus:]**0.5, fmt='o', 
            capsize=4, c='r')
plt.errorbar(known_ts[:nplus], known_ffs[:nplus], yerr=np.diag(COV_input)[:nplus]**0.5, fmt='o', 
            capsize=4, c='r')

plt.legend(['f0','f+'])
plt.xlabel('t')

#plt.figure()
#plt.plot(accepted_ts, bnds[:,0], c='b')
#plt.fill_between(accepted_ts, bnds[:,0]+bnds_err[:,0], bnds[:,0]-bnds_err[:,0],
#                alpha=0.2, color='b')
#plt.plot(accepted_ts, bnds[:,1], c='r')
#plt.fill_between(accepted_ts, bnds[:,1]+bnds_err[:,1], bnds[:,1]-bnds_err[:,1],
#                alpha=0.2, color='r')
#plt.legend(['lo','up'])
#plt.xlabel('t')

plt.show()



        









































