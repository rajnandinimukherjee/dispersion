import numpy as np 
import h5py
import masses

n=10
# predict BstoK formfactors and covariance matrix for any choice of reference kineamtics based on HMChPT
def ff_E(Evec,pole,coeff):
  # construct ff from HMChPT in continuum limit
  return [1./(E+pole)*np.sum([E**i*coeff[i] for i in range(len(coeff))]) for E in Evec]
def cov_ff_p0(Evec_p,Evec_0,C,Np,N0,pole_p,pole_0):
  # construct covariance matrix for ff from HMChPT in continuum limit
  Y_E_p_vec   	= lambda E_p: np.r_[ np.array([1./(E_p+pole_p)*E_p**i for i in range(Np)])]
  Y_E_0_vec   	= lambda E_0: np.r_[ np.array([1./(E_0+pole_0)*E_0**i for i in range(N0)])]
  Cpp		= np.array([[np.dot(Y_E_p_vec(E1),np.dot(C[:Np,:Np],Y_E_p_vec(E2)))
					for E1 in Evec_p] for E2 in Evec_p])
  C00		= np.array([[np.dot(Y_E_0_vec(E1),np.dot(C[Np:,Np:],Y_E_0_vec(E2)))
					for E1 in Evec_0] for E2 in Evec_0])
  Cp0		= np.array([[np.dot(Y_E_p_vec(E1),np.dot(C[:Np:,Np:],Y_E_0_vec(E2)))
					for E1 in Evec_p] for E2 in Evec_0])
  M0		= np.r_['-1',Cpp  ,Cp0.T]
  M1		= np.r_['-1',Cp0  ,C00  ]
  M		= np.r_[M0,M1]
  return M

# define kinematics
mKphys		= masses.mK 
mBsphys		= masses.mBs
#
#qsq_refK	= np.array([23.7283556,22.11456,20.07895,17.5000000]) # you can choose this freely 
qsq_refK	= np.linspace(17.5,23.7283556,num=n) 
#
ksq_refK 	= (mBsphys**4+(mKphys**2-qsq_refK)**2-2*mBsphys**2*(mKphys**2+qsq_refK))/(4*mBsphys**2)
ErefK 	 	= np.sqrt(mKphys**2+ksq_refK)
Deltapar	= + 0.263
Deltaperp	= - 0.0416


f=h5py.File('data_all/BstoK_ref_ff_dat.hdf5','r')
cp_BstoK=np.array(f.get('cp'))
c0_BstoK=np.array(f.get('c0'))
Cp0_BstoK=np.array(f.get('Cp0'))
fp_BstoK 	= np.array(ff_E(ErefK,Deltaperp,cp_BstoK))
f0_BstoK 	= np.array(ff_E(ErefK,Deltapar ,c0_BstoK))
ff_ref		= np.r_[ fp_BstoK, f0_BstoK]
Cp0_ref 	= cov_ff_p0(ErefK,ErefK,Cp0_BstoK,2,3,Deltaperp,Deltapar)

import pickle
pickle.dump([qsq_refK, ff_ref, Cp0_ref], open(f'BstoK_Data_{n}x{n}.p','wb'))
# some IO
#print 'ff results ',np.r_[fp_BstoK,f0_BstoK]
#print 'dff results' np.sqrt(np.diag(Cp0_ref))





