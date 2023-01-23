#!/usr/bin/python3
'''
-------------------------------------------------------------------------------------------------------------------
21st January 2023.
Task list.
1. Despike data. There are few spikes in the data that cause error in numChannels estimate.
2. Double check forward/backward. The likelihood must increase monotonically.
3. Revise/rectify Viterbi. It should give the initial state probability distribution which then feeds into forward/backward.
Revised Viterbi must reduce cost function at each iteration, and increase likelihood at each iteration.
4. Simulate synthetic data to show code reliability.
5. Revise PSD for sigma^2 and lambda_2. In theory, there should be a sigma for each currentLevel.
6. Do a check point so that the Beluga/Graham simulations can be continued.
7. Do simulated annealing instead of steepest gradient (see FIP 2018 for starting point).
8. Some parameter either as argv[1] or otherwise that picks an abf file from cell*/{Control,Pressure}/*

-------------------------------------------------------------------------------------------------------------------
8th Jan 2023. I did a flow chart and bugs list. Now is the time to put the code into iterations.

v12. This version is cleaned up while I get used to the recursion.
SRK.
17th Dec. 2022 started.
To program the MLE for Galina's data.
1. Simulate the baseline trace using zeta, rho, kappa, and numChannels.
2. Analyse the data again together with Galina.
3. Add noise to (1).
4. MLE, Baum forward-backward procedure.
5. MLE, Vertibri procedure.
This is the binary Markov chain by Chung et al. that provides potential coupling constant.
Since the coupling is critical, there have to be at least 2 channels.


In this version:
1) Noisy current using full P matrix is done.
2) The aggregated L, R, and A are available.

Jan 5, 2023. Worked out Aagg from the recursion which may allow many more channels to 
be simulated on the Mac.

January 13, 2023.
this program now has the basics. 
a) Estimation of zeta, rho, sigmaSquared, numChannels, pi_open, single_channel_current from the data.
b) forward/backward algorithm.
c) Baum-Welch algorithm.
d) Viterbi algorithm.

It is now time to put it into iterations.

'''

################ includes/imports.
import pyabf # this needs to be added to the standard headers file.
import matplotlib.pyplot as plt
import numpy as np
import statistics

# Standard imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import math
import random
from random import seed
import os
import itertools
sys.setrecursionlimit(1000000) # the default depth is 1000, I made it 1M for now.
import scipy
from scipy.stats import moment # to do the central moments.

# see: https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.ccovf.html
import statsmodels.tsa.api as smt # alongwith the np, this is for autocovariance.
from numpy.fft import fft, ifft
import lcapy as lc # get z transform directly without ffr and autocovariance.
from lcapy.discretetime import n
from matplotlib import mlab # for the PSD from time series.
from scipy import optimize # for the PSD Syomega curve fitting.

import time

# for reading in file accoring to input.
# import linecache
import linecache as lc

######### global variables.
debg=1

################ custom functions. ##################################################
#################################################################################
def per(n):
	pdim = np.power(2 , n)
	for i in range(1<<n):
		s=bin(i)[2:]
		s='0'*(n-len(s))+s
		print(s)
#	print (map(int,list(s)))
#################################################################################
#################################################################################
# this is the bjyk array for emission probability.
def emissionProbablities(myData, myCurrentLevels, numChannels, sigma): # myI has numChannels+1 levels.
	intT = len(myData[:,2])
#	print(intT)
#	sys.exit()
	bjyk = np.zeros([numChannels+1, intT])
# Time loop to calculate bjyk. bj is a vector. At each time k, b is a vector calculated for each state q_j .
	for k in range(0, intT):
		yk = myData[k,2]	
		for j in range(0, numChannels+1):
# pass every state to this yk.
#			myI = float(j) * s_est # this is the dimensional q_j.
			myI = myCurrentLevels[j,0]
#			print(myI)
# A factor of mean may  be missing from the below. myI -> myI + mean where mean is during the clamp.
			bjyk[j,k] = (1.0/(np.sqrt(2.0*math.pi)*sigma)*np.exp(-0.5*(yk-myI)*(yk-myI)/(sigma*sigma)))
			if bjyk[j,k]<=0.0: # error check.
				print('the bjyk is error. exiting.')
				sys.exit()

# normalize to 1 if any row has a bjyk more than 1.
	for k in range(0, intT):
		if np.max(bjyk[:,k])>1.0:
			bjyk[:,k] = bjyk[:,k]/np.max(bjyk[:,k])

	return bjyk
#################################################################################
#################################################################################
# see: https://stackoverflow.com/questions/20110590/how-to-calculate-auto-covariance-in-python
def myautocovariance(Xi, N, k, Xs): # data array, total size of Xi data array, k is the lag, Xs is the mean.
	autoCov = 0
	for i in np.arange(0, N-k):
		autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
	return (1.0/(N-1.0))*autoCov

#################################################################################
#################################################################################
def mySyomega(omega, sigmaSquared, lambdaa):
	global DELTAT
	global Npiopics2Squared
	locsyomega = sigmaSquared + Npiopics2Squared * (1.0 - lambdaa*lambdaa)/(1.0 + lambdaa*lambdaa + 2.0*lambdaa*np.cos(omega*DELTAT))
	return locsyomega
#################################################################################
#################################################################################
def myViterbi(Aagg, myState, bjyk, myData):
	"""
	Arguments:
	Aggregate matrix, Aagg or Aagg_est.
	Initial state. Right now, it is just one state.
	emission matrix.
	the expt. data.
	
	Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D (np.ndarray): Accumulated probability matrix
        E (np.ndarray): Backtracking matrix
        
        see: MarkovModelReconstruction_of_ionic_single-channel_currents_based_on_hidden_Markov_model.pdf
        see: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
        The likelihood has to be log, else you get a trivial estimate.
	"""
		
	numStates 						= Aagg.shape[0]
	intT 								= len(bjyk[0,:])
	tiny 								= np.finfo(0.).tiny
#	print(tiny)
	# declare arrays.
	myInitialStates 					= np.zeros([numStates, 1])
	myIncomingState 					= int(myState)
	myInitialStates[myIncomingState, 0] 	= myIncomingState

	Aagg_log 						= np.log(Aagg + tiny)
	bjyk_log 							= np.log(bjyk + tiny)
	
	D 								= np.zeros([numStates, intT])
	E 								= np.zeros([numStates, intT-1])

	D_log 							= np.zeros([numStates, intT])
	myInitialStatesLog 					= np.log(myInitialStates + tiny)
	
	# first intT = 0 entry of D.
	for i in range(0, numStates):
		D[i, 0] 		= myInitialStates[i,0]*bjyk[i,0]
		D_log[i,0] 	= myInitialStatesLog[i, 0] + bjyk_log[i, 0]
	
	# calculate D and E.
	for n in range(1, intT):
		for i in range(0, numStates):
			temp_product 	= np.multiply(Aagg[:, i], D[:, n-1])
			temp_sum 		= Aagg_log[:, i], D_log[:, n-1]
#			print(temp_product) # this is a vector as long as numStates. I need to use the log version because this is 0 for larger intT.
			D[i, n] 			= np.max(temp_product) * bjyk[i, 0]
#			E[i, n-1] 			= np.argmax(temp_product)
			
			D_log[i, n] 		= np.max(temp_sum) + bjyk_log[i, 0]
			E[i, n-1] 			= np.argmax(temp_sum)
	
	# Backtracking
	S_opt 			= np.zeros([intT, 1])
#	S_opt[-1] 		= np.argmax(D[:, -1])
	S_opt[-1] 		= np.argmax(D_log[:, -1])
	for n in range(intT-2, -1, -1):
		S_opt[n, 0] 	= E[int(S_opt[n+1, 0]), n]

	plt.plot(S_opt)
	plt.show()
	
	print('Inside myViterbi draft, this is not working yet.')
	print(numStates, intT, myState)
#################################################################################
def coupledMatrices(zeta, rho, numChannels):
# define P(C) as an aggregate states matrix. This matrix does not do xi and eta as Rutford does, but takes zeta/rho given by Chung 1996b.
	ACp 							= np.zeros([numChannels+1, numChannels+1])
# derivatives.
	dACpdzeta 	= np.zeros([numChannels+1, numChannels+1])
	dACpdrho 	= np.zeros([numChannels+1, numChannels+1])
#	dACpdk 		= np.zeros([numChannels+1, numChannels+1]) # not needed. dAcpdk = Acp.
				
	for L in range(0, numChannels+1):
		ACp[L,0] 					= 0.5 # xi. This is 1-delta.
		ACp[L,numChannels] 			= 0.5 # delta. This is delta.

	ACp[0,0] 						= zeta # this does not have to be zeta/rho, can be another estimatable paramter.
	ACp[0,numChannels] 				= 1.0 - zeta
	ACp[numChannels,0] 				= 1.0 - rho
	ACp[numChannels, numChannels] 	= rho # P, coupled in aggregated state.

	dACpdzeta[0,0] 						=  1.0 # derivative of coupled aggregate matrix w.r.t. zeta.
	dACpdzeta[0,numChannels] 				= -1.0

	dACpdrho[numChannels,0] 				= -1.0
	dACpdrho[numChannels,numChannels] 	=  1.0 # derivative of coupled aggregate matrix w.r.t. rho.

	return ACp, dACpdzeta, dACpdrho

#################################################################################
#################################################################################
def independentMatrices2(zeta, rho, numChannels):
	AInd 		= np.zeros([numChannels+1, numChannels+1])
	dAInddrho 	= np.zeros([numChannels+1, numChannels+1])	
	dAInddzeta 	= np.zeros([numChannels+1, numChannels+1])	
	
	s = numChannels # for notation consistency with Biometrics 253*.pdf equation 6.
	for i in range(0, s+1): # rows.
		for j in range(0, s+1):
			min_k 	= max(i-j, 0)
			max_k 	= min(s-j, i)
#			print(i, j, min_k, max_k)
			aij 		= 0.0
			daijdrho 	= 0.0
			daijdzeta	= 0.0
			for k in range(min_k, max_k+1): # this is from Biometrics paper.
				alpha_sijk = float( math.comb(i, k)*math.comb(s-i, k+j-i) )
				asijkrho	 = math.pow( 1.0 - rho, k)	*math.pow(rho, i-k)
				asijkzeta	 = math.pow(1.0 - zeta, k+j-i)	*math.pow(zeta, s-j-k)
				
				aij = aij + alpha_sijk*asijkrho*asijkzeta
				
				dasijkdzeta = -(k+j-i)*math.pow(1.0 - zeta, k+j-i-1.0)*math.pow(zeta, s-j-k)+(s-j-k)*math.pow(1.0 - zeta, k+j-i)	*math.pow(zeta, s-j-k-1.0)
				daijdzeta = daijdzeta + alpha_sijk*asijkrho*dasijkdzeta
				
				dasijkdrho = -k*math.pow(1.0-rho, k-1.0)*math.pow(rho, i-k) + math.pow(1.0-rho, k)*(i-k)*math.pow(rho, i-k-1.0)
				daijdrho = daijdrho + alpha_sijk*asijkzeta*dasijkdrho

			AInd[i,j] 			= aij
			dAInddrho[i,j] 		= daijdrho
			dAInddzeta[i,j] 	= daijdzeta
#			print(i, j, AInd[i,j])

	if debg==2:
		print(AInd)
		print(dAInddrho)
		print(dAInddzeta)
		sys.exit()

	return AInd, dAInddzeta, dAInddrho

#################################################################################
def independentMatrices(zeta, rho, numChannels):
# 5 Jan 2023. Do the Aindependent using the recursion. Since the derivation is mysterious, double check by comparing it with
# the Kronecker product for a few values of numChannels. numChannels 1 to 10.
# define L = 1 AIndprev. This is to be compared to L (PInd ) R for all (1 to 10) values of L.
# Notes: The A(Ind) worked out using the Chung recursion is not the same as PInd done using the Kronecker product.
# This may be because the matrix L (and potentially R) that the recursion assumes is not as I have defined above.
# Regardless, the cumulative sum of this AInd works out to 1, indicating that it can at least act as a placeholder if not
# be the actual transition matrix that I need. Strategy is to use this, and try to derive the Chung recursion for AInd
# so that the calculation below can be verified.
# MarkovModelChung1996b.pdf , MarkovModelChung1996.pdf , MarkovModelChung1992.pdf .
	for L in range(1, numChannels+1):
#		print('constructing AInd, %d of %d' % (L, numChannels) )
		if L==1:
			AIndprev 		= np.zeros([2, 2])
			dAIndprevdzeta	= np.zeros([2, 2])
			dAIndprevdrho	= np.zeros([2, 2])			

			# define the independent matrix.
			AIndprev[0, 0] 	= zeta
			AIndprev[0, 1] 	= 1.0 - zeta
			AIndprev[1, 0] 	= 1.0 - rho
			AIndprev[1, 1] 	= rho
			
			# define the derivative of independent matrix w.r.t. zeta.
			dAIndprevdzeta[0, 0] 	=  1.0
			dAIndprevdzeta[0, 1] 	= -1.0
			dAIndprevdzeta[1, 0] 	= 0.0
			dAIndprevdzeta[1, 1] 	= 0.0
			
			# define the derivative of independent matrix w.r.t. rho.
			dAIndprevdzeta[0, 0] 	=  0.0
			dAIndprevdzeta[0, 1] 	= 0.0
			dAIndprevdzeta[1, 0] 	= -1.0
			dAIndprevdzeta[1, 1] 	=  1.0

			AInd 		= np.copy(AIndprev)
			dAInddzeta 	= np.copy(dAIndprevdzeta)
			dAInddrho 	= np.copy(dAIndprevdrho)
									
			continue # this continue will stop any destroy/delete of AIndprev if numChannels==1.
		if L>=2:
			del AInd
			del dAInddzeta
			del dAInddrho
		AInd 		= np.zeros([L+1, L+1]) # this goes from L = 2 to numChannels.
		dAInddzeta 	= np.zeros([L+1, L+1]) # this goes from L = 2 to numChannels.
		dAInddrho 	= np.zeros([L+1, L+1]) # this goes from L = 2 to numChannels.
	
		for m in range(0, L+1):
			for n in range(0, L+1):
				if m<=(L-1):
					if n==0:
						AInd[m,n] 		= zeta*AIndprev[m,n]
						dAInddzeta[m,n] 	= AIndprev[m,n] 		+ zeta*dAIndprevdzeta[m,n]
						dAInddrho[m,n] 	= 					   zeta*dAIndprevdrho[m,n]
																		
					elif n>=1 and n<=(L-1):
						AInd[m,n] 		= zeta*AIndprev[m,n] + (1.0 - zeta)*AIndprev[m,n-1]
						dAInddzeta[m,n] 	= AIndprev[m,n] + zeta*dAIndprevdzeta[m,n] - AIndprev[m,n-1] + (1.0 - zeta)*dAIndprevdzeta[m,n-1]
						dAInddrho[m,n] 	= zeta*dAIndprevdrho[m,n] + (1.0 - zeta)*dAIndprevdrho[m,n-1]												
					elif n==L:
						AInd[m,n] 		= (1.0 - zeta)*AIndprev[m,n-1]
						dAInddzeta[m,n] 	= - AIndprev[m,n-1] + (1.0 - zeta)*dAIndprevdzeta[m,n-1]	
						dAInddrho[m,n] 	= (1.0 - zeta)*dAIndprevdrho[m,n-1]										
				elif m==L:
					if n==0:
						AInd[m,n] 		= (1.0 - rho)*AIndprev[m-1,n]
						dAInddzeta[m,n] 	= (1.0 - rho)*dAIndprevdzeta[m-1,n]
						dAInddrho[m,n] 	= - AIndprev[m-1,n] + (1.0 - rho)*dAIndprevdrho[m-1,n]											
					elif n>=1 and n<=(L-1):
						AInd[m,n] 		= (1.0 - rho)*AIndprev[m-1,n] + rho * AIndprev[m-1,n-1]
						dAInddzeta[m,n] 	= (1.0 - rho)*dAIndprevdzeta[m-1,n] + rho * dAIndprevdzeta[m-1,n-1]
						dAInddrho[m,n] 	= - AIndprev[m-1,n] + (1.0 - rho)*dAIndprevdrho[m-1,n] + AIndprev[m-1,n-1] + rho * dAIndprevdrho[m-1,n-1]																
					elif n==L:
						AInd[m,n] 		=  rho * 	AIndprev[m-1,n-1]
						dAInddzeta[m,n] 	=  rho * 	dAIndprevdzeta[m-1,n-1]
						dAInddrho[m,n] 	=  		AIndprev[m-1,n-1] 			+ rho * dAIndprevdrho[m-1,n-1]

		del AIndprev
		del dAIndprevdzeta
		del dAIndprevdrho
		AIndprev 		= np.zeros([L+1, L+1])
		dAIndprevzeta 	= np.zeros([L+1, L+1])
		dAIndprevrho 		= np.zeros([L+1, L+1])				
#		print('L = %d' % L)
		AIndprev 		= np.copy(AInd)
		dAIndprevdzeta 	= np.copy(dAInddzeta)
		dAIndprevdrho 	= np.copy(dAInddrho)

#	print('The Aagg using recursion given by Chung papers:')
#	print(AInd)
#	print('each aggregate matrix must have a cumulative sum to 1:')
#	print(np.cumsum(AInd, axis=1))

	return AInd, dAInddzeta, dAInddrho
#################################################################################
def 	forwardBackward(bjyk, pi_init, Aagg):

	numChannels = len(pi_init[:,0])-1
	intT = len(bjyk[0,:])
#	print(numChannels, intT)

	myAlpha 	= np.zeros([numChannels+1, intT]) # see Chung 1990.
	myBeta 		= np.zeros([numChannels+1, intT])
	c_sum 		= np.zeros([intT, 1])
	mypi 		= np.zeros([numChannels+1, 1])	
	
	histogramT 	= np.zeros([numChannels+1, 1]) 
	gammaki 	= np.zeros([numChannels+1, intT])		
	myXi 		= np.zeros([numChannels+1, numChannels+1, intT])


# the forward/backward and Baum Welch optimization starts here.
# MarkovModelChung1996b.pdf , MarkovModelChung1996.pdf , MarkovModelChung1992.pdf , also MarkovModelChung1991.pdf
# assign the initial conditions, pi_init. pi_init is of size numChannels + 1.
#	j 				= myStates[0, 0] # the intial value in myStates. Done in the calling function.
#	pi_init[ int(j) , 0 ] 	= 1.0 # sum of all pi_init has to be 1.

	'''
	Applying Hidden Markov Models to the Analysis of Single Ion Channel Activity, Biophysical Journal 82(4) 1930–1942. page 2:
	Likelihood is defined as the probability of observing a particular data set given a model. The evaluation of the likelihood of HMMs has been made practical by an algo- rithm called the forward-backward procedure. Optimization of the parameters of the model is aided by the Baum-Welch procedure, which through iteration causes a maximum of the likelihood to be approached. These algorithms have been well described in the speech-recognition literature (Baum et al., 1970; Liporace, 1982; Rabiner, 1989).
	page 3: 
	...the probability of the observations given the state se- quence (Eq. 5) is seen to be the product of T Gaussians (i.e. bjyk over k).
	Note how multiplying numbers between 0 and 1 over intT (around 2k) will lead to a VERY small number, unless
	they are all really close to 1.
	There are also as many means (L+1 or numChannels+1) as there are states (L+1 or numChannels+1).
	
	Also see: Hidden_Markov_modeling_of_ion_.pdf starting page 34.
	'''
	# time k = 0.
	k 					= 0
	c_sum[k,0] 			= 0.0 
	for j in range(0, numChannels+1):
		myAlpha[j, 0] 		= pi_init[j,0] * bjyk[j, 0] # p271, Chung 1990.
		myBeta[j,intT-1] 	= 1.0
		c_sum[k,0] 		= c_sum[k,0] + myAlpha[j, 0]
# normalise myAlpha so far using c_sum.
	if c_sum[k,0] <= 0.0:
		print('c_sum didnt work for alpha_1. exiting.')
		sys.exit()
	for j in range(0, numChannels+1):
		myAlpha[j, 0] 		= myAlpha[j,0] / c_sum[k,0]
#		print(j, myAlpha[j,0], pi_init[j, 0], bjyk[j,0], myBeta[j, intT-1])

# define the alpha loop. Although Chung 1990, Sigworth 2000, and others use the word "recursion",
# the formula is a simple nested loop where this row of alpha depends only on the previous row.	
# normalise w.r.t. sum of myAlpha over all j. The normalisation is close to 1, but not exactly.
	for k in range(1, intT):
		for j in range(0, numChannels+1):
			summ = 0.0
			for i in range(0, numChannels+1):
				summ = summ + myAlpha[i,k-1] * Aagg[i,j]
			myAlpha[j,k] = bjyk[j, k] * summ # make sure bjyk is defined as you think it should be.
#		print(np.sum(myAlpha[:,k]))
		c_sum[k,0] = np.sum(myAlpha[:,k])
		if c_sum[k,0] <= 0.0:
			print('c_sum at %d time is 0 or less, exiting' % k)
			sys.exit()
		for j in range(0, numChannels+1):
			myAlpha[j,k] = myAlpha[j,k] / c_sum[k,0] # the max myAlpha j is not strictly 1.
#		print(myAlpha[:,k])

# the myAlpha was not strictly 1 after normalization for some given j. So do a hack.
# fix the machine precision error: hack, maybe there is a different way to do this.
	for k in range(1, intT):
		maxAlpha = np.max(myAlpha[:,k])
#		print(maxAlpha)
		if maxAlpha <= 0.0:
			print('maxAlpha turned out wrong, exiting.')
			sys.exit()
		myAlpha[:,k] = myAlpha[:,k]/maxAlpha
#		print(myAlpha[:,k])
#	sys.exit()

# define the beta. The normalisation is w.r.t. alpha.
	for k in range(intT - 2, -1, -1): # the second limit is not included. With -1, it runs up to k = 0.
		for i in range(0, numChannels+1):
			summ = 0.0
			for j in range(0, numChannels+1):
				summ = summ + Aagg[i,j] * bjyk[j,k+1] * myBeta[j,k+1] # see MarkovModelChung1990.pdf page 8 eq 2.
			if c_sum[k+1,0]<=0.0:
				print('bad c_sum in myBeta, exiting.')
				sys.exit()
			myBeta[i,k] = summ / c_sum[k+1, 0]
#		print(myBeta[:,k])

# The myBeta itself is not strictly 1 in expected places. So force it to be normalized in each row. This may be a hack.		
	for k in range(intT - 2, -1, -1): # with -1, it runs up to k = 0.
		myMaxBeta = np.max(myBeta[:,k])
		if myMaxBeta <= 0.0:
			print('the max beta is wrong, exiting.')
			sys.exit()
		myBeta[:,k] = myBeta[:,k] / myMaxBeta

# Likelihood is sum of last row of myAlpha.
# Equation 3 of Chung 1990. The likelihood.
	LT = 0.0
	for i in range(0, numChannels+1):
		LT = LT + myAlpha[i, intT-1]

# gammaki, Chung 1990 formula 4. Formula 4 says its till intT - 1, but then the last gammaki is left dangling. So I changed it to intT
	for k in range(0, intT):
		for i in range(0, numChannels+1):
			summ = 0.0
			for j in range(0, numChannels+1):
				summ = summ + myAlpha[j,k] * myBeta[j,k]
#			print(summ)
			if summ<=0.0:
				print('bad sum alpha * beta, this should never happen. Exiting.')
				sys.exit()
			else:
				gammaki[i,k] = myAlpha[i,k] * myBeta[i,k] / summ

# Is the sum of gammaki equal to 1 for each discrete time k? Answer: Yes, it is.
	if 1==2:
		for k in range(0, intT):
			print(np.sum(gammaki[:,k]))
#	sys.exit()

# histogram.
	for i in range(0, numChannels+1):
		summ = 0.0
		for k in range(0, intT-1):
			summ = summ + gammaki[i, k]
		histogramT[i] = summ / float(intT) # right now, it will turn out 0 as gammaki needs to be revised.
	
# Xi, its a 3D array.
	for k in range(0, intT-1):
		for j in range(0, numChannels+1):
			for i in range(0, numChannels+1):
				if LT<=0.0:
					print('The LT is 0, cannot work out myXi, exiting.')
					sys.exit()
				myXi[i, j, k] = myAlpha[i, k] * Aagg[i,j] * bjyk[j, k+1] * myBeta[j, k+1] / LT 
#	print(myXi)

# sum myXi(i,j, 1) over j is my new initial state probablity.
	for i in range(0, numChannels+1):
		mypi[i, 0] = np.sum(myXi[i,:,0]) # eq. 11, MarkovModelChung1990.pdf .
#	print(mypi) # initial state probablities. They are really small as of now.

	return myAlpha, myBeta, LT, gammaki, histogramT, myXi, mypi
#################################################################################
def estimatematrixCurrentLevels(myXi, gammaki, myData):
	intT 				= len(gammaki[0,:])
	numChannels 	= len(gammaki[:,0])-1

	Aagg_est 		= np.zeros([numChannels+1, numChannels+1], 	dtype=float)
	myCurrentLevels_est 	= np.zeros([numChannels+1, 1])
	
	sumk = 0.0
	sumkj = 0.0
	for i in range(0, numChannels+1):
		for j in range(0, numChannels+1):
			sumk 	= np.sum(myXi[i,j,0:(intT-2)])
			sumkj 	= np.sum(myXi[i,:,0:(intT-2)]) # eq 10 in MarkovModelChung1990.pdf .
#			print(sumk, sumkj, sumk/sumkj)
			Aagg_est[i, j] = sumk/sumkj # eq 10 in MarkovModelChung1990.pdf . Potential division by 0 for large numChannels.

#	print(Aagg)
#	print(Aagg_est)
#	print(Aagg - Aagg_est)
	
	Aagg_est_csum 	= np.cumsum(Aagg_est, axis=1)
#	print(Aagg_est_csum) # this is really busy - most entries are 1, or close to 1.

# new current levels.
# you may need to hold myCurrentLevels_est[0, 0] to be zero by default.
#	for i in range(0, numChannels+1):
	for i in range(1, numChannels+1):
		sumgami 	= np.sum(gammaki[i,:])
		sumgamiyk 	= np.sum(gammaki[i,:] * myData[:,2])
# the predicited current values come within range of the experimental trace. See the figs and estimated values.
#		print(i, sumgami, sumgamiyk, sumgamiyk/sumgami) # the sequence of the ratio is not monotonic.
		myCurrentLevels_est[i, 0] = sumgamiyk/sumgami # potential division by 0 for large numChannels.

	return Aagg_est, myCurrentLevels_est
#################################################################################
def brutforceTrace(Aagg_est, myCurrentLevels_est, sigma, intT):
	myStates 	= np.zeros([intT, 1])
	myCurrent 	= np.zeros([intT, 1])
	numChannels = int(int(len(myCurrentLevels_est[:,0])) - 1)

	Aaggcsum 	= np.cumsum(Aagg_est, axis=1)

# The initial state here is always 0, resting.
	for i in range(1,intT):
		myS 		= int(myStates[i-1,0]) # this is the row in Pcsum that you want.
		U 			= random.random()
# do current trace using aggregated A.
		for j in range(0, numChannels+1): # go thru' all columns of the row myS of matrix Aagg.
			if j==0:
				if U <= Aaggcsum[myS, j]:
					myStates[i, 0] = j # this is potentially my z_k over the space u_k.
					myCurrent[i, 0] = myCurrentLevels_est[j, 0]
			else:
				if Aaggcsum[myS,j-1] < U and U <= Aaggcsum[myS, j]: # check if you are missing the use of j-1 at j = 1.
					myStates[i, 0] = j
					myCurrent[i, 0] = myCurrentLevels_est[j, 0]

	# generate your intT gaussian noise here.
	# https://www.adamsmith.haus/python/answers/how-to-add-noise-to-a-signal-using-numpy-in-python
	currentNoise = np.random.normal(0.0, sigma, myCurrent.shape)
#	print(currentNoise)
#	print(currentNoise.shape)
	myCurrentNoisy = myCurrent + currentNoise

	return myStates, myCurrent, myCurrentNoisy

#################################################################################	
#################################################################################	
#################################################################################		
#################################################################################	
#################################################################################
def simulateChungHMM(myDataRaw):

# 13th January 2023. Refine the draft model.
	"""
	Initialize:
	1. Drive current to 0. done.
	2. Get all moments and PSD.
	3. Calculate initial zeta, rho, sigmaSquared, lambda_2, numChannels, pi_open_ss, pi_closed_ss.
	You need the mean, second order central moment, and third order central moment.
	http://www.milefoot.com/math/stat/rv-moments.htm
	http://homepages.gac.edu/~holte/courses/mcs256/documents/Moments_of_a_Discrete_Random_Variable.pdf
	"""
	
# the preclamp is not long, get the noise variance from the postclamp duration.
	global DELTAT
	DELTAT 	= myDataRaw[2,0] - myDataRaw[1,0] # seconds.
#	print(DELTAT)
	T0 		= 0.0 # recording starts, preclamp voltage is applied.
	T1 		= 0.23 # preclamp voltage is replaced by clamp voltage.
	T2 		= 0.27
	T3 		= 5.23 # clamp goes back to preclamp voltage.
	T4 		= 5.27
	T5 		= 6.0 # final time, seconds.
	epsT 	= 0.02 # 20 ms buffer time at T1 and T2; and T3 and T4.
	
	intT0 		= int(T0/DELTAT)
	intT1 		= int(T1/DELTAT) 
	intT2 		= int(T2/DELTAT) 
	intT3 		= int(T3/DELTAT) 
	intT4 		= int(T4/DELTAT) 
	intT5 		= int(T5/DELTAT)
	intepsT 		= int(epsT/DELTAT) 
	
	preClampData 	= myDataRaw[intT0:intT1, :]
	myData 			= myDataRaw[intT2:intT3,:]
	postClampData 	= myDataRaw[intT4:intT5,:]
	
	preI 			= np.mean(preClampData[:,2])
	postI 			= np.mean(postClampData[:,2])
	
	print('Driving current before and after clamp are different, hope is it does not affect the quantals.')
	print(preI, postI)
	
	# Organize the data as you need.
	preClampData[:,1] = -	preClampData[:,1]
	preClampData[:,2] = 	preClampData[:,2] 		- postI
	postClampData[:,1] = -	postClampData[:,1]
	postClampData[:,2] = 	postClampData[:,2] 	- postI

	myData[:,1] = 	-myData[:,1]
	myData[:,2] = 	 myData[:,2] - postI
	
	intT = len(myData[:,2])

	if debg==2:
		plt.figure(figsize=(8, 5))
		plt.plot(preClampData[:,0], preClampData[:,1])
		plt.show()
		plt.figure(figsize=(8, 5))
		plt.plot(preClampData[:,0], preClampData[:,2])
		plt.show()

# this figure may show spikes. remove the spikes.
	if debg==2:
		plt.figure(figsize=(8, 5))
		plt.plot(myData[:,0], myData[:,1])
		plt.show()
		plt.figure(figsize=(8, 5))
		plt.plot(myData[:,0], myData[:,2])
		plt.show()
		
# a simple hack. It seems that the max of any signals is not over 0.5 and never below -5.5. Replace any outliers with that.
# replace with something better/more sensible.
	if 1==1:
		for i in range(1, len(myData[:,2])-1):
			if myData[i,2]<-5.5:
				myData[i,2] = (myData[i-1,2]+myData[i+1,2])/2.0
			if myData[i,2]>0.5:
				myData[i,2] = (myData[i-1,2]+myData[i+1,2])/2.0

	if debg==2:
		plt.figure(figsize=(8, 5))
		plt.plot(myData[:,0], myData[:,1])
		plt.show()
		plt.figure(figsize=(8, 5))
		plt.plot(myData[:,0], myData[:,2])
		plt.title('Simple hack for despiking. see the program')
		plt.show()

	
#	sys.exit()
# Chung 1993 formulas to estimate zeta, rho, numChannels, single channel conductance.
# The noise variance could be calculated from the pre-clamp voltage duration. But if you see the cell 1 control
# trace, it seems to have non-zero mean. that means the noise variance from data will still have some openings.
# As an approximation, I took the noise variance I worked out before, and calculated other quantities.
# The noise variance and the non-unity eigenvalue must be calculated from the Z-transform of the signal, i.e. the power spectrum.
# see equations 12-16 and more in: MarkovModelChung1993.pdf.

# you need noise variance. do it using the postClamp current.
# this is an array, an entry for each current level.
	sigma 		= statistics.stdev(postClampData[:,2]) # 0 mean noise variance.

# see equations 12-16 and more in: MarkovModelChung1993.pdf.
# 1. Mean.
	m1y 		= np.mean(myData[:,2])		
#  2. Variance (2nd central moment).
	mu2y 		= scipy.stats.moment(myData[:,2], moment=2) # variance of the signal.
# 3. third moment.
	mu3y 		= scipy.stats.moment(myData[:,2], moment=3) # skewness.
# define the signal quantities. y = noisy signal. x = non-noisy signal.
	m1x 		= m1y
	# see Hidden_Markov_modeling_of_ion_.pdf equation 1.17 p. 36 for a slightly different definition of s_est that uses mu2x.
	mu2x		= mu2y - sigma*sigma # the sigma^2 could also come out of the power spectrum as a double check.
	mu3x 		= mu3y
	gammaEst 	= m1y * mu3y / (mu2x * mu2x )
	pi_closed 	= 1.0 / (2.0 - gammaEst)
	pi_open 		= 1.0 - pi_closed
	s_est 		= mu2x / (pi_closed * m1y)
	# the pi_open and pi_closed have to be positive, mu2x also must be positive.
	# the numChannels will be negative if mu2x is negative, i.e. mu2y is less than sigma^2
	numChannels = round( pi_closed * m1y * m1y / (pi_open * mu2x) , 0) # this replaces numChannels.
	if numChannels<=0:
		print('unreasonable numChannels. exiting.')
		print(numChannels)
		sys.exit()
	print('Estimated numchannels, number of channels:')
	print(numChannels)

	global Npiopics2Squared
	Npiopics2Squared = numChannels * pi_closed * pi_open * s_est * s_est


# you need this to get alpha, beta.
	myStates		= np.zeros([intT, 1])
	NC 				= int(numChannels)
	pi_init 			= np.zeros([NC+1, 1])
	
	print('If $\{mu}_2(y)$ is less than $\{sigma}^2$, then the number of channels works out negative. Investigate.')
	print('sigma, m1y, mu2y, mu3y, m1x, single channel current, numChannels, pi_open, pi_closed, mu2x.')
	print(sigma, m1y, mu2y, mu3y, m1x,  s_est,                            numChannels, pi_open, pi_closed, mu2x)

# PSD to give sigma^2 and lambda_2. #################
	
	"""
	get the second eigenvalue from the power spectrum.
	# see: https://python-advanced.quantecon.org/arma.html for autocovariance and FFT.
	# matplotlib mlab psd.
	#	https://matplotlib.org/stable/api/mlab_api.html
	# also read: http://www.scholarpedia.org/article/1/f_noise for 1/f noise due to ion channels and other processes.
	# the detrend can be none, mean, linear. It has an effect on the right of the plot.
	# the s_mlab is defined in: MarkovModel_HowToZSpectrum.pdf
	# the shape of the fit below does not match the results shown in MarkovModel_HowToZSpectrum.pdf .
	# noverlap is half of the NFFT. see MarkovModelPSD.pdf
	# You have to use the hanning window explicitly somewhere.

	See page 52 in: MLE_2DHistograms5.pdf . Implement that method to see if the psd looks anything 
	reasonable.
	The PSD from data is not what Chung expects. He used 10k or 500k data points, while I have 6k.
	
	The PSD formula assumes s1 (of s1 and s2 states in the 2x2 single channel MC) to be 0. cell1 may not
	have s1 to be 0 and a shift may be needed before the graph looks reasonable.
	The noise variance sigmaSquared_est can be estimated outside of the PSD.
	If the noise variance from fitting the data is negative (or wrong), do it using the holding potential part of the data. see page 125 of MarkovModel_HowToZSpectrum.pdf .
	The PSD has to be done over non-overlapping segments. 
	"""
	s_mlab, f_mlab = mlab.psd(myData[:,2], NFFT=512, Fs=1.0/DELTAT, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=256, pad_to=256, sides='default', scale_by_freq=None)
	tot_s = np.sum(s_mlab)

# Fit the Syomega curve to get lambda and sigma^2.
	'''	
	https://scipy-lectures.org/intro/scipy/auto_examples/plot_curve_fit.html
	There is a Lecture_*.pdf in the pdfs/ directory.
	see: MarkovModel_HowToZSpectrum.pdf
	In above paper, the formula given is:
	Syomega (f) = sigma^2 + N * pi_open * pi_closed * s2 * s2 * ( 1 - lambda*lambda) / (1 + lambda * lambda - 2*lambda*cos(f*dt) )
	s_mlab is Syomega, f_mlab is f.
	p[0] = sigma^2. p[1] = N * pi_open * pi_closed * s2 * s2 . p[2] = lambda.
	[ 4.45960139e+02 -4.49706757e+02  4.51390257e-03]
	'''

#	print(type(s_mlab), type(f_mlab)) # they are both 1D numpy ndarray.
# the initial guess is inspired by the sigmasquared given by trace, p1 is N*pi_o*pi_c*s2, s2 = -0.5, N = 55, pi_c = 0.4 (see above). p3 is between 0 and 1 far from 0 and 1.
#
	params, params_covariance = optimize.curve_fit(mySyomega, f_mlab, s_mlab, p0=[sigma*sigma, 0.5], maxfev=1000000)	
	sigmaSquared_est 		= params[0]
	lambda_est 				= params[1]
	simulatedSyomega		= mySyomega(f_mlab, sigmaSquared_est, lambda_est)

	if debg==2:
		matplotlib.rc('figure', figsize=(14, 6))
		plt.loglog(f_mlab, s_mlab,'o', color='black', label='data psd')
		plt.loglog(f_mlab, simulatedSyomega,color='red', label='fitted to data')	
	#	plt.loglog(f_mlabsim, s_mlabsim,color='blue', label='psd of simulation')
		plt.legend()
		plt.title('PSD of data gives $\lambda_2$ and $\sigma^2$.')
		plt.xlabel('freq. (Hz).')
		plt.ylabel('PSD (pA)^2 / Hz.')	
		plt.show()

		print('The PSD generated noise variance and second eigenvalue:')
		print(sigmaSquared_est, lambda_est)
#		sys.exit()
	"""
	The lambda_est must be on or inside unit circle, i.e. must be -1 < lambda_est < 1.
	For most of the data, the fit nor the lambda_est are what they should be - see the figure generated above.
	The only reason I need lambda_est is to be able to initialise the zeta and rho values.
	The zeta, rho, and k are always refined which I hope does not depend on this lambda_est but 
	more on the fundamental matrix [zeta 1-zeta; 1-rho rho].
	For now, I will use some value of zeta and rho to get on with rest of the code.
	"""
# in case the lambda_est was to be sensible, then the zeta and rho initialisation would read as following 2 lines:
#	zeta = pi_closed 	+ pi_open 	* lambda_est
#	rho = pi_open	+ pi_closed	* lambda_est

# but if the lambda_est is off, then initialize like this.
# the zeta and rho are always just less than 1. The kappa must be small for the implemented theory to hold true.
	zeta 	= 0.95
	rho		= 0.91
	kappa 	= 0.01
	NC 						= int(numChannels) # make an int out of numChannels because it is an iterator.

###################### iterations. 
#########xxxxxxxxxxxxxx#############
# The iterations start here. There are a lot of matrices that are created.
# at some point, you need to make sure that the many matrices are being updated as you think they are.
# The iterations loop starts here and goes to the end of this function. it is also marked by ##xx symbols.
# 50 iterations with NC = 12 takes: real	17m24.323s. As of Jan 16, we need a lot of iterations.
# iterations, NC, LT, costfunction, zeta, rho, kappa
	myEstimates = [[0, 0, 0.0, 0.0, zeta, rho, kappa, 0.0]]
	for iterations in range(0, 1000):
	# define the independent and coupled matrices using zeta, rho, kappa and numChannels.
		ACp,dACpdzeta, dACpdrho 	= coupledMatrices(zeta, rho, NC) # derivative w.r.t. k is just the coupled matrix itself.
#		AInd, dAInddzeta, dAInddrho = independentMatrices(zeta, rho, NC) # derivative w.r.t. k is negative of the independent matrix.
		AInd, dAInddzeta, dAInddrho = independentMatrices2(zeta, rho, NC) # derivative w.r.t. k is negative of the independent matrix.
		if debg==2:
			print( ACp, dACpdzeta, dACpdrho)
			print(AInd, dAInddzeta, dAInddrho)

#		sys.exit()
		
	# define the aggregate matrix obtained from AInd and ACp.
		Aagg 		= (1.0 - kappa)*  AInd 		+ kappa*ACp 			# the aggregate (L+1)x(L+1) elements.
		dAaggdzeta 	= (1.0 - kappa)*dAInddzeta 	+ kappa*dACpdzeta 
		dAaggdrho 	= (1.0 - kappa)*dAInddrho 	+ kappa*dACpdrho 	# the aggregate (L+1)x(L+1) elements.
		dAaggdk 	= 			-  AInd 		+              ACp 		# the aggregate (L+1)x(L+1) elements.		

	# define your emission matrix, bjyk. 
		NC = int(numChannels)
	# this memory allocation is outside iterations.
		myCurrentLevels = np.zeros([NC+1, 1]) # this takes myCurrentLevels_est at iteration>1.
		for i in range(0, NC+1):
			myCurrentLevels[i, 0] = s_est * float(i)
	#	bjyk = emissionProbablities(myData, s_est, NC, sigma) # at iteration>1, the s_est is CurrentLevels_est.
		bjyk = emissionProbablities(myData, myCurrentLevels, NC, sigma)

		if debg==2:		
			for k in range(0, 46000, 100):
				plt.plot(bjyk[:,k])
			plt.axhline(y = 1.0, color = 'r', linestyle = '-')
			plt.show()

	# forward/backward. ################################################
	# do alpha, beta, gammai, ...
		j 			= int(myStates[0,0]) # myStates are revised in the Viterbi call or in the brutforce call.
		pi_init[j,0] 	= 1.0
	# The mypi may need to be put into pi_init. As of now, mypi has small value entries.
	# On the other hand, mypi may not be required as the Viterbi will calculate the initial state dbn. below.
		myAlpha, myBeta, LT, gammaki, histogramT, myXi, mypi = forwardBackward(bjyk, pi_init, Aagg)

	#	print('histogram:')
	#	print(histogramT)
	#	plt.plot(histogramT)
	#	plt.show()
	#	print(mypi)
	#	print(pi_init)

	# estimate the Aagg_est and myCurrent_est.
	# use this estimated current to work out the emission.
	# use the estimated Aagg_est to estimate zeta, rho, kappa.
		Aagg_est, myCurrentLevels_est = estimatematrixCurrentLevels(myXi, gammaki, myData)
		
	#	print(Aagg_est)
	# the myCurrent is not in ordered state which may cause issues when assigning values to the states.
		if debg==2:
			print('Estimated currrent levels and s_est current levels, make sure level0 is what you think it is:')
			for i in range(0, int(numChannels)+1):
				print(myCurrentLevels_est[i,0], myCurrentLevels[i,0])

	############################################################################
	# set up the steepest gradient here for now. This program has to be reorganized for flow after this and after Viterbi algorithm is there.
	# A better option is simulated annealing.
		smallmu = 0.0001 # this is the gradient.

	#	print("The transition matrics going into the cost function:")
	#	print(Aagg_est)
	#	print(Aagg)

		costfunction 	= 0.0
		dFdzeta 		= 0.0
		dFdrho 		= 0.0
		dFdk 		= 0.0
		NC = int(numChannels)
		for i in range(0, NC+1):
			for j in range(0, NC+1):
				costfunction 	+= (Aagg[i,j] - Aagg_est[i,j])*(Aagg[i,j] - Aagg_est[i,j])
				dFdzeta 		+= (Aagg[i,j] - Aagg_est[i,j])*dAaggdzeta[i,j]
				dFdrho 		+= (Aagg[i,j] - Aagg_est[i,j])*dAaggdrho[i,j]
				dFdk 		+= (Aagg[i,j] - Aagg_est[i,j])*dAaggdk[i,j]

	# do the estimate of zeta, rho, k.
		zeta_revised 		= zeta 	- smallmu * dFdzeta
		rho_revised 		= rho 	- smallmu * dFdrho
		
# some simple hack. More clever use of kappa becoming negative may help.
		if kappa 	- smallmu * dFdk > 0.0:
			kappa_revised 	= kappa 	- smallmu * dFdk
		else:
			kappa_revised = 0.0

	# error trapping.
		if zeta_revised<=0.0 or zeta_revised>1.0:
			print('revised zeta not a probablity. exiting.')
			sys.exit()
		if rho_revised<=0.0 or rho_revised>1.0:
			print('revised rho not a probablity. exiting.')
			sys.exit()
			
#		if kappa_revised<=0.0 or kappa_revised>1.0:
#			print('revised kappa not a probablity. exiting.')

		if kappa_revised <0.0:
			kappa_revised = 0.0
		
		if kappa_revised>1.0:
			print('revised kappa more than 1, not a probablity. exiting.')
			sys.exit()

	############################################################################
	# do the Viterbi for the sake of the code. 2 Jan 2023.
	# send in: Aagg, initial states vector, emission matrix, the data.
	# As set up above, I have one state chosen randomly and all other states are 0. myState[0,0] is the state between 0 and numChannels.
	# this Viterbi is not working yet.
	# Viterbi is needed to know the initial state of the channel, pi_init or mypi[0, 0].
	#	myViterbi(Aagg, myStates[0,0], bjyk, myData)
	#	sys.exit()

	# do a brut force trace that you can plot.
	# Turn this off when you have the Viterbi working.
	# Aagg_est is numChannels x numChannels. myCurrentLevels_est is numChannels x 1. sigma is a number.
		myStates, myCurrent, myCurrentNoisy = brutforceTrace(Aagg_est, myCurrentLevels_est, sigma, intT)
		if debg==1: # plot the data, current, and noisy current.
			plt.figure(figsize=(8, 5))		
			plt.plot(myData[:,0], myData[:,2], 			color='black', 	label='data.')
			plt.plot(myData[:,0], myCurrentNoisy[:,0], 	color='red', 	label='simulation, white noise.')
			plt.plot(myData[:,0], myCurrent[:,0], 		color='blue', 	label='simulation, no noise.'	 )
			plt.legend()			
			plt.show()

	# put new into old.
		if zeta_revised>0.0 and zeta_revised<=1.0:
			zeta 	= zeta_revised
		if rho_revised>0.0 and rho_revised<=1.0:			
			rho 		= rho_revised
		if kappa_revised>0.0 and kappa_revised<=1.0:		
			kappa 	= kappa_revised
		
		if zeta_revised < 0.0 or rho_revised < 0.0 or kappa_revised < 0.0:
			print('The probabilities became negative, exiting.')
			sys.exit()

		for i in range(0, int(numChannels)+1):
			myCurrentLevels[i,0] = myCurrentLevels_est[i,0]

		open_probability = (1.0 - rho)/(2.0 - rho - zeta)
		
		print('iter, NC, estimated likelihood, costfunction, zeta, rho, kappa, and open probability:')	
		print(iterations, NC, LT, costfunction, zeta, rho, kappa, open_probability)
		myEstimates.append([iterations, NC, LT, costfunction, zeta, rho, kappa, open_probability])

	# end of iterations loop.
	#########xxxxxxxxxxxxxx#############
	
	myEstimates.pop(0) # could also be: del myEstimates[0]
	np.savetxt("myEstimates.txt", myEstimates)
	
	# this return is redandant - all calculations are finished, all plots are finished.
	return  myStates, myCurrent, myCurrentNoisy

############## Main/driver. ##################################################
def main():
	# initial conditions/parameters. numChannels, zeta, rho, kappa are all up for estimation.
	# 3 Jan 2023. The max numChannels I can handle on the mac is around 15.
	# 13 January 2023. The dataset is being revised.
	# read in your data here, so that all of it is available.
	# some files will give negative number of channels.
#	abf 		= pyabf.ABF("cell1/Control/23109294.abf") # negative numChannels.
#	abf 		= pyabf.ABF("cell1/Control/23109295.abf") # 370 numChannels.
#	abf 		= pyabf.ABF("cell1/Control/23109296.abf") # works fine so far.
#	abf 		= pyabf.ABF("cell1/Control/23109297.abf") # NC = 18.
#	abf 		= pyabf.ABF("cell1/Control/23109299.abf") # has a spike, but it went thru.
#	abf 		= pyabf.ABF("cell1/Control/23109300.abf") # massive spike, negative numChannels. Despiking helped, NC=2.
#	abf 		= pyabf.ABF("cell1/Control/23109301.abf")
#	abf 		= pyabf.ABF("cell1/Control/23109302.abf")
#	abf 		= pyabf.ABF("cell1/Control/23109303.abf") # spike.
#	abf 		= pyabf.ABF("cell1/Control/23109304.abf") # spike, negative numChannels. Simple despiking did not help.
#	abf 		= pyabf.ABF("cell1/Control/23109305.abf")	
#	abf 		= pyabf.ABF("cell1/Pressure/23109315.abf")
#	abf 		= pyabf.ABF("cell1/Pressure/23109316.abf")
#	abf 		= pyabf.ABF("cell1/Pressure/23109317.abf")
#	abf 		= pyabf.ABF("cell1/Pressure/23109318.abf") # 2 channels.
#	abf 		= pyabf.ABF("cell1/Pressure/23109319.abf")
#	abf 		= pyabf.ABF("cell1/Pressure/23109320.abf")
#	abf 		= pyabf.ABF("cell1/Pressure/23109321.abf") # spike.	
#	abf 		= pyabf.ABF("cell1/Pressure/23109322.abf") # spike.
	abf 		= pyabf.ABF("cell1/Pressure/23109323.abf")
#	abf 		= pyabf.ABF("cell1/Pressure/23109329.abf")

#	strng = lc.getline('flist.txt', 10)
#	print(strng)
#	abf = pyabf.ABF("strng")	
#	sys.exit()

	abf.setSweep(sweepNumber=0, channel=0)
	timee 	= abf.sweepX
	voltage 	= abf.sweepY
	# get the current, channel 1.
	abf.setSweep(sweepNumber=0, channel=1)
	# timee 	= abf.sweepX
	current 	= abf.sweepY
	myData 	= np.array(np.transpose([timee, voltage, current]))
		
	# the estimator function and its children/branches.
	myStates, myCurrent, myCurrentNoisy = simulateChungHMM(myData)

	if debg==2:
		plt.figure(figsize=(8, 5))
		plt.plot(myData[:,0], myData[:,1])
		plt.show()
		plt.figure(figsize=(8, 5))
		plt.plot(myData[:,0], myData[:,2], color='black', label='recording.')
		plt.show()


if __name__ == "__main__":
	main()
