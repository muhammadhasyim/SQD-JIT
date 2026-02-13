import numpy as np
import sys
import os
import signal
from contextlib import contextmanager
from numpy import random as rn
from opt_einsum import contract
import random
import time
#from time import time
from numba import jit, objmode
import vsc1m_qph as model
from vsc1m_qph import Upolaron, Hel, dHel0, dHel, initR, FILENAME
import tqdm

FILENAME = 'mfe'+FILENAME
replica = sys.argv[1]

@contextmanager
def uninterrupted():
    def handler(signum, frame):
        print(f"Signal {signum} received. Ignoring.")
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        yield  # This yields control to the code inside the 'with' block
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)

# Initialization of the electronic part
def initElectronic(Nstates, initState = 0):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    c = np.zeros((Nstates), dtype='complex128')
    c[initState] = 1.0
    c = np.conj(Upolaron).T @ c
    return c

@jit(nopython=True)
def propagateCi(ci,Vij, dt):
    Vij = Vij.astype(np.complex128)
    c = ci * 1.0
    # https://thomasbronzwaer.wordpress.com/2016/10/15/numerical-quantum-mechanics-the-time-dependent-schrodinger-equation-ii/
    ck1 = (-1j) * (Vij @ c)
    ck2 = (-1j) * Vij @ (c + (dt/2.0) * ck1 )
    ck3 = (-1j) * Vij @ (c + (dt/2.0) * ck2 )
    ck4 = (-1j) * Vij @ (c + (dt) * ck3 )
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)
    return c

@jit(nopython=True)
def Force(ci,R):

    dH = dHel(R) #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0  = dHel0(R) #dat.dH0 

    F = -dH0 #np.zeros((len(dat.R)))
    for i in range(len(ci)):
        F -= dH[i,i,:]  * (ci[i] * ci[i].conjugate() ).real
        for j in range(i+1, len(ci)):
            F -= 2.0 * dH[i,j,:]  * (ci[i].conjugate() * ci[j] ).real
    return F

@jit(nopython=True)
def VelVer(R,P,ci,F1,M,Hij,dt):#dat) : 
    v = P/M
    
    # electronic wavefunction
    ci = ci * 1.0
    
    #EStep = int(par.dtN/par.dtE)
    #dtE = par.dtN/EStep

    # half electronic evolution
    #for t in range(int(np.floor(EStep/2))):
    ci = propagateCi(ci, Hij, dt/2)  
    ci /= np.sum(ci.conjugate()*ci) 
    ci = ci * 1.0 

    # ======= Nuclear Block ==================================
    R += v * dt + 0.5 * F1 * dt ** 2 / M
    
    #------ Do QM ----------------
    Hij  = Hel(R)
    #dat.dHij = dHel(dat.R)
    #dat.dH0  = dHel0(dat.R)
    
    #-----------------------------
    F2 = Force(ci,R) # force at t2
    v += 0.5 * (F1 + F2) * dt / M
    # ======================================================
    # half electronic evolution
    ci = propagateCi(ci,Hij, dt/2)  
    ci /= np.sum(ci.conjugate()*ci)  
    ci = ci * 1.0 

    return R, v*M, ci, F2, Hij


@jit(nopython=True)
def pop(ci):
    return np.outer(ci.conjugate(),ci)

def runTraj():
    # Parameters ---------------
    NSteps = model.NSteps
    NSkip = model.NSkip
    NTraj = model.NTraj
    NStates = model.NStates
    NBath = model.NBath
    M = model.M
    dtN = model.dtN
    initState = model.initState
    hvij = model.hvij
    #---------------------------
    
    if NSteps%NSkip == 0:
        pl = 0
    else:
        pl = 1
    
    iskip = 0 # counting variable to determine when to store the current timestep data
    Popp_final = np.zeros(NSteps // NSkip +pl, dtype=float)
    with open("{}".format(FILENAME+".txt"),"w") as PiiFile: 
        for t in range(NSteps): # single trajectory
            #------- ESTIMATORS-------------------------------------
            if(t % NSkip == 0): # this is what lets NSkip choose which timesteps to store
                PiiFile.write(f"{t * dtN / model.ps} \t")
                PiiFile.write(str(Popp_final[iskip]) + "\t")
                PiiFile.write("\n")#,flush=True)
                PiiFile.flush()
                iskip += 1
    itraj = 0
    
    # Ensemble
    for iTraj in tqdm.tqdm(range(NTraj)): 
        try:
            rho_ensemble = np.zeros((NStates,NStates,NSteps//NSkip + pl), dtype=complex)
            
            # Trajectory data
            R, P = initR(initState)#,initial=True)#np.array([np.zeros(model.NBath)])) # initialize nuclear positions and momenta
            
            # Call function to initialize mapping variables
            ci = initElectronic(NStates, initState) # np.array([0,1])

            #----- Initial QM --------
            Hij  = Hel(R)
            F1 = Force(ci, R) # initial force
            #----------------------------
            iskip = 0 # please modify
            for t in range(NSteps): # One trajectory
                #------- ESTIMATORS-------------------------------------
                if(t % NSkip == 0): # this is what lets NSkip choose which timesteps to store
                    popl = pop(ci)#, NStates) # add current populations in diabatic basis
            
                    rho_ensemble[:,:,iskip] += popl
                    
                    Popp = np.trace(hvij @ rho_ensemble[:,:,iskip].T) 
                    Popp_final[iskip] = (Popp.real+itraj*Popp_final[iskip])/(itraj+1)
                    
                    iskip += 1
                    #PiiFile.flush()
                #-------------------------------------------------------
                dt = 1.0*dtN # reset timestep as dtN
                R, P, ci, F1, Hij = VelVer(R, P, ci,  F1, M, Hij, dt) # run full verlet step
            with uninterrupted():
                iskip = 0
                with open("{}".format(FILENAME+".txt"),"w") as PiiFile: 
                    PiiFile.write(f"#Ntrajs: {itraj+1} \n")
                    for t in range(NSteps): # single trajectory
                        #------- ESTIMATORS-------------------------------------
                        if(t % NSkip == 0): # this is what lets NSkip choose which timesteps to store
                            PiiFile.write(f"{t * dtN / model.ps} \t")
                            PiiFile.write(str(Popp_final[iskip]) + "\t")
                            PiiFile.write("\n")#,flush=True)
                            PiiFile.flush()
                            iskip += 1
                itraj += 1
        except Exception as e:
            print(e)
            print("Continue to the next trajectory.")
        # Ensure stdout is flushed
        sys.stdout.flush()
    return rho_ensemble

if __name__ == "__main__": 
    runTraj()

