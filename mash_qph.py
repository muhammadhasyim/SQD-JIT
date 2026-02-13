# Multi-state MASH code by Muhammad R. Hasyim, adapted from Eric Koessler and Johan Runeson
import numpy as np
import os
import signal
from contextlib import contextmanager
from numpy import random as rn
from opt_einsum import contract
import sys
import random
import time
#from time import time
from numba import jit, objmode
import vsc1m_qph as model
from vsc1m_qph import β, Upolaron, dtN, Hel, dHel0, dHel, initR, FILENAME
import tqdm

FILENAME = 'fs-mash'+FILENAME
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

#This R is the classical DOFs
#@jit(nopython=True,fastmath=True)
# initialization of the coefficients
def initMapping(NStates, initState, R): 
    #print(R)    
    #Next, we sample the initial state, crucial part of MASH
    #Assume this initialization is on the diabatic basis
    sumN = np.sum(np.array([1/n for n in range(1,model.NStates+1)]))
    alpha = (model.NStates - 1)/(sumN - 1)
    beta = (alpha - 1)/model.NStates
    c = np.sqrt(beta/alpha) * np.ones((model.NStates), dtype = np.complex_)
    c[initState] = np.sqrt((1+beta)/alpha)
    for n in range(model.NStates):
        uni = random.random()
        c[n] = c[n] * np.exp(1j*2*np.pi*uni)
    #Transform to MH basis
    c = np.conj(Upolaron).T @ c
    E, U = np.linalg.eigh(Hel(R))
    c = np.conj(U).T @ c
    return c

@jit(nopython=True,fastmath=True)
def getACST(c): # return state with largest population
    return np.argmax(np.abs(c))

@jit(nopython=True,fastmath=True)
def Umap(c, dt, E): # this updates the quantum state using adiabatic energies
    return np.exp(-1j*dt*E)*c

@jit(nopython=True,fastmath=True)
def Force(acst, R, U): # this calculates the classical force on each nuclear DOF using only the active state
    F = -dHel0(R) # state-independent force
    dF = -dHel(R) #U,acst) #state-dependent force, transformed to adiabatic basis
    
    #dF = contract('i,ijk,j -> k',U[:,acst].conjugate(),dF, U[:,acst],optimize='auto')
    U_conj = U[:, acst].conjugate()
    U_slice = U[:, acst]

    # Broadcasting and element-wise multiplication
    term1 = U_conj[:, np.newaxis, np.newaxis] * dF
    term2 = term1 * U_slice[np.newaxis, :, np.newaxis]

    # Summing over the first axis (axis 0)
    sum_over_i = np.sum(term2, axis=0)
    # Summing over the second axis (axis 1)
    dF = np.sum(sum_over_i, axis=0)
    dF = dF.ravel()
    #print(np.shape(dF),np.shape(F))
    F += dF
    return F.real

@jit(nopython=True,fastmath=True)     
def HOPDAT(acst, c): # calculate current active state and store result
    NStates = model.NStates
    hopdat = np.array([False, acst, acst]) # array elements: [hop needed?, previous active state, current active state]
    n_max = getACST(c)
    if(acst != n_max):
        hopdat[0] = True
        hopdat[1] = acst
        hopdat[2] = n_max
    return hopdat

@jit(nopython=True,fastmath=True)
def hopping_nocheck(c, P, R,M, Ead, U, a, b): # attempt a hop (dont check if a==b)
    if a != b:
        # a is previous active state, b is current active state
        P *= 1/np.sqrt(M) # momentum rescaled
        ΔE = np.real(Ead[b] - Ead[a]) 
        
        dHij = dHel(R)
        #contract('ij,k -> ijk',model.Rij, model.ck)
        # dij is nonadiabatic coupling
        # <i | d/dR | j> = < i |dH | j> / (Ej - Ei)
        
        Ψa = U[:,a]
        Ψb = U[:,b]
        
        # -----------------------------------------------------------------------------------------
        # Sharon-Tully Approach
        # dij = contract('j,jkl->kl',np.conj(Ψa).T,dat.dHij)
        # dij = -contract('kl,k->l', dij, Ψb)/(Ead[b]-Ead[a])
        # δk = dij * 4 *(np.conj(dat.ci[a])*dat.ci[b]).real  
        # -----------------------------------------------------------------------------------------
        
        
        # # direction -> 1/√m ∑f Re (c[f] d [f,a] c[a] - c[f] d [f,b] c[b])  # c[f] = ∑m <m | Ψf> 
        # #            =Re ( 1/√m ∑f ∑nm Ψ[m ,f]^ . (<m | dH/dRk | n> ) . Ψ[n ,a] /(E[a]-E[f])
        j = np.arange(len(Ead))
        ΔEa, ΔEb = (Ead[a] - Ead), (Ead[b] - Ead)
        ΔEa[a], ΔEb[b] = 1.0, 1.0 # just to ignore error message
        rΔEa, rΔEb = (a != j)/ΔEa, (b != j)/ΔEb
        
        U_conj = U.conjugate()
        dHab = np.zeros((U.shape[1], U.shape[1], dHij.shape[2]), dtype=np.complex128)
        for k in range(dHij.shape[2]):
            dHij_k = dHij[:, :, k]
            dHab[:, :, k] = np.dot(U_conj.T, np.dot(dHij_k, U))

        # Compute terms without using einsum
        term1 = np.sum(c.conjugate()[:, np.newaxis] * dHab[:, a, :] * c[a] * rΔEa[:, np.newaxis], axis=0)
        term2 = np.sum(c.conjugate()[:, np.newaxis] * dHab[:, b, :] * c[b] * rΔEb[:, np.newaxis], axis=0)

        """dHab =   contract('ia, ijk, jb -> abk', U.conjugate(), dHij, U,optimize='auto')
        term1 = contract('n, nj, n -> j', c.conjugate(),  dHab[:, a, :] * c[a], rΔEa,optimize='auto')
        term2 = contract('n, nj, n -> j', c.conjugate(),  dHab[:, b, :] * c[b], rΔEb,optimize='auto')
        """
        δk = (term1 - term2).real * 1/np.sqrt(M) 

        #Project the momentum to the new direction
        P_proj = np.dot(P,δk) * δk / np.dot(δk, δk) 
        #print('np.dot(δk, δk) ',np.dot(δk, δk) )
        #Compute the orthogonal momentum
        P_orth = P - P_proj # orthogonal P

        #Compute projected norm, which will be useful later
        P_proj_norm = np.sqrt(np.dot(P_proj,P_proj))
        
        #Compute the total kinetic energy in the projected direction

        if(P_proj_norm**2 < 2*ΔE): # rejected hop
            P_proj = -P_proj # reverse projected momentum
            P = P_orth + P_proj
            accepted = False
        else: # accepted hop
            P_proj_old = np.copy(P_proj)
            Pold = np.copy(P)
            P_proj = np.sqrt(P_proj_norm**2 - 2*ΔE)/P_proj_norm * P_proj #re-scale the projected momentum
            P = P_orth + P_proj
            
            #Compute LZ probability
            #dP = P-Pold
            #dF = Force(a,R,U)-Force(b,R,U)
            #fac = np.dot(dP,dF)
            #plz = np.exp(-np.pi*0.5*np.sqrt(0.5*dtN*np.abs(ΔE)**3/np.abs(fac)))
            
            #Compute Tully's probability
            term = c.conjugate()[a]*c[a]#-c.conjugate()[b]*c[b]*np.exp(-β*(Ead[a]-Ead[b]))
            plz = dtN*(c.conjugate()[b]*c[a]) * dHab[b, a, :]/(term*ΔE) #* rΔEa[b])#, axis=0)
            plz = plz.real
            plz = max(0,np.dot(Pold,plz))

            if np.random.uniform(0,1) < plz:
                accepted = True
            else:
                P_proj = -P_proj_old # reverse projected momentum
                P = P_orth + P_proj
                accepted = False
        P *= np.sqrt(M)
        
        P = P.real
        return P.real, accepted
    return P, False

def propagateCi(ci,Vij, dt):
    c = ci * 1.0
    # https://thomasbronzwaer.wordpress.com/2016/10/15/numerical-quantum-mechanics-the-time-dependent-schrodinger-equation-ii/
    ck1 = (-1j) * (Vij @ c)
    ck2 = (-1j) * Vij @ (c + (dt/2.0) * ck1 )
    ck3 = (-1j) * Vij @ (c + (dt/2.0) * ck2 )
    ck4 = (-1j) * Vij @ (c + (dt) * ck3 )
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)
    return c

@jit(nopython=True,fastmath=True)
def eigh_numba(A):
    return np.linalg.eigh(A)

@jit(nopython=True,fastmath=True)
def VelVer(R, P, c, acst, F1, M, E, U, dt): # this updates the nuclear DOFs using Velocity Verlet (quantum state is also updated within)
    
    v = P/M # initial nuclear velocity
    
    # half electronic evolution
    c = Umap(c, dt/2, E) # unitary mapping which updates the quantum state
    #c /= np.sum(c.conjugate()*c) 
    #''' NECESSARY FOR DYNAMICS: change adiabatic to diabatic basis '''
    c = U.astype(np.complex128) @ c 
    
    v += 0.5 * F1 * dt / M
    
    R += v * dt
    
    E, U = np.linalg.eigh(Hel(R)) # find E, U in new adiabatic basis
    F2 = Force(acst, R, U) # update force
    
    v += 0.5 * F2 * dt / M

    #''' NECESSARY FOR DYNAMICS: change diabatic to (new) adiabatic basis '''
    c = np.conj(U.astype(np.complex128)).T @ c 

    c = Umap(c, dt/2, E) # unitary mapping which updates the quantum state
    #c /= np.sum(c.conjugate()*c) 
    
    return R, v*M, c, F2, acst, E, U

@jit(nopython=True,fastmath=True)
def pop(c, NStates): # returns the density matrix estimator (populations and coherences)
    sumN = np.sum(np.array([1/n for n in range(1,NStates+1)])) # constant based on NStates
    alpha = (NStates - 1)/(sumN - 1) # magnitude scaling
    beta = (1-alpha )/NStates # effective zero-point energy
    prod = np.outer(c,np.conj(c)) 
    return alpha * prod + beta * np.identity(NStates) 

@jit(nopython=True,fastmath=True)
def pop_diff(c, acst): # return population difference between current active state and other highest populated state
    pop = np.abs(c)**2
    pop[acst] = 0
    return np.abs(c[acst])**2 - np.max(pop)

def runTraj(): # this is the main function that runs everything; call this function to run this method
    
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
    for iTraj in tqdm.tqdm(range(NTraj)): # repeat simulation for each trajectory
        try:
            rho_ensemble = np.zeros((NStates,NStates,NSteps//NSkip+pl), dtype=np.complex_) # stores density matrix in desired basis for each (printed) timestep (averaged over trajectories)
            #psi = np.zeros((NTraj,NStates,NSteps//NSkip + pl), dtype=np.complex_) # stores wavefunction in desired basis for each (printed) timestep and for each trajectory
            #R_t = np.zeros((NTraj,NBath,NSteps//NSkip + pl)) # stores each nuclear position for each (printed) timestep and for each trajectory
            #active_state = np.zeros((NTraj,NSteps//NSkip)) # stores active state for each (printed) timestep and for each trajectory
            
            R, P = initR(initState)#,initial=True)#np.array([np.zeros(model.NBath)])) # initialize nuclear positions and momenta
            
            E, U = np.linalg.eigh(Hel(R)) # find initial E, U
            
            #WHat's returned is in the adiabatic basis
            c = initMapping(NStates, initState, R) # initialize coefficients in adiabatic basis
            
            # this c is propagated in the adiabatic basis
            acst = getACST(c) # in adiabatic basis
            
            #popl = pop(U @ c, NStates) # add current populations in diabatic basis
            #print(np.diag(popl))
            #This U is not truncated.
            F1 = Force(acst, R, U) # initial force
            iskip = 0 # counting variable to determine when to store the current timestep data
            for t in range(NSteps): # single trajectory
                #------- ESTIMATORS-------------------------------------
                if(t % NSkip == 0): # this is what lets NSkip choose which timesteps to store
                    popl = pop(U @ c, NStates) # add current populations in diabatic basis
            
                    rho_ensemble[:,:,iskip] += popl
                    
                    Popp = np.trace(hvij @ rho_ensemble[:,:,iskip].T) 
                    Popp_final[iskip] = (Popp.real+itraj*Popp_final[iskip])/(itraj+1)
                    
                    iskip += 1
                    #PiiFile.flush()
                #-------------------------------------------------------
                dt = 1.0*dtN # reset timestep as dtN
                R0, P0, c0, F10, acst0, E0, U0 = 1.0*R, 1.0*P, 1.0*c, 1.0*F1, 1*acst, 1.0*E, 1.0*U
                R, P, c, F1, acst, E, U = VelVer(R, P, c, acst, F1, M, E, U, dt) # run full verlet step
                if(HOPDAT(acst, c)[0]==True): # hop needed
                    tm = dt/2 # midpoint time
                    R, P, c, F1, acst, E, U = 1.0*R0, 1.0*P0, 1.0*c0, 1.0*F10, 1*acst0, 1.0*E0, 1.0*U0
                    R, P, c, F1, acst, E, U = VelVer(R, P, c, acst, F1, M, E, U, tm) 
                    p = np.abs(c)**2
                    p[acst] = 0
                    b = np.argmax(p) # possible new active state
                    P, accepted = hopping_nocheck(c, P, R, M, E, U, acst, b) # attempt hop to state b; change momentum
                    if(accepted==True): # hop successful 
                        acst = b
                        F1 = Force(acst, R, U) # update force for new active surface
                    R, P, c, F1, acst, E, U = VelVer(R, P, c, acst, F1, M, E, U, dt-tm) # do extra verlet to finish timestep
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

if __name__ == "__main__":
    #print("Test")##
    runTraj()
