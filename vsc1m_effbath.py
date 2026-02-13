"""

Python file defining the model Hamiltonian for the single-molecule vibrational strong coupling problemm. 
The double-well system and photon is treated quantum mechanically, while 

"""
import numpy as np
from numpy import kron as ꕕ
from numpy import random as rn
from numba import jit
from opt_einsum import contract
import math
from scipy.special import hermite

# ----- Unit conversion ----- #
au = 27.2113961 # 1 a.u. in eV units (a.u. to eV units)
ps = 41341.374575751 #1 ps in a.u. units (ps to a.u. units)
fs = 41.34136 # 1 fs in a.u.units (fs to a.u.)
cm = 1/4.556335e-6 # 219474.63 (Hz to cm-1)
kB = 3.166829e-6 #Boltzmann constant in a.u. units
eV = 1.0/27.2114 #1 eV in a.u. units (eV to a.u.)
cmn1 = 1.0/219474.63 #Convert cm-1 to a.u. units

# ---------------------------------------- #
# ----- Runfile and model parameters ----- #
# ---------------------------------------- #

FILENAME = 'vsc1m-eff'
wigner = True#False #Whether we want Wigner sampling for the bath modes
NTraj = 10000 #Number of trajectories
finalT = 50*ps # Final simulation time in ps
Nframes = 5000 #Number of frames
if wigner:
    FILENAME += '-wigner'
else:
    FILENAME += '-cl'

# ----- Matter part ----- #
Nm = 4 # Number of matter states
NStates = Nm
wb  = 1000*cmn1  # Barrier frequency, cm-1 to a.u.
Eb  = 2250*cmn1  # Barrier energy, cm-1 to a.u.
T = 300 #Temperature, Kelvin
β = 1/(kB*T) #Inverse temperature, a.u. (Hartree) units 
nR = 2001   #Number of positional grid points.
Rmax = 100  #Maximum Rgrid, a.u. units
Rgrid = np.linspace(-Rmax,Rmax,nR) #Gridpoints

# ----- Photon part ----- #
wc = np.loadtxt('frequency.txt')*cmn1 #Frequency, cm -1

# ----- Interaction part ----- #
eta = 1.25e-3 #Coupling constant in a.u. units

# ----- Cavity and molecule loss part -----  #
#Defines the classical bath parameters, frequency and coupling coefficients. 

Nmbath = 300 #Number of DOFs for mol. bath
Ncbath = 300 #Number of DOFs for cav. bath
M = 1.0     # Mass, a.u.
etav = 0.1 #friction of mol bath, unitless
gammv = 200.0*cmn1 #char. freq. of molecular bath, cm-1 to a.u.
gammc = 1000.0*cmn1 #char. freq of cavity bath, cm-1 to a.u.
tauc = 2000.0*fs #cavity lifetime, fs to a.u.

lambv = 0.5*etav*(wb)*(gammv) #Reorganization energy of mol. bath, a.u. 
lambc = (1-np.exp(-β*wc))*(wc**2 + gammc**2)/(2*tauc*gammc) #Reorganization of cavity bath, a.u.

upper_wv = 10*gammv #Upper cutoff for mol. bath frequencies
upper_wc = 3*gammc #Upper cutoff for cav. bath frequencies

# ------------------------------ #
# ----- Matter Hamiltonian ----- #
# ------------------------------ #

#The matter Hamiltonian is derived from the vibrational states of a symmetric double-well system.
#Working in the high-barrier limit, we only need at most the lowest four eigenstates (actual number can be changed). 
#Note that we rotate the two lowest eigenstates to make them two degenerate ground states, localized in the left (L)
#and (R) side of the barrier.  #The double-well potential and kinetic energy operator are discretized over space, 
#using a discrete variable representation (DVR). 

#The symmetric double well potential 
@jit(nopython=True,fastmath=True)
def Vnew(R, wb= 0.02, Eb = 0.4):
    #wb = wb * cmn1
    #Eb = Eb * cmn1
    a1 = wb*wb/2.0
    a2 = a1*a1/(4*Eb)
    V  = -a1*R**2 + a2*R**4
    V0 = np.min(V)
    return - V0 + V

#Kinetic energy operator for single particle
@jit(nopython=True,fastmath=True)
def Te(re):
    N = float(len(re))
    mass = 1.0
    Tij = np.zeros((int(N),int(N)))
    Rmin = float(re[0])
    Rmax = float(re[-1])
    step = float((Rmax-Rmin)/N)
    K = np.pi/step

    for ri in range(int(N)):
        for rj in range(int(N)):
            if ri == rj:  
                 Tij[ri,ri] = (0.5/mass)*K**2.0/3.0*(1+(2.0/N**2)) 
            else:    
                 Tij[ri,rj] = (0.5/mass)*(2*K**2.0/(N**2.0))*((-1)**(rj-ri)/(np.sin(np.pi*(rj-ri)/N)**2)) 
    return Tij

#The matter Hamiltonian
@jit(nopython=True,fastmath=True)
def Hm(Rgrid):
    V = Vnew(Rgrid,wb,Eb) #Gridded V(R) 
    Hij = Te(Rgrid) + np.diag(V) #KE+PE
    return Hij

Hmij = Hm(Rgrid) #Matter Hamiltonian, size nR x nR
Ei, Um   = np.linalg.eigh(Hmij) #Diagonalize to get the diabatic basis

#Construct the position operator in this basis
Rij = Um.T @ np.diag(Rgrid) @ Um 

#Construct the Heaviside operatir in this basis as well
hv = np.array([1 if Rgrid[i] < 0 else 0 for i in range(nR)])
hvij = Um.T @ np.diag(hv) @ Um 

#Make a rotation in the lowest 2 states, to make them degenerate
Urot = np.identity(nR)
Urot[0,0] = 1/np.sqrt(2) 
Urot[0,1] = 1/np.sqrt(2)
Urot[1,0] = -1/np.sqrt(2)
Urot[1,1] = 1/np.sqrt(2) 

Hmij = Urot.T @ np.diag(Ei) @ Urot
Rij = Urot.T @ Rij @ Urot
hvij = Urot.T @ hvij @ Urot


#Next, we construct the truncated Hamiltonian and operators
Rij = Rij[:Nm,:Nm]
hvij = hvij[:Nm,:Nm]
Hmij = Hmij[:Nm,:Nm]


# ----- Initial State  ----- #
#Decide whiich state of the lowest two eigen states (index 0 or 1) is on the left
initState = 0 if Rij[0,0] < 0 else 1
Ψm0 = np.zeros((Nm,))
Ψm0[initState] = 1

# ------------------------------- #
# ----- Eff. and loss baths ----- #
# ------------------------------- #
#Both the baths attached to the molecule and photon are separate but defined through
#the Debye spectral density. Discretization involve defining upper frequency cutoff

def J(w,lamb, gamm):
    return 2*lamb*gamm*w/(w**2+gamm**2)

def F(w, lamb, gamm):
    return (2*lamb/np.pi)*np.arctan(w/gamm)

def Jeff(w):
    return 2*(1/tauc)*eta**2*wc**3*w/((wc**2-w**2)**2+w**2/tauc**2)

def discretize(N, F, wmax, lamb, gamm):
    w = np.linspace(0.00000001, wmax, N * 1000)
    dw = w[1] - w[0]
    Fw = F(w,lamb,gamm)
    lambs = Fw[-1] 
    wj = np.zeros((N))
    cj = np.zeros((N))
    for i in range(N):
        j = i+1
        wj[i] = w[np.argmin(np.abs(Fw - ((j-0.5)/N) *  lambs))] 
        cj[i] =  wj[i] * (2 * lambs/ N)**0.5 
    return wj, cj

def discretize_gen(N, J, w = np.linspace(0.00001,0.1,10000)/27.2114):
    dw = w[1] - w[0]
    Jw = J(w)
    Fw = np.zeros((len(w)))
    for iw in range(len(w)):
        Fw[iw] = (1.0/np.pi) * np.sum( Jw[:iw]/w[:iw] ) * dw
    lambs = Fw[-1] 
    #print (lambs * 27.2114)
    wj = np.zeros((N))
    cj = np.zeros((N))
    for i in range(N):
        j = i+1
        wj[i] = w[np.argmin(np.abs(Fw - ((j-0.5)/N) *  lambs))] 
        cj[i] =  wj[i] * (2 * lambs/ N)**0.5 
    return wj, cj

ωk, ck = discretize(Nmbath, F, upper_wv, lambv, gammv)  
w = np.linspace(1e-16,upper_wc,20*10**4)#*4.55633e-6 #*29.9792458*1e+9#/27.2114
ωeff, ceff = discretize_gen(Ncbath, Jeff, w)#lambc, gammc)
NBath = Nmbath+Ncbath
ωk = np.concatenate((ωk,ωeff))
ck = np.concatenate((ck,ceff))
dtN = 2*np.pi/(10*max(ωk)) #timestep in a.u. units

NSteps = int(finalT/dtN) #total number of timesteps
NSkip = int(NSteps // Nframes) # period for sampling the data
if NSkip < 1:
    NSkip = 1

lambs =  (np.sum( (1/2) * (ck**2.0) / (ωk**2) ) )
#print(f"Total Reorganization Energy = {lambs} a.u.",lambv)
lambs =  (np.sum( (1/2) * (ceff**2.0) / (ωeff**2) ))
#print(f"Total Reorganization Energy = {lambs} a.u.",lambc)

# ---- Functions for Dynamics ---- #

#The total Hamiltonian, which also includes the coupling term
#In diabatic or MH (polaron == True) basis
@jit(nopython=True,fastmath=True)
def Hel(R):
    Vij = Hmij + Rij * np.sum( ck * R)#[:Nmbath] ) + Qcij * np.sum( cph * R[Nmbath:] )
    Vij += 0.5 * np.sum(ck**2 / ωk**2) * (Rij @ Rij)  
    #Vij += 0.5 * np.sum(cph**2 / ωph**2) * (Qcij @ Qcij)  
    return Vij

#Gradient of the Hamiltonian that is dependent of coupling
#In diabatic or MH (polaron == True) basis
@jit(nopython=True,fastmath=True)
def dHel(R):
    #Need to reshape this array, give it the correct three dimensional view
    #Using opt_einsum
    #dHij = np.concatenate((contract('ij,k->ijk',Rij,ck,optimize='auto'),contract('ij,k->ijk',Qcij,cph,optimize='auto')), axis=2)
    
    # Perform the equivalent of contract('ij,k->ijk', Rij, ck) using broadcasting
    dHij = Rij[:, :, np.newaxis] * ck[np.newaxis, np.newaxis, :]

    # Perform the equivalent of contract('ij,k->ijk', Qcij, cph) using broadcasting
    #dHij_part2 = Qcij[:, :, np.newaxis] * cph[np.newaxis, np.newaxis, :]

    # Concatenate along the third axis (axis=2)
    #dHij = np.concatenate((dHij_part1, dHij_part2), axis=2)

    # Using np.tensordot to perform the equivalent operation
    #dHij_part1 = np.tensordot(Rij, ck, axes=0)  # Equivalent to contract('ij,k->ijk', Rij, ck)
    #dHij_part2 = np.tensordot(Qcij, cph, axes=0)  # Equivalent to contract('ij,k->ijk', Qcij, cph)

    # Concatenating along the third axis (axis=2)
    #dHij = np.concatenate((dHij_part1, dHij_part2), axis=2)
    return dHij

#Gradient of the Hamiltonian that is indepndent of coupling
#In diabatic or MH (polaron == True) basis
@jit(nopython=True,fastmath=True)
def dHel0(R):
    dH0 = ωk**2 * R#[:Nmbath] 
    #dH0 = np.concatenate((dH0,ωph**2 * R[Nmbath:]))
    return dH0

#Initialization of bath modes
def initR(initC):
    R0 = - ck * (Ψm0.T @ Rij @ Ψm0) /ωk**2
    P0 = np.zeros((NBath,))#0.0
    
    #omegas = np.concatenate((ωk,ωph))
    if wigner: 
        sigP = np.sqrt(ωk/(2.0 * np.tanh(0.5 * β * ωk))) 
        sigR = sigP/(ωk) 
    else:
        sigP = np.sqrt(1/β)*np.ones_like(ωk)
        sigR = 1/np.sqrt(β*ωk**2)

    R = np.zeros(( NBath ))
    P = np.zeros(( NBath ))
    for d in range(NBath):
        R[d] = np.random.normal()*sigR[d] +R0[d]
        P[d] = np.random.normal()*sigP[d] +P0[d] 
    return R, P
