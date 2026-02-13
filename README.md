# Code for Performing Semi-Classical Quantum Dynamics
The methods that are implimented in this code are : PLDM (Parital Linearized Density Matrix), spin-PLDM, MFE (Mean-Field Ehrenfest), various SQC (Symmetric Quasi-Classical Approach) and N-RPMD (Nonadiabatic Ring-Polymer Molecular Dynamics). The present code works for slurm based High-Performance Computing Cluster (HPCC), HTcondor based High-Throughput Computing (HTC) as well as on personal computers.  

# Usage  
### Step 1
Create a folder and git clone this repository.
```
git clone https://github.com/arkajitmandal/SemiClassical-NAMD
```
### Step 2
Code up the model system in a python file inside the "Model" folder and name it  'modelName.py'.  

The 'modelName.py' should look like:
```py
import numpy as np

class parameters():
   # some parameters
   # Nuclear Timestep (a.u.)
   dtN = 2

   # Number of Nuclear Steps 
   # length of simulation : dtN x Nsteps
   NSteps = 600  
   
   # Number trajectories per cpu
   NTraj = 100

   # Electronic Timestep (a.u.)
   # Please use even number
   dtE = dtN/40

   # Mass of nuclear particles in a.u.
   # is a vector of length NR : number 
   # of nuclear DOF
   M = np.array([1836.0])

   # Initial electronic state
   initState = 0
   
   # Save data every nskip steps
   nskip = 5

def Hel(R):
    # Diabatic potential energies Vij(R) in a.u., 
    # return Vij(R) : NxN Matrix 
    # (N  = number of electronic states)
    # R : Vector of length NR
    # (NR = number of nuclear degrees of freedom)
    # See details below 
    return Vij

def dHel0(R):
    # State independent gradient 
    # return dV0(R) : Vector of length NR
    # (NR = number of nuclear degrees of freedom) 
    # See details below
    return dV0

def dHel(R):
    # Gradient of the diabatic
    # potential energy matrix elements
    # Vij(R) in a.u.  
    # return dVij(R) : Array of dimention (N x N x NR)
    # See details below
    return dVij

def initR():
    # Provide initial values of R, P in a.u.
    # R, P : Both are vectors of length NR
    # (NR = number of nuclear degrees of freedom) 
    # See details below
    return R, P

def initHel0(R):
    #------ This part will be only for NRPMD----------
    #-------while running condensed phase system------
    #R : is a 2D array of dimensionality ndof,nb
         #where, nb is the bead and ndof is the number of dofs
    #M : mass of the particle
    #ω = frequency of the particle
    #R0 = initial positon of the photo-excitation 
    # see details below
    return  np.sum(0.5 *M* ω**2 * (R-R0)**2.0)
```

You can find several examples of model files inside the "Model" folder. I will explain each parts of this file in more detain in a section below.


### Step 3 (simple | serial; on your pc/mac/linux)
Prepare an input file (name it : 'whateverInput.txt'):
```
Model                = tully2
Method               = pldm-focused 
```

* Model : The right hand side of the first line, _tully2_, tells the code to look for tully2.py inside the folder "Model". If you name your model file as  modelName.py then you should write 'Model = modelName' (without the '.py' part). 
* Method : Written as, method-methodOption. Select a quantum dynamics method. The available methods are :
  - **mfe** : Mean-Field Ehrenfest Approach [8]. Kind of worst approach you can think of.
   - **pldm-focused** : Partial Linearized Density Matrix (PLDM) [1] with focused initial conditions. Should be similar to mfe. Maybe slightly better. 
   - **pldm-sampled** : Partial Linearized Density Matrix (PLDM) [1] with sampled initial conditions or the original PLDM approach. Most of the time works well, sometimes does not. Very good if your potentials are Hermonic (like Spin-Boson systems)
   - **spinpldm-all**: The Spin-Mapping PLDM [2] approach with full sampling. Often better than PLDM. Reliable but slighly slow. If your initial electronic state is a pure state |i⟩⟨i| (you could start from a super position state, but you have to hack into this code to do that) use spinpldm-half to get the same result but much faster (by half).
   - **spinpldm-half**: The Spin-Mapping PLDM approach, but with our in-house approximation. Works perfectly if starting with an initial electronic state that is a pure state |i⟩⟨i| (you could start from a super position state, but you have to hack into this code to do that). 
   - **spinpldm-focused**: The Spin-Mapping PLDM approach, approximated. Good for short-time calculation and to get a general trend for longer time. 
   - **sqc-square**: The Symmetric Quasi-Classical Approach, with square window [3]. Better than MFE. Cannot use it for more than several electronic states.  
   - **sqc-triangle**: The Symmetric Quasi-Classical Approach, with triangle window [4]. Better than sqc-square.   
   - **zpesqc-triangle**: The zero-point energy corrected Symmetric Quasi-Classical Approach [5], with triangle window. As good as spin-PLDM or better.  
   - **zpesqc-square**: The zero-point energy corrected Symmetric Quasi-Classical Approach [5], with square window. Slightly worse than zpesqc-triangle.
   - **spinlsc**: Spin-LSC approach [13,14], sort of simpler version of Spin-PLDM. I think this is actually a great method. 

   - **nrpmd-n** : The non-adiabatic ring polymer molecular dynamics[6] framework for aims to captures nuclear quantum effects while predicting efficient short-time and reliable longer time
   dynamics. Reasonable results for electron/charge transfer dynamics. Here n represents the number of beads, i.e. nrpmd-5 means each nuclear degrees of freedom is described with 5 ring-polymer beads.  

   - **mash** : Multistate Mapping Approach to Surface Hopping approach. [7,17] 

The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_pldm-focused-tully2.txt_

### Step 3 (advanced | parallel jobs; slurm on HPCC) 
Prepare an input file (name it : 'whateverInput.txt') for slurm submission in your computing cluster:
```
Model                = tully2
Method               = pldm-focused 

System               = slurm
Nodes                = 2
Cpus                 = 24
Partition            = action    
```
For first two lines see previous section. 

Last four lines provide additional commands for slurm submission. For adding additional slurm ('#SBATCH') command, add them in the preamble of the 'serial.py'. The default preamble looks like:
```py
#!/usr/bin/env python
#SBATCH -o output.log
#SBATCH -t 1:00:00
```
Please dont add lines like "#SBATCH -N 1" or 
"#SBATCH --ntasks-per-node=24" in the preamble as they are declared in the input file. 


The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_pldm-focused-tully2.txt_

### Step 3 (advanced | parallel jobs; htcondor on HTC) 
Prepare an input file (name it : 'whateverInput.txt') for slurm submission in your computing cluster:
```
Model                = morse1
Method               = mfe

System               = htcondor
Cpus                 = 10

```
For first two lines described the model and method. 

Last line provide just as before indicate the number of cpus to be used.
After all jobs are done, run the following python script to get the output file. 
```
$ python avg.py
```

The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_mfe-morse1.txt_


### Step 4 

Run the this code with python3. 

```
python3 run.py whateverInput.txt
```
Where 'whateverInput.txt' is the input file described above.
If your inputfile is named 'input.txt' then you could also just run,
```
python3 run.py
```

# Details of Model Hamiltonian
In all of the approaches coded up here, the nuclear DOF $\{R_k, P_k\}$ are evolved classically (their equation motion evolves under a classical like force) and the electronic DOF are described with the diabatic electronic states { $|i\rangle$ }.  

A molecular Hamiltonian in the diabatic representation is written as:

$$\hat{H} = \frac{P_k^2}{2M_k} + V_{0}(  \\{   R_k \\}  ) + \sum_{ij}V_{ij}(\\{R_k\\})|i \rangle \langle j| = \sum_{k} T_{R_k} + \hat{H}_{el}(\\{R_k\\})$$


where $P_k$ is the momentum for the $k$ th nuclear degrees of freedom with mass $M_k$. Further, $V_{0}(\\{R_k\\})$ and  $V_{ij}(\\{R_k\\})$ are the state-independent and state-dependent part of the electronic Hamiltonian in the diabatic basis { $|i \rangle$ }. That is: 

$$ \langle i| \hat{H}_{el} (\\{R_k\\}) |j\rangle = V_0 (\\{R_k\\}) \delta\_{ij} + V\_{ij}(\\{R_k\\})$$


Write the analytical form of $V_{ij}(\\{R_k\\})$ you can write a model file: **modelName.py**. 


One can always, set $V_{0}(\\{R_k\\})= 0$, and instead redefine $V_{ij}(\\{R_k\\}) \rightarrow V_{0}(\\{R_k\\})\delta_{ij} + V_{ij}(\\{R_k\\})$ and they should be equivalent in principle. However, some of the semiclassical approaches (**pldm-sampled**, **sqc-square** and **sqc-triangle**) produce results that depend on how one separates the state-independent and state-dependent parts of the gradient of the electronic Hamiltonian. The nuclear forces computed in all of these approaches assumes this general form:

$F_k = - \nabla_k V_{0}(\\{R_k\\}) - \sum_{ij}  \nabla_k V_{ij}(\\{R_k\\}) \cdot \Lambda_{ij}$

where the definition of $\Lambda_{ij}$ depends on the quantum dynamics method. For example, in MFE, $\Lambda_{ij} = c_i^* c_j$.  For methods that have ∑<sub>i</sub>Λ<sub>ii</sub> = 1 (like MFE) for individual trajectories this separation of state-dependent and independent does not matter. 

## Details of a model file ('modelName.py')
**_NOTE:_** You **dont** need to code up  $V_{0}(\\{R_k\\})$. 

### Hel(R)
In the Hel(R) function inside the 'modelName.py' one have to define NxN matrix elements of the state-dependent electronic part of the Hamiltonian. Here you will code up  $V_{ij}(\\{R_k\\})$.

### dHel(R)
In the dHel(R) function inside the 'modelName.py' one have to define NxNxNR matrix elements of the state-dependent gradient electronic part of the Hamiltonian. Here you will code up  $\nabla_k V_{ij}(\\{R_k\\})$.

### dHel0(R)
In the dHel0(R) function inside the 'modelName.py' one have to define a array of length NR describing the state-independent gradient of electronic part of the Hamiltonian. Here you will code up  $\nabla_k V_{0}(\\{R_k\\})$.


### initR()
Sample $R, P$ from a Wigner distribution. To obtain the wigner distribution, one needs to start with an initial density matrix. For example, for an wavefunction $|\chi \rangle$ write the density matrix $\hat{\rho}_N = |\chi  \rangle \langle \chi|$, then the Wigner transform is performed as,

${\hat{\rho}^W_N}({R, P}) = \frac{1}{\pi\hbar} \int_{-\infty}^{\infty} \langle {R} - \frac{S}{2}|\hat{\rho}_N |{R} + \frac{S}{2} \rangle e^{iPS} dS$

The $R, P$ is then sampled from $\hat{\rho}_N^{W}({R, P})$.

_____________

## Methods

All semiclassical methods implemented here propagate nuclei classically while treating electronic degrees of freedom through various mapping or mean-field strategies. The key distinction between methods lies in how they represent the electronic DOF and compute the nuclear force kernel $\Lambda_{ij}$. Below we summarize each method's key ideas and equations.

### MFE (Mean-Field Ehrenfest) [8]

The Ehrenfest method is the simplest mixed quantum-classical approach: nuclei evolve on a mean-field potential energy surface averaged over all electronic states, weighted by the electronic amplitudes. The electronic coefficients $c_i$ are propagated via the time-dependent Schrodinger equation $i\hbar \dot{c}_i = \sum_j H_{ij}^{el} c_j$. The nuclear force kernel is $\Lambda_{ij} = c_i^* c_j$, so the force on each nucleus is:

$$F_k = -\nabla_k V_0 - \sum_{ij} \nabla_k V_{ij} \cdot c_i^* c_j$$

While computationally cheap and straightforward, Ehrenfest dynamics does not satisfy detailed balance and can yield incorrect long-time populations because the nuclei never fully commit to a single electronic state.

### PLDM (Partial Linearized Density Matrix) [1,11]

PLDM uses the Meyer-Miller-Stock-Thoss (MMST) [9,10] classical mapping Hamiltonian to replace discrete electronic states with continuous harmonic oscillator variables $(q_i, p_i)$. It employs a forward-backward decomposition with two independent sets of mapping variables $(q_F, p_F)$ and $(q_B, p_B)$ that propagate the ket and bra sides of the density matrix, respectively. The reduced density matrix is estimated as:

$$\rho_{ij}(t) \propto \left\langle \left(q_F^i(t) + i p_F^i(t)\right) \left(q_B^j(t) - i p_B^j(t)\right) \right\rangle$$

The nuclear force kernel takes the symmetrized form $\Lambda_{ij} = \tfrac{1}{4}(q_F^i q_F^j + p_F^i p_F^j + q_B^i q_B^j + p_B^i p_B^j)$. With "focused" initial conditions, PLDM reduces approximately to Ehrenfest; with "sampled" initial conditions, it captures more quantum coherence effects and is particularly accurate for harmonic (e.g., spin-boson) models.

### Spin-PLDM (Spin-Mapping PLDM) [2,12]

Spin-PLDM replaces the MMST mapping with spin-mapping variables derived from the Stratonovich-Weyl transform of SU($N$) coherent states. The mapping variables are constrained to lie on a hypersphere, which eliminates the zero-point energy leakage problem that plagues MMST-based methods. The approach uses forward-backward complex mapping variables $z_F$ and $z_B$, and the key parameter is the spin-mapping zero-point energy:

$$g_w = \frac{2}{N}\left(\sqrt{N+1} - 1\right)$$

The population estimator takes the form $\hat{A}_{ij} \propto z_F^{i*} z_F^{j} - g_w \delta_{ij}$. Three variants are available: "all" (full forward-backward sampling), "half" (an efficient approximation valid for pure initial states), and "focused" (best for short-time dynamics).

### Spin-LSC (Spin Linearized Semiclassical) [13,14]

Spin-LSC is the fully linearized (single-trajectory) version of spin-mapping dynamics, using only a single set of forward mapping variables $z$ rather than the forward-backward pair used in spin-PLDM. It can be viewed as a spin-mapping generalization of the classical Wigner (linearized semiclassical) approach. The reduced density matrix is estimated as:

$$\rho_{ij}(t) = \frac{1}{2}\left\langle z^{i*}(t)\, z^{j}(t) - g_w\, \delta_{ij} \right\rangle$$

Despite its simplicity compared to spin-PLDM, spin-LSC provides remarkably accurate results for many condensed-phase models and is computationally efficient since it requires only half the mapping degrees of freedom.

### SQC (Symmetric Quasi-Classical) [3,4]

The SQC approach, developed by Cotton and Miller, uses the MMST mapping Hamiltonian with symmetrical windowing functions applied to the electronic action variables at both the initial and final times. For each electronic state, the action variable is defined as $n_i = \frac{1}{2}(q_i^2 + p_i^2) - \gamma$, where $\gamma$ is a zero-point energy parameter. The population of state $i$ at time $t$ is assigned as 1 if $n_i(t)$ falls within a specified window and 0 otherwise. Two window functions are implemented:

$$\text{Square: } \gamma = \frac{\sqrt{3}-1}{2}, \qquad \text{Triangle: } \gamma = \frac{1}{3}$$

The triangle window generally outperforms the square window, especially for systems with many electronic states.

### ZPE-SQC (Zero-Point Energy Corrected SQC) [5]

ZPE-SQC addresses a key limitation of standard SQC by adjusting the zero-point energy parameter $\gamma$ on a per-trajectory basis according to the instantaneous electronic state, preventing spurious zero-point energy flow between electronic states. Instead of using a fixed $\gamma$ for all states, the correction defines a trajectory-specific parameter:

$$\gamma_0^{(i)} = n_i(0) - \delta_{i,\,\mathrm{init}}$$

where $n_i(0)$ is the initial action variable for state $i$ and $\delta_{i,\mathrm{init}}$ is 1 for the initially occupied state and 0 otherwise. This ensures the initial nuclear force matches the correct quantum electronic state. ZPE-SQC achieves accuracy comparable to or better than spin-PLDM for many benchmark models.

### N-RPMD (Nonadiabatic Ring-Polymer Molecular Dynamics) [6,15,16]

N-RPMD combines the ring-polymer path-integral representation of nuclear degrees of freedom with MMST mapping variables for the electronic states, enabling the simultaneous capture of nuclear quantum effects (tunneling, zero-point energy) and nonadiabatic transitions. Each nuclear DOF is represented by $n_b$ ring-polymer beads connected by harmonic springs, and each bead carries its own set of electronic mapping variables. The ring-polymer spring potential is:

$$V_{\mathrm{RP}} = \sum_{b=1}^{n_b} \frac{1}{2} M_k \omega_{n_b}^2 \left(R_k^{(b)} - R_k^{(b+1)}\right)^2, \qquad \omega_{n_b} = \frac{n_b}{\beta \hbar}$$

The electronic population is obtained by averaging the mapping-variable estimator over all beads: $\rho_{ij} = \frac{1}{n_b}\sum_{b} \frac{1}{2}(q_i^{(b)} q_j^{(b)} + p_i^{(b)} p_j^{(b)} - \delta_{ij})$. Normal-mode transformations are used to efficiently propagate the free ring-polymer dynamics.

### MASH (Mapping Approach to Surface Hopping) [7,17]

MASH is derived rigorously from the quantum-classical Liouville equation and propagates nuclei on a single active adiabatic potential energy surface, with deterministic (not stochastic) hops between surfaces. The active state at any time is determined by the electronic coefficient with the largest amplitude, $a = \arg\max_i |c_i|^2$, and the nuclear force is simply the gradient of the active-state adiabatic potential. The population estimator combines a projection-like term with a uniform correction:

$$\hat{\rho}_{ij} = \alpha \, c_i \, c_j^* + \beta \, \delta_{ij}, \qquad \alpha = \frac{N-1}{\displaystyle\sum_{n=1}^{N} \frac{1}{n} - 1}, \qquad \beta = \frac{\alpha - 1}{N}$$

Upon a surface hop, the nuclear momentum is rescaled along the nonadiabatic coupling direction to conserve total energy. MASH is guaranteed to satisfy detailed balance and correctly recovers Marcus theory rate constants without requiring decoherence corrections.

## Authors
* Arkajit Mandal
* Braden Weight
* Sutirtha Chowdhury
* Eric Koessler
* Elious M. Mondal
* Haimi Nguyen
* Muhammad R. Hasyim
* Wenxiang Ying
* James F. Varner

## References
_____________
[1] P. Huo and D. F. Coker, __J. Chem. Phys. 135, 201101 (2011)__\
[2] J. R. Mannouch and J. O. Richardson, __J. Chem. Phys. 153, 194109 (2020)__\
[3] S. J. Cotton and W. H. Miller, __J. Chem. Phys. 139, 234112 (2013)__\
[4] S. J. Cotton and W. H. Miller, __J. Chem. Phys. 145, 144108 (2016)__\
[5] S. J. Cotton and W. H. Miller, __J. Chem. Phys. 150, 194110 (2019)__\
[6] S. N. Chowdhury and P. Huo, __J. Chem. Phys. 147, 214109 (2017)__\
[7] J. R. Mannouch and J. O. Richardson, __J. Chem. Phys. 158, 104111 (2023)__\
[8] J. C. Tully, __J. Chem. Phys. 93, 1061 (1990)__\
[9] H.-D. Meyer and W. H. Miller, __J. Chem. Phys. 70, 3214 (1979)__\
[10] G. Stock and M. Thoss, __Phys. Rev. Lett. 78, 578 (1997)__\
[11] P. Huo and D. F. Coker, __J. Chem. Phys. 137, 22A535 (2012)__\
[12] J. R. Mannouch and J. O. Richardson, __J. Chem. Phys. 153, 194110 (2020)__\
[13] J. E. Runeson and J. O. Richardson, __J. Chem. Phys. 151, 044119 (2019)__\
[14] J. E. Runeson and J. O. Richardson, __J. Chem. Phys. 152, 084110 (2020)__\
[15] N. Ananth, __J. Chem. Phys. 139, 124102 (2013)__\
[16] J. O. Richardson and M. Thoss, __J. Chem. Phys. 139, 031102 (2013)__\
[17] J. E. Runeson and D. E. Manolopoulos, __J. Chem. Phys. 159, 094115 (2023)__


email: mandal@tamu.edu
