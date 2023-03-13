# %%
import cv2

# %%
hdr_path = "/Users/nkusanda/Documents/GitHub/aps360-group-45/Data/tea_NIR_17.hdr"
img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)

# %%
import imageio
# The following line only needs to run once for a user
# to download the necessary binaries to read HDR.
imageio.plugins.freeimage.download()
img = imageio.imread(hdr_path, format='HDR-FI')

# %%
bin_path = "/Users/nkusanda/Documents/GitHub/aps360-group-45/Data/tea_NIR_1,7.bin"
with open(bin_path, mode='rb') as file: # b is important -> binary
    fileContent = file.read()
# %%
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdChemReactions
RDLogger.DisableLog("rdApp.*")

import selfies
import sys
import pandas as pd
import numpy as np

# SAScore import
from SAS_calculator.sascorer import calculateScore

# Buried volume imports
from morfeus import conformer, BuriedVolume, read_xyz
from morfeus.utils import get_radii

# Redox potential imports
from pathlib import Path
from morfeus.io import write_xyz
from rdkit.Chem import GetFormalCharge
import os, re, sys, subprocess
import shutil
from random import gauss  
import time




# %%
def stitch_diquat(pyr_smi):
    stitch_pyridines = rdChemReactions.ReactionFromSmarts(
        '[c:1]1[n;x2][c:2]([Br])[c:3][c:4][c:5]1.\
            [c:6]1[n;x2][c:7]([Br])[c:8][c:9][c:10]1\
                >>[c:3]1[c:4][c:5][c:1][n+]2[c:2]1-[c:7]1[c:8][c:9][c:10][c:6][n+]1CC2')

    pyridine1 = pyridine2 = Chem.MolFromSmiles(pyr_smi) # ensures symmetry
    reacts = (pyridine1,pyridine2)
    products = stitch_pyridines.RunReactants(reacts)
    m = products[0][0]
    Chem.SanitizeMol(m)
    s = Chem.MolToSmiles(m)

    return m, s # mol, smile
# %%
def redox_potential(elements, coordinates, i, m):
    # TODO: what does 'i' mean for JANUS? Just use placeholder 0 for now
    # Maybe delete created folder after each iteration?
    # Defining directory
    cwd = Path.cwd() / f"{i}"
    cwd.mkdir()

    # Defining environment such that XTB doesn't use more than 1 core
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1,1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_MAX_ACTIVE_LEVELS"] = "1"

    # Timeout handler for subprocess calls
    def timeout_handler(cwd):
        max_norm_spin = [float("nan")] * 6
        max_spin_idx = [float("nan")] * 6
        redox_pot = -404 # identify timeouts

        retry = True
        try_num = 1
        while (retry==True):
            try:
                shutil.rmtree(str(cwd))
                retry = False
            except:
                try_num += 1
                if try_num > 3:
                    shutil.rmtree(str(cwd), ignore_errors=True)
                    retry = False

        return redox_pot, max_norm_spin, max_spin_idx

    # i. Write XYZ file
    write_xyz(cwd / "smi.xyz", elements, coordinates)

    # ii. Run XTB on XYZ to produce optimal oxidized structure
    chrg_ox = GetFormalCharge(m)
    chrg_red = chrg_ox - 1

    try:
        args = "xtb smi.xyz --chrg " + str(chrg_ox) + " --uhf 0 --ohess --norestart --alpb water"
        with open(cwd / "ox.txt", "w") as file:
            subprocess.run([args], shell=True, env=env, cwd=cwd, stdout=file, timeout=300)
    except subprocess.TimeoutExpired:
        redox_pot, max_norm_spin, max_spin_idx = timeout_handler(cwd)
        return redox_pot, max_norm_spin, max_spin_idx

    # iii. Run XTB on optimized XYZ to produce reduced structure
    try:
        args = "xtb xtbopt.xyz --chrg " + str(chrg_red) + " --uhf 1 --ohess --norestart --alpb water"
        with open(cwd / "red.txt", "w") as file:
            subprocess.run([args], shell=True, env=env, cwd=cwd, stdout=file, timeout=300)
    except subprocess.TimeoutExpired:
        redox_pot, max_norm_spin, max_spin_idx = timeout_handler(cwd)
        return redox_pot, max_norm_spin, max_spin_idx
        

    # iv. Extract free energy from XTB output files
    # TODO: is there a way to directly extract the total free energy without re-opening the file?
    with open(cwd / "ox.txt", "r") as file:
        ox = file.read()
    with open(cwd / "red.txt", "r") as file:
        red = file.read()

    # v. Compute PM7 spin density from optimized reduced structure
    try:
        # xtbopt.xyz here is from the reduced calculation
        args = "obabel -ixyz xtbopt.xyz -omop -O pm7.mop"
        # Convert optimized xyz to mopac file format
        subprocess.run([args], shell=True, env=env, cwd=cwd, timeout=100)
    except subprocess.TimeoutExpired:
        redox_pot, max_norm_spin, max_spin_idx = timeout_handler(cwd)
        return redox_pot, max_norm_spin, max_spin_idx

    try:
        # Inputs parameters for MOPAC calculation
        args = "sed -i '1c PM7 CHARGE\={} 1SCF' pm7.mop".format(chrg_red)
        subprocess.run([args], shell=True, env=env, cwd=cwd, timeout=100)
    except subprocess.TimeoutExpired:
        redox_pot, max_norm_spin, max_spin_idx = timeout_handler(cwd)
        return redox_pot, max_norm_spin, max_spin_idx

    try:
        # Run MOPAC
        args = "mopac pm7.mop"
        subprocess.run([args], shell=True, env=env, cwd=cwd,timeout=100)
    except subprocess.TimeoutExpired:
        redox_pot, max_norm_spin, max_spin_idx = timeout_handler(cwd)
        return redox_pot, max_norm_spin, max_spin_idx

    with open(cwd / "pm7.out", "r") as file:
        spin = file.read() 

    spin_dict = {}
    # @Nat: I am not sure how you want to error handle MOPAC
    if (len(re.findall("JOB ENDED NORMALLY", spin)) != 0):
        reg = r'(?=ATOMIC ORBITAL SPIN POPULATIONS)([\s\S]*?)(?=JOB ENDED NORMALLY)'
        # Extract area of file where spin information is
        m = re.findall(reg, spin)[0].splitlines()[4:-4] 
        # Remove all spins related to hydrogen
        m = [x for x in m if "H" not in x]

        for ele in m:
            # Extract element, atom number and spin density
            tmp = ele.split()[0:3]
            # Zero-index: 1C becomes 0C
            new_key = str(int(tmp[0])-1) + tmp[1]
            # {'0C' : -0.1234}
            spin_dict[new_key] = tmp[2]

        # I can propose that we use the Paton way of normalizing the spin here and output max spin
        # All spins are made absolute, summed then normalized
        spin_norm = 0
        norm_spin_dict = {}
        for val in spin_dict.values():
            spin_norm += abs(float(val))
        
        for key in spin_dict.keys():
            norm_spin_dict[key] = abs(float(spin_dict[key]))/spin_norm

        # Taking top 6 spin densities, indices
        values = list(norm_spin_dict.values())
        max_spin_idx = np.argsort(values)[::-1][:6]
        max_norm_spin = np.array(values)[max_spin_idx]
    else:
        max_norm_spin = [float("nan")] * 6
        max_spin_idx = [float("nan")] * 6

    shutil.rmtree(str(cwd))

    # vi. Compute redox potential using free energies
    # check if optimization completed successfully, else output NaN for object
    if len(re.findall("finished run on.*", red)) == 0 or len(re.findall("finished run on.*", ox)) == 0:
        redox_pot = float("nan")
    else:
        gibbs_ox = re.findall("TOTAL FREE ENERGY.*", ox)[0].split()[3]
        gibbs_red = re.findall("TOTAL FREE ENERGY.*", red)[0].split()[3]
        redox_pot = (float(gibbs_ox) - float(gibbs_red))*27.2114-4.846-4.281
    
    return redox_pot, max_norm_spin, max_spin_idx
# %%
def buried_vol(elements, coordinates, max_spin_idx):
    # i. Convert to Paton's version of Bondi radii
    radii = get_radii(elements, radii_type="bondi")
    radii = [1.09 if radius == 1.20 else radius for radius in radii]

    # ii. Compute buried volume at radical center
    bv = BuriedVolume(elements, coordinates, max_spin_idx, include_hs=True, radii=radii, radii_scale=1.0) 
    bv_percent = bv.fraction_buried_volume * 100

    return bv_percent
# %%
def generate_params():
    """
    Parameters for initiating JANUS. The parameters here are picked based on prior 
    experience by the authors of the paper. 
    """

    params_ = {}

    # Record data from every generation in individual directories
    params_["verbose_out"] = True

    # Number of iterations that JANUS runs for:
    params_["generations"] = 200  # 200

    # The number of molecules for which fitness calculations are done, within each generation
    params_["generation_size"] = 50  # 5000

    # Location of file containing SMILES that will be user for the initial population.
    # NOTE: number of smiles must be greater than generation size.
    # params_["start_population"] = "./DATA/C#C_STONED_fixed_220505.txt"
    params_["start_population"] = "./DATA/pyridines_80.txt"

    # Number of molecules that are exchanged between the exploration and exploitation
    # componenets of JANUS.
    params_["num_exchanges"] = 5

    # An option to generate fragments and use then when performing mutations.
    # Fragments are generated using the SMILES provided for the starting population.
    # The list of generated fragments is stored in './DATA/fragments_selfies.txt'
    params_["use_fragments"] = True  # Set to true

    # An option to use a classifier for sampling. If set to true, the trailed model
    # is saved at the end of every generation in './RESULTS/'.
    params_["use_NN_classifier"] = True  # Set this to true!

    # Number of top molecules to conduct local search
    params_["top_mols"] = 5

    # Number of randomly sampled SELFIE strings from alphabet
    params_["num_sample_frags_mutation"] = 100

    # Number of samples from random mutations in exploration population
    params_["explr_num_random_samples"] = 100

    # Number of random mutations in exploration population
    params_["explr_num_mutations"] = 100

    # Number of samples from random mutations in exploitation population
    params_["exploit_num_random_samples"] = 100

    # Number of random mutations in exploitation population
    params_["exploit_num_mutations"] = 100

    # Number of random crossovers
    params_["crossover_num_random_samples"] = 5  # 1

    # Use discriminator to modify fitness
    params_["use_NN_discriminator"] = False

    # Optional filter to ensure mutations do not create unwanted molecular structures
    params_["filter"] = True
    print('params')
    return params_

def fitness_function(smiles: str) -> float:
    try: 
        # 1) Calculating SAScore based on pyridine to discard unsynthesizable molecules
        pyr_mol = Chem.MolFromSmiles(smiles)
        sas_val = calculateScore(pyr_mol)

        # 2) Stitching diquat from pyridine
        m, s = stitch_diquat(smiles)

        # 3) Converting diquat smiles to xyz
        try:
            ce = conformer.ConformerEnsemble.from_rdkit(s, optimize="MMFF94")
            ce.prune_rmsd()
            # finds, uses lowest energy conformation
            ce.sort()
            conformation = ce[0]
            elements = ce.elements
            coordinates = conformation.coordinates
        except:
            print("Smile: {0}, Conformer Search FAILED".format(s))
            return (10000,-1000,1000,1000) 

        # 4) Computing redox potential, radical center with XTB
        i = abs(int(100000 * gauss(0,1)))
        redox, max_spin, max_spin_idx = redox_potential(elements, coordinates, i, m)
        
        #print(smiles, redox, max_spin, max_spin_idx)
        if (redox == float("nan")) or (redox == -404) or (float("nan") in max_spin) or (float("nan") in max_spin_idx):
        #if (redox == float("nan")) or (redox == -404) or (max_spin[0] == float("nan")) or (max_spin_idx[0] == float("nan")):
            if redox == -404:
                print("Molecule timed out: ", smiles)
            return (10000,-1000,1000,1000) 
        target = -1
        redox_dft = 0.72460394 * redox - 0.2828738736384303 # linear regression for xtb -> dft
        
        # 5) Computing buried volume with MORFEUS
        # 6) Computing RSS for each of top 6 spins, taking minimum
        rss_vals = []
        for i_spin in range(len(max_spin)):
            bv_percent = buried_vol(elements, coordinates, max_spin_idx[i_spin]) # MORFEUS uses 1 indexing
            rss = bv_percent + 50 * (1 - max_spin[i_spin])
            rss_vals.append(rss)
        rss_val = min(rss_vals)
        print(smiles, redox_dft, rss_val)

        # 7) Computing number of heavy atoms in pyridine molecules
        size_val = pyr_mol.GetNumHeavyAtoms()
        
        #fitnesses = (redox_val, rss_val, sas_val, size_val)
        fitnesses = (redox_dft, rss_val, sas_val, size_val)
        return fitnesses
    except:
        print(smiles, '\t--------------------failed--------------------')
        return (10000,-1000,1000,1000)       # for maximizing the objective (minimizing the function)

def main():
    print('name=main')
    params = generate_params()

    properties = ['redox', 'rss', 'sascore', 'size']
    objectives = ['min', 'max', 'min', 'min']
    kind = 'Chimera'
    supplement = [0.0025,83,4,0]

    agent = JANUS(
        work_dir='RESULTS', 
        num_workers = 128,
        fitness_function = fitness_function,
        properties = properties,
        objectives = objectives,
        kind = kind,
        supplement = supplement, 
        **params
    )

    agent.run()
    print('done!')

if __name__ == '__main__':
    main()




