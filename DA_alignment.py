# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, rdMolAlign, rdchem
import numpy as np

def smiles_to_molecules(product, reactant_a, reactant_b): # takes 3 SMILES strings, outputs 3 molecules
    product_mol = Chem.rdmolops.AddHs(Chem.rdmolfiles.MolFromSmiles(product))
    reactant_a_mol = Chem.rdmolops.AddHs(Chem.rdmolfiles.MolFromSmiles(reactant_a))
    reactant_b_mol = Chem.rdmolops.AddHs(Chem.rdmolfiles.MolFromSmiles(reactant_b))

    return product_mol, reactant_a_mol, reactant_b_mol

    
def diels_alder_identifier(product_mol, reactant_a_mol, reactant_b_mol): # takes 3 SMILES molecules, outputs a list of tuples
    matched_a_final, _ = diels_alder_index(product_mol, reactant_a_mol, reactant_b_mol)
    bond_candidates = []
    
    for atom_idx in matched_a_final:
        atom_obj = product_mol.GetAtomWithIdx(atom_idx)
        for neighbor_obj in atom_obj.GetNeighbors():
            neighbor_idx = neighbor_obj.GetIdx()
            if neighbor_idx not in matched_a_final:
                bond_candidates.append((atom_idx, neighbor_idx))
                
    return bond_candidates

def diels_alder_index(product_mol, reactant_a_mol, reactant_b_mol): # takes 3 molecules, outputs two tuples
    
    params = Chem.AdjustQueryParameters()
    params.makeBondsGeneric = True
    params.adjustDegree = False
    params.adjustRingCount = False

    query_a = Chem.AdjustQueryProperties(reactant_a_mol, params)
    query_b = Chem.AdjustQueryProperties(reactant_b_mol, params)

    matched_a_idx = product_mol.GetSubstructMatches(query_a)
    matched_b_idx = product_mol.GetSubstructMatches(query_b)

    if len(matched_a_idx) == 0 or len(matched_b_idx) == 0:
        print("did not find matching substructures")
        return null

    matched_a_final = ()
    matched_b_final = ()
    
    for match_a in matched_a_idx:
        for match_b in matched_b_idx:
            if set(match_a).isdisjoint(match_b):
                matched_a_final += match_a
                matched_b_final += match_b
    
    if type(matched_a_final[0]) is not int or type(matched_b_final[0]) is not int:
        print("too many matching substructures")
        return null

    return matched_a_final, matched_b_final

    
def diels_alder_reverse(product_mol, reactant_a_mol, reactant_b_mol, idx, separation): # takes 3 SMILES strings, 1 tuple list, 1 integer, outputs a molecule
    if len(idx) == 2:
        idx1, idx2, idx3, idx4 = idx[0][0], idx[0][1], idx[1][0], idx[1][1]

    else:
        print("more than 2 bonds found, aborting run")
        return null
    
    status = AllChem.EmbedMolecule(product_mol)

    if status == -1:
        print("embedmolecule could not generate 3D coordinates")
    else:
        AllChem.MMFFOptimizeMolecule(product_mol)
        product_conf = product_mol.GetConformer()

    bond12_vector = np.array(product_conf.GetAtomPosition(idx1)) - np.array(product_conf.GetAtomPosition(idx2))
    bond34_vector = np.array(product_conf.GetAtomPosition(idx3)) - np.array(product_conf.GetAtomPosition(idx4))
    separation_vector = separation * ((bond12_vector + bond34_vector)/np.linalg.norm(bond12_vector + bond34_vector)) # calculates reaction coordinate

    matched_a_final, _ = diels_alder_index(product_mol, reactant_a_mol, reactant_b_mol)

    for idx in matched_a_final: # moves apart reactant atoms
        p_orig = product_conf.GetAtomPosition(idx)
        new_x = p_orig.x + separation_vector[0]
        new_y = p_orig.y + separation_vector[1]
        new_z = p_orig.z + separation_vector[2]
        new_pos = Point3D(new_x, new_y, new_z)
        
        product_conf.SetAtomPosition(idx, new_pos)

    rw_mol = Chem.RWMol(product_mol)
    rw_mol.RemoveBond(idx1, idx2)
    rw_mol.RemoveBond(idx3, idx4) # breaks formed bonds
    product_mol = rw_mol.GetMol()
    Chem.SanitizeMol(product_mol)
    
    return product_mol

def diels_alder_align(product_mol_reversed, reactant_a_mol, reactant_b_mol): # takes 3 molecules, outputs one molecule
    matched_a, matched_b = diels_alder_index(product_mol_reversed, reactant_a_mol, reactant_b_mol)

    AllChem.EmbedMolecule(reactant_a_mol)
    AllChem.EmbedMolecule(reactant_b_mol)
    AllChem.MMFFOptimizeMolecule(reactant_a_mol)
    AllChem.MMFFOptimizeMolecule(reactant_b_mol)

    atom_map_a = []
    atom_map_b = []

    for i, atom_idx in enumerate(matched_a):
        atom_map_a.append((i, atom_idx))

    for i, atom_idx in enumerate(matched_b):
        atom_map_b.append((i, atom_idx))
    
    rdMolAlign.AlignMol(reactant_a_mol, product_mol_reversed, atomMap=atom_map_a)
    rdMolAlign.AlignMol(reactant_b_mol, product_mol_reversed, atomMap=atom_map_b)

    final_complex = Chem.CombineMols(reactant_a_mol, reactant_b_mol)
    return final_complex


def xyz_block(mol): # takes a molecule, outputs an xyz block
    return Chem.rdmolfiles.MolToXYZBlock(mol,-1,6) #converts molecule to xyz coordinates

prod, a, b = smiles_to_molecules("N#CC1CC2C=CC1C2", "C1=CCC=C1", "C=CC#N")

bond_candidates = diels_alder_identifier(prod, a, b)
print(bond_candidates)

reactant_idx = diels_alder_index(prod, a, b)
print(reactant_idx)

print(xyz_block(diels_alder_reverse(prod, a, b, bond_candidates, 3)))

prod_rev = diels_alder_reverse(prod, a, b, reactant_idx, 3)

print(xyz_block(diels_alder_align(prod_rev, a, b)))

def diels_alder_full(product_smi, reactant_a_smi, reactant_b_smi, separation):
    product_mol, reactant_a_mol, reactant_b_mol = smiles_to_molecules(product_smi, reactant_a_smi, reactant_b_smi)
    bond_candidates = diels_alder_identifier(product_mol, reactant_a_mol, reactant_b_mol)

    if len(bond_candidates) == 2:
        print("2 bounds found! continuing with breaking bonds between", bond_candidates)

    else:
        print("more than 2 bonds found, run aborted")
        return null

    reactant_idx = diels_alder_index(product_mol, reactant_a_mol, reactant_b_mol)
    prod_rev = diels_alder_reverse(product_mol, reactant_a_mol, reactant_b_mol, reactant_idx, separation)
    
    return xyz_block(diels_alder_align(prod_rev, reactant_a_mol, reactant_b_mol))

print(diels_alder_full("N#CC1CC2C=CC1C2", "C1=CCC=C1", "C=CC#N", 3))
    

    
    



# %%
