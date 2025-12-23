# Diels-Alder Aligner

A computational chemistry tool that automates the generation of pre-reaction complexes for ORCA/Gaussian scans. Takes product and reactant SMILES strings, expands along the reaction coordinate, and maps reactants onto the fragments to reduce unnecessary DFT optimisation.

### Note: endo approach is not guaranteed

## Installation

```bash
pip install numpy rdkit