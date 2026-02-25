from concurrent.futures import ThreadPoolExecutor
import copy
import sys
import traceback
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random

import yaml
MINIMUM = 1e-10
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import requests

from rdkit.Chem import QED
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

load_dotenv()

# client = OpenAI(api_key=os.getenv("GPT_KEY"))
client = OpenAI(base_url="https://gpt-oss-120b-svarambally.nrp-nautilus.io/v1", api_key=os.getenv("OSS_KEY"))
API_URL = "https://andrew-boltz-api.nrp-nautilus.io/predict_affinity"

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors


ALKYL_FRAGMENTS = {
    "methyl": "[*:1]C",
    "ethyl": "[*:1]CC",
    "propyl": "[*:1]CCC",
    "isopropyl": "[*:1]C(C)C",
    "tert_butyl": "[*:1]C(C)(C)C",
    "cyclopropyl": "[*:1]C1CC1",
    "cyclobutyl": "[*:1]C1CCC1",
    "cyclopentyl": "[*:1]C1CCCC1",
    "cyclohexyl": "[*:1]C1CCCCC1"
}

HALOGEN_FRAGMENTS = {
    "fluoro": "[*:1]F",
    "chloro": "[*:1]Cl",
    "bromo": "[*:1]Br",
    "iodo": "[*:1]I"
}

HETEROATOM_FRAGMENTS = {
    "hydroxyl": "[*:1]O",
    "methoxy": "[*:1]OC",
    "ethoxy": "[*:1]OCC",
    "amine": "[*:1]N",
    "methylamine": "[*:1]NC",
    "dimethylamine": "[*:1]N(C)C",
    "thiol": "[*:1]S",
    "methylthio": "[*:1]SC"
}

CARBONYL_FRAGMENTS = {
    "aldehyde": "[*:1]C=O",
    "ketone_methyl": "[*:1]C(=O)C",
    "carboxylic_acid": "[*:1]C(=O)O",
    "ester_methyl": "[*:1]C(=O)OC",
    "amide": "[*:1]C(=O)N",
    "amide_methyl": "[*:1]C(=O)NC",
    "urea": "[*:1]NC(=O)N",
    "carbamate": "[*:1]OC(=O)N"
}

POLAR_FRAGMENTS = {
    "hydroxymethyl": "[*:1]CO",
    "aminoethyl": "[*:1]CCN",
    "dimethylaminoethyl": "[*:1]CCN(C)C",
    "morpholine": "[*:1]N1CCOCC1",
    "piperazine": "[*:1]N1CCNCC1",
    "piperidine": "[*:1]C1CCNCC1"
}

LARGE_RING_FRAGMENTS = {
    # Aromatics
    "phenyl": "[*:1]c1ccccc1",
    "benzyl": "[*:1]Cc1ccccc1",
    "phenoxy": "[*:1]Oc1ccccc1",
    "benzoyl": "[*:1]C(=O)c1ccccc1",
    
    # Heteroaromatics (Single Ring)
    "pyridyl_4": "[*:1]c1ccncc1",         # Attached at 4-position
    "pyridyl_3": "[*:1]c1cnccc1",         # Attached at 3-position
    "pyridyl_2": "[*:1]c1ncccc1",         # Attached at 2-position
    "pyrimidinyl": "[*:1]c1ncccn1",
    "thienyl": "[*:1]c1sccc1",
    "furanyl": "[*:1]c1occc1",
    
    # Fused Aromatics & Heterocycles
    "naphthyl_1": "[*:1]c1cccc2ccccc12",
    "naphthyl_2": "[*:1]c1ccc2ccccc2c1",
    "indole_3": "[*:1]c1c[nH]c2ccccc12",  # Common attachment point (tryptophan-like)
    "indole_N": "[*:1]n1ccc2ccccc12",     # Attached at Nitrogen
    "benzimidazole": "[*:1]c1nc2ccccc2[nH]1",
    "quinoline": "[*:1]c1cccc2ncccc12",
    "isoquinoline": "[*:1]c1cncc2ccccc12",
    "benzodioxole": "[*:1]c1ccc2OCOc2c1", # Often used to improve solubility/metabolic stability
    "benzofuran": "[*:1]c1oc2ccccc2c1",

    # Bulky/Bicyclic Aliphatics
    "adamantyl": "[*:1]C12CC3CC(CC(C3)C1)C2", # Very bulky lipophilic group
    "norbornyl": "[*:1]C1CC2CCC1C2",
    "biphenyl": "[*:1]c1ccc(cc1)c2ccccc2"
}

FRAGMENTS = {
    **ALKYL_FRAGMENTS,
    **HALOGEN_FRAGMENTS,
    **HETEROATOM_FRAGMENTS,
    **CARBONYL_FRAGMENTS,
    **POLAR_FRAGMENTS,
    **LARGE_RING_FRAGMENTS
}


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{smiles}: Invalid SMILES")
    Chem.SanitizeMol(mol)
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def validate_smiles(smiles: str) -> dict:
    """Validates SMILES string and returns basic properties."""
    try:
        mol = mol_from_smiles(smiles)
        return {
            "is_valid": True,
            "canonical_smiles": canonical_smiles(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds()
        }
    except Exception as e:
        return {"is_valid": False, "error": str(e)}


def get_attachment_points(smiles: str) -> dict:
    """Identifies atoms with available hydrogens for substitution."""
    try:
        mol = mol_from_smiles(smiles)
        pts = []

        for atom in mol.GetAtoms():
            h_count = atom.GetTotalNumHs()
            if h_count > 0:
                pts.append({
                    "atom_index": atom.GetIdx(),
                    "element": atom.GetSymbol(),
                    "substitutable_hydrogens": h_count,
                })
        return {"attachment_points": pts}
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_atom(
    smiles: str,
    target_atom_index: int,
    new_atom: str,
    bond_type: str = "SINGLE"
) -> dict:
    """
    Adds a single atom to the molecule.
    """
    try:
        mol = mol_from_smiles(smiles)
        
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        if target_atom.GetTotalNumHs() == 0:
            return {
                "success": False, 
                "error": f"Atom {target_atom_index} ({target_atom.GetSymbol()}) has no available hydrogens for substitution"
            }
        
        rw = Chem.RWMol(mol)

        bond_map = {
            "SINGLE": Chem.BondType.SINGLE,
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE
        }

        if bond_type not in bond_map:
            return {"success": False, "error": f"Invalid bond type: {bond_type}. Use SINGLE, DOUBLE, or TRIPLE"}

        new_idx = rw.AddAtom(Chem.Atom(new_atom))
        rw.AddBond(target_atom_index, new_idx, bond_map[bond_type])
        
        target = rw.GetAtomWithIdx(target_atom_index)
        target.SetNumExplicitHs(max(0, target.GetNumExplicitHs() - 1))

        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    
def replace_atom(
smiles: str,
atom_index: int,
new_element: str
) -> dict:
    """
    Replaces an atom with a different element.
    """
    try:
        mol = mol_from_smiles(smiles)
        
        # Validate atom index
        if atom_index < 0 or atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {atom_index}"}
        
        rw = Chem.RWMol(mol)
        old_atom = rw.GetAtomWithIdx(atom_index)
        old_element = old_atom.GetSymbol()
        
        try:
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_element)
        except:
            return {"success": False, "error": f"Invalid element symbol: {new_element}"}

        old_atom.SetAtomicNum(atomic_num)
        
        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw),
            "old_element": old_element,
            "new_element": new_element,
            "warning": "Atom replacement may create invalid valence states. Always validate result."
        }
    except Chem.AtomValenceException as e:
        return {
            "success": False, 
            "error": f"Valence error: {new_element} cannot have the same bonding pattern as the original atom. {str(e)}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_functional_group(
    smiles: str,
    target_atom_index: int,
    group: str,
    bond_type: str = "SINGLE"
) -> dict:
    """
    Adds a functional group by replacing a hydrogen.
    """
    try:
        if group not in FRAGMENTS:
            available = ", ".join(sorted(FRAGMENTS.keys()))
            return {"success": False, "error": f"Unknown functional group: {group}. Available: {available}"}

        mol = mol_from_smiles(smiles)
        
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        if target_atom.GetTotalNumHs() == 0:
            return {
                "success": False,
                "error": f"Atom {target_atom_index} ({target_atom.GetSymbol()}) has no available hydrogens for substitution"
            }

        frag = Chem.MolFromSmiles(FRAGMENTS[group])
        if frag is None:
            return {"success": False, "error": f"Invalid fragment SMILES for group: {group}"}

        dummy_atoms = [
            atom for atom in frag.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetAtomMapNum() == 1
        ]
        if len(dummy_atoms) != 1:
            return {"success": False, "error": "Fragment must contain exactly one [*:1] dummy atom"}

        dummy_atom = dummy_atoms[0]
        dummy_idx = dummy_atom.GetIdx()

        neighbors = list(dummy_atom.GetNeighbors())
        if len(neighbors) != 1:
            return {"success": False, "error": "Dummy atom must have exactly one neighbor"}

        attach_idx_frag = neighbors[0].GetIdx()

        combo = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combo)

        mol_n_atoms = mol.GetNumAtoms()
        dummy_idx_combo = mol_n_atoms + dummy_idx
        attach_idx_combo = mol_n_atoms + attach_idx_frag

        bond_map = {
            "SINGLE": Chem.BondType.SINGLE,
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE
        }

        rw.AddBond(
            target_atom_index,
            attach_idx_combo,
            bond_map[bond_type]
        )

        rw.RemoveAtom(dummy_idx_combo)

        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw),
            "group_added": group
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def add_substructure(
    smiles: str,
    target_atom_index: int,
    fragment_string: str,
    bond_type: str = "SINGLE"
) -> dict:
    """
    Attaches a custom structure to a scaffold at a specific atom index.
    
    Args:
        smiles: The SMILES string of the scaffold molecule.
        target_atom_index: The index of the atom on the scaffold to attach to.
        fragment_string: A SMILES/SMARTS string of the group to add. 
                         MUST contain exactly one attachment point labeled '[*:1]'.
                         Example: "[*:1]C1CC1" (Cyclopropyl)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "Invalid scaffold SMILES provided."}
        
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        if target_atom.GetTotalNumHs() == 0:
            return {
                "success": False,
                "error": f"Atom {target_atom_index} ({target_atom.GetSymbol()}) has no available hydrogens to replace."
            }

        frag = Chem.MolFromSmiles(fragment_string)
        if frag is None:
            frag = Chem.MolFromSmarts(fragment_string)
            if frag is None:
                return {"success": False, "error": f"Could not parse fragment structure: {fragment_string}"}

        dummy_atoms = [
            atom for atom in frag.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetAtomMapNum() == 1
        ]
        
        if len(dummy_atoms) != 1:
            return {
                "success": False, 
                "error": "Fragment must contain exactly one attachment point labeled '[*:1]'"
            }

        dummy_atom = dummy_atoms[0]
        dummy_idx = dummy_atom.GetIdx()

        neighbors = list(dummy_atom.GetNeighbors())
        if len(neighbors) != 1:
            return {"success": False, "error": "Attachment point [*:1] must have exactly one neighbor"}

        attach_idx_frag = neighbors[0].GetIdx()

        combo = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combo)

        mol_n_atoms = mol.GetNumAtoms()
        dummy_idx_combo = mol_n_atoms + dummy_idx
        attach_idx_combo = mol_n_atoms + attach_idx_frag

        bond_map = {
            "SINGLE": Chem.BondType.SINGLE,
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE
        }
        rw.AddBond(
            target_atom_index,
            attach_idx_combo,
            bond_map[bond_type]
        )
        rw.RemoveAtom(dummy_idx_combo)

        try:
            Chem.SanitizeMol(rw)
        except ValueError as e:
            return {
                "success": False, 
                "error": f"Chemical sanitization failed (likely valence error): {str(e)}"
            }

        return {
            "success": True,
            "new_smiles": Chem.MolToSmiles(rw),
            "structure_added": fragment_string
        }
        
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def replace_substructure(
    smiles: str,
    query_smarts: str,
    replacement_smiles: str,
    replace_all: bool = False
) -> dict:
    try:
        mol = mol_from_smiles(smiles)
        if mol is None:
            return {"success": False, "error": "Invalid scaffold SMILES provided."}

        # ── Validate query ────────────────────────────────────────────────
        query = Chem.MolFromSmarts(query_smarts.replace("[*:1]", ""))
        if query is None:
            return {"success": False, "error": f"Could not parse query SMARTS: {query_smarts}"}

        # ── Validate replacement ──────────────────────────────────────────
        if "[*:1]" not in replacement_smiles:
            replacement_smiles = "[*:1]"+replacement_smiles
            
        replacement = Chem.MolFromSmiles(replacement_smiles)
        if replacement is None:
            replacement = Chem.MolFromSmarts(replacement_smiles)
            if replacement is None:
                return {"success": False, "error": f"Could not parse replacement structure: {replacement_smiles}"}

        repl_dummies = [
            atom for atom in replacement.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetAtomMapNum() == 1
        ]
        if len(repl_dummies) != 1:
            return {
                "success": False,
                "error": (
                    f"replacement_smiles must contain exactly one '[*:1]' marking the "
                    f"attachment point (found {len(repl_dummies)}). "
                    f"Example: '[*:1]C1CC1' for cyclopropyl."
                ),
            }

        if len(list(repl_dummies[0].GetNeighbors())) != 1:
            return {
                "success": False,
                "error": "Attachment point '[*:1]' in replacement_smiles must have exactly one neighbor.",
            }

        # ── Match & replace ───────────────────────────────────────────────
        matches = mol.GetSubstructMatches(query)
        if not matches:
            return {"success": False, "error": "No matching substructure found in the molecule."}

        result_mol = mol
        replacements_done = 0
        match_set = set(matches[0])
        
        # Find all scaffold atoms neighbouring the match (the cut bonds)
        scaffold_attach_indices = []
        for atom_idx in match_set:
            for neighbor in result_mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                if neighbor.GetIdx() not in match_set:
                    scaffold_attach_indices.append(neighbor.GetIdx())
        if len(scaffold_attach_indices) != 1:
            return {
                "success": False,
                "error": (
                    f"Matched substructure has {len(scaffold_attach_indices)} bond(s) "
                    "to the rest of the scaffold; expected exactly 1. "
                    "Refine your query_smarts to select a terminal substructure."
                ),
            }

        scaffold_attach_idx = scaffold_attach_indices[0]


        repl_dummy = repl_dummies[0]
        repl_dummy_idx = repl_dummy.GetIdx()
        repl_attach_idx = list(repl_dummy.GetNeighbors())[0].GetIdx()

        n_scaffold_atoms = result_mol.GetNumAtoms()
        rw = Chem.RWMol(Chem.CombineMols(result_mol, replacement))

        repl_dummy_combo = n_scaffold_atoms + repl_dummy_idx
        repl_attach_combo = n_scaffold_atoms + repl_attach_idx

        rw.AddBond(scaffold_attach_idx, repl_attach_combo, Chem.BondType.SINGLE)

        # Remove matched atoms + replacement dummy in reverse index order
        # to keep earlier indices valid during deletion
        for idx in sorted(match_set | {repl_dummy_combo}, reverse=True):
            rw.RemoveAtom(idx)

        result_mol = rw.GetMol()
        replacements_done += 1

        # ── Connectivity check ────────────────────────────────────────────
        frags = Chem.GetMolFrags(result_mol, asMols=True)
        if len(frags) > 1:
            return {
                "success": False,
                "error": (
                    "Replacement produced disconnected fragments. "
                    "Check that '[*:1]' in query_smarts is on the correct attachment bond."
                ),
            }

        # ── Sanitize ──────────────────────────────────────────────────────
        try:
            Chem.SanitizeMol(result_mol)
        except ValueError as e:
            return {
                "success": False,
                "error": f"Chemical sanitization failed (likely valence error): {str(e)}",
            }

        return {
            "success": True,
            "new_smiles": canonical_smiles(result_mol),
            "num_matches": len(matches),
            "replaced_all": replace_all,
            "warning": "Only first match replaced" if not replace_all and len(matches) > 1 else None,
        }

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def remove_substructure(
    smiles: str,
    substructure_smarts: str
) -> dict:
    """
    Removes all instances of a substructure match from the molecule.
    Fails if the removal causes the molecule to break into disconnected fragments.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        query = Chem.MolFromSmarts(substructure_smarts)

        if not mol:
            return {"success": False, "error": "Invalid base SMILES"}
        if not query:
            return {"success": False, "error": "Invalid SMARTS pattern"}

        matches = mol.GetSubstructMatches(query)
        
        if not matches:
            return {"success": False, "error": "Substructure not found in molecule"}

        indices_to_remove = set()
        for match in matches:
            indices_to_remove.update(match)
            
        rw = Chem.RWMol(mol)
        
        sorted_indices = sorted(list(indices_to_remove), reverse=True)
        
        for idx in sorted_indices:
            rw.RemoveAtom(idx)

        frags = Chem.GetMolFrags(rw, asMols=True)
        
        if len(frags) > 1:
            return {
                "success": False, 
                "error": "Operation failed: Removal resulted in disconnected fragments (e.g., broken linker)."
            }
            
        final_mol = rw

        Chem.SanitizeMol(final_mol)
        return {
            "success": True,
            "new_smiles": Chem.MolToSmiles(final_mol, isomericSmiles=True),
            "removed_count": len(sorted_indices)
        }

    except Exception as e:
        return {"success": False, "error": f"Removal failed: {str(e)}"}
    

def _fragment_molecule(smiles: str, cut_atom_idx: int) -> dict:
    """
    Splits a molecule at the first non-ring bond from the specified atom.
    Returns head (containing cut_atom_idx) and tail as RDKit mols,
    each with exactly one [*:1] marking the cut point.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {"success": False, "error": f"Invalid SMILES: {smiles}"}

    n_atoms = mol.GetNumAtoms()
    if cut_atom_idx < 0 or cut_atom_idx >= n_atoms:
        return {
            "success": False,
            "error": f"Invalid atom index {cut_atom_idx} for molecule with {n_atoms} atoms.",
        }

    atom = mol.GetAtomWithIdx(cut_atom_idx)

    # Find the first non-ring bond from this atom — ring bonds can't produce 2 clean fragments
    cut_bond_idx = None
    for neighbor in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(cut_atom_idx, neighbor.GetIdx())
        if not bond.IsInRing():
            cut_bond_idx = bond.GetIdx()
            break

    if cut_bond_idx is None:
        return {
            "success": False,
            "error": (
                f"Atom {cut_atom_idx} ({atom.GetSymbol()}) has no non-ring bonds. "
                "Choose an atom with at least one acyclic bond to the scaffold."
            ),
        }

    frag_mol = Chem.FragmentOnBonds(mol, [cut_bond_idx], addDummies=True)
    frag_idx_sets = Chem.GetMolFrags(frag_mol)           # atom indices in frag_mol coords
    frag_mols    = Chem.GetMolFrags(frag_mol, asMols=True)

    if len(frag_mols) != 2:
        return {
            "success": False,
            "error": (
                f"Expected 2 fragments after cutting atom {cut_atom_idx}, "
                f"got {len(frag_mols)}. The atom may be fully embedded in a ring system."
            ),
        }

    def _normalize_dummy(mol_raw: Chem.Mol) -> Chem.Mol:
        """Set the dummy atom's map number to 1 and clear any isotope label."""
        rw = Chem.RWMol(mol_raw)
        for a in rw.GetAtoms():
            if a.GetAtomicNum() == 0:   # wildcard / dummy
                a.SetAtomMapNum(1)
                a.SetIsotope(0)
        return rw.GetMol()

    head_mol, tail_mol = None, None
    for idx_set, frag in zip(frag_idx_sets, frag_mols):
        if cut_atom_idx in idx_set:
            head_mol = _normalize_dummy(frag)
        else:
            tail_mol = _normalize_dummy(frag)

    if head_mol is None or tail_mol is None:
        return {"success": False, "error": "Internal error: could not identify head/tail fragments."}

    return {"success": True, "head": head_mol, "tail": tail_mol}


def _join_fragments(frag_a: Chem.Mol, frag_b: Chem.Mol) -> Chem.Mol | None:
    """
    Joins two fragments at their [*:1] attachment points.
    Uses the same CombineMols → AddBond → RemoveAtom pattern as add_substructure.
    Returns the sanitized product mol, or None if anything fails.
    """
    def _get_attachment(mol: Chem.Mol) -> tuple[int, int] | tuple[None, None]:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == 1:
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) == 1:
                    return atom.GetIdx(), neighbors[0].GetIdx()
        return None, None

    dummy_a, attach_a = _get_attachment(frag_a)
    dummy_b, attach_b = _get_attachment(frag_b)

    if dummy_a is None or dummy_b is None:
        return None

    n_a = frag_a.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(frag_a, frag_b))

    attach_b_combo = n_a + attach_b
    dummy_b_combo  = n_a + dummy_b

    rw.AddBond(attach_a, attach_b_combo, Chem.BondType.SINGLE)

    # Remove both dummies highest-index first to keep earlier indices stable
    for idx in sorted([dummy_a, dummy_b_combo], reverse=True):
        rw.RemoveAtom(idx)

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def crossover_molecules(
    smiles_a: str,
    cut_idx_a: int,
    smiles_b: str,
    cut_idx_b: int,
) -> dict:
    """
    Splits two molecules at specified atoms and recombines their fragments.
    Tries all 4 combinations (head+head, head+tail, tail+head, tail+tail)
    and returns the first chemically valid result.

    Each molecule is cleaved at the first non-ring bond from the specified atom.
    The fragment that retains the cut atom is the 'head'; the other is the 'tail'.

    Args:
        smiles_a:   SMILES of the first molecule.
        cut_idx_a:  Atom index in molecule A at which to cut.
        smiles_b:   SMILES of the second molecule.
        cut_idx_b:  Atom index in molecule B at which to cut.
    """
    # ── Fragment both molecules ───────────────────────────────────────────
    split_a = _fragment_molecule(smiles_a, cut_idx_a)
    if not split_a["success"]:
        return {"success": False, "error": f"Could not fragment molecule A: {split_a['error']}"}

    split_b = _fragment_molecule(smiles_b, cut_idx_b)
    if not split_b["success"]:
        return {"success": False, "error": f"Could not fragment molecule B: {split_b['error']}"}

    head_a, tail_a = split_a["head"], split_a["tail"]
    head_b, tail_b = split_b["head"], split_b["tail"]

    # ── Try all 4 recombinations ──────────────────────────────────────────
    combinations = [
        ("head_a + head_b", head_a, head_b),
        ("head_a + tail_b", head_a, tail_b),
        ("tail_a + head_b", tail_a, head_b),
        ("tail_a + tail_b", tail_a, tail_b),
    ]

    all_results = []
    first_valid = None
    all_valid = []
    for label, frag1, frag2 in combinations:
        product = _join_fragments(frag1, frag2)
        if product is not None:
            smi = canonical_smiles(product)
            all_results.append({"combination": label, "smiles": smi, "valid": True})
            all_valid.append({"combination": label, "smiles": smi, "valid": True})
            if first_valid is None:
                first_valid = {"combination": label, "smiles": smi}
        else:
            all_results.append({"combination": label, "smiles": None, "valid": False})

    if first_valid is None:
        return {
            "success": False,
            "error": "All 4 fragment combinations failed sanitization.",
            "all_results": all_results,
        }
    choice = np.random.choice(all_valid)
    return {
        "success": True,
        "new_smiles": choice["smiles"],
        "combination_used": choice["combination"],
    }


def calculate_properties(smiles: str) -> dict:
    """Calculates common molecular descriptors."""
    try:
        mol = mol_from_smiles(smiles)
        return {
            "qed": round(QED.qed(mol), 2),
            "sa": round(sascorer.calculateScore(mol), 2),
            "mw": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 2),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------
# Tool registry
# ------------------------------------------------------------
TOOLS = {
    "get_attachment_points": get_attachment_points,
    "calculate_properties": calculate_properties,
    "add_atom": add_atom,
    "replace_atom": replace_atom,
    "add_functional_group": add_functional_group,
    "add_substructure": add_substructure,
    "replace_substructure": replace_substructure,
    "remove_substructure": remove_substructure,
    "crossover_molecules": crossover_molecules
}


# ------------------------------------------------------------
# Tool schemas (OpenAI Responses API)
# ------------------------------------------------------------
TOOL_SCHEMAS = [
#        {
#     "type": "function",
#     "name": "crossover_molecules",
#     "description": "Splits two molecules at the specified indices and recombines the resulting fragments. Try to split each molecule roughly in half.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "smiles_a": {
#                 "type": "string",
#                 "description": "SMILES string of the first molecule."
#             },
#             "cut_idx_a": {
#                 "type": "integer",
#                 "description": "Index of the atom in Molecule 1 where the split should occur."
#             },
#             "smiles_b": {
#                 "type": "string",
#                 "description": "SMILES string of the second molecule."
#             },
#             "cut_idx_b": {
#                 "type": "integer",
#                 "description": "Index of the atom in Molecule 2 where the split should occur."
#             },
#         },
#         "required": [
#             "smiles_a",
#             "cut_idx_a",
#             "smiles_b",
#             "cut_idx_b"
#         ]
#     }
# },
    # {
    #     "type": "function",
    #     "name": "get_attachment_points",
    #     "description": "Returns list of all atoms in the input smiles that have substitutable hydrogens for replacement",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "smiles": {"type": "string"},
    #         },
    #         "required": ["smiles"],
    #     },
    # },
        {
        "type": "function",
        "name": "calculate_properties",
        "description": "Returns chemical properties for the input SMILES (QED, SA, Molecular Weight, etc)",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
            },
            "required": ["smiles"],
        },
    },
    {
        "type": "function",
        "name": "add_atom",
        "description": "Attach new atom to the specified index",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "target_atom_index": {"type": "integer"},
                "new_atom": {"type": "string"},
                "bond_type": {"type": "string", "enum": ["SINGLE", "DOUBLE", "TRIPLE"]},
            },
            "required": ["smiles", "target_atom_index", "new_atom"],
        },
    },
    {
        "type": "function",
        "name": "replace_atom",
        "description": "Replace atom at specified index",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "atom_index": {"type": "integer"},
                "new_element": {"type": "string"},
            },
            "required": ["smiles", "atom_index", "new_element"],
        },
    },
    {
        "type": "function",
        "name": "add_functional_group",
        "description": "Attach specified functional group (from provided enum) at the specified index. For names in the enum with a suffix, _N means to attach that group via nitrogen, while _# for any number indicates the attachment position on the ring.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "target_atom_index": {"type": "integer"},
                "group": {
                    "type": "string",
                    "enum": [
                        "methyl","ethyl","propyl","isopropyl","tert_butyl",
                        "cyclopropyl","cyclobutyl","cyclopentyl","cyclohexyl",
                        "fluoro","chloro","bromo","iodo",
                        "hydroxyl","methoxy","ethoxy",
                        "amine","methylamine","dimethylamine",
                        "thiol","methylthio",
                        "aldehyde","ketone_methyl","carboxylic_acid",
                        "ester_methyl","amide","amide_methyl","urea","carbamate",
                        "hydroxymethyl","aminoethyl","dimethylaminoethyl",
                        "morpholine","piperazine","piperidine",
                        "phenyl", "benzyl", "phenoxy", "benzoyl",
                        "pyridyl_4", "pyridyl_3", "pyridyl_2", "pyrimidinyl",
                        "thienyl", "furanyl", "naphthyl_1", "naphthyl_2",
                        "indole_3", "indole_N", "benzimidazole", "quinoline",
                        "isoquinoline", "benzodioxole", "benzofuran",
                        "adamantyl", "norbornyl", "biphenyl"
                    ],
                },
                "bond_type": {"type": "string", "enum": ["SINGLE", "DOUBLE", "TRIPLE"]},
            },
            "required": ["smiles", "target_atom_index", "group", "bond_type"],
        },
    },
    {
        "type": "function",
        "name": "add_substructure",
        "description": "Attaches a custom SMILES group or structure to a specific atom index.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                },
                "target_atom_index": {
                    "type": "integer",
                },
                "fragment_string": {
                    "type": "string",
                    "description": "MUST contain exactly one attachment point labeled '[*:1]'. Example: '[*:1]C1CC1' for cyclopropyl."
                },
                "bond_type": {"type": "string", "enum": ["SINGLE", "DOUBLE", "TRIPLE"]},

            },
            "required": ["smiles", "target_atom_index", "fragment_string"]
        }
    },
    {
        "type": "function",
        "name": "replace_substructure",
        "description": "Replace terminal SMARTS-specified substructure with custom SMILES; only replaces first SMARTS match",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "query_smarts": {
                    "type": "string",
                    "description": "IMPORTANT: Do NOT include any labeled attachment points in this SMARTS argument."
                },
                "replacement_smiles": {
                    "type": "string",
                    "description": "MUST contain exactly one labeled attachment point '[*:1]'"
                },
            },
            "required": ["smiles", "query_smarts", "replacement_smiles"],
        },
    },
    {
        "type": "function",
        "name": "remove_substructure",
        "description": "Delete SMARTS-specified substructure",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "substructure_smarts": {"type": "string"},
            },
            "required": ["smiles", "substructure_smarts"],
        },
    },
]


# ------------------------------------------------------------
# Agent state
# ------------------------------------------------------------
@dataclass
class AgentState:
    current_smiles: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None


# ------------------------------------------------------------
# Tool executor
# ------------------------------------------------------------
def execute_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"success": False, "error": "Unknown tool"}
    try:
        return TOOLS[name](**args)
    except Exception as e:
        print(e)
        return {"success": False, "error": e}


# ------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------
def run_agent(
    current_smiles: str,
    curr_messages: str,
    max_steps: int = 20,
    index: int = 0
) -> AgentState:
    state = AgentState(
        current_smiles=current_smiles
    )

    # system_context = (
    #     "You are a molecular design agent.\n"
    #     "You may ONLY modify molecules using tools.\n"
    #     "Only make one modification at a time.\n"
    #     "Read the parameter descriptions for the tools very carefully.\n"
    #     "Always ensure that your modifications don't break valence rules and does not result in a fragmented molecule."
    # )
    
    # no crossovers (1 molecule)
    # user_goal = (
    #     f"Goal: {prompt}\n"
    #     f"Initial SMILES: {initial_smiles}"
    # )
    
    # messages = [
    #     {"role": "system", "content": system_context},
    #     {"role": "user", "content": user_goal},
    #     {"role": "user", "content": f"Possible attachment points: {str(get_attachment_points(initial_smiles))}"},
    #     {"role": "user", "content": f"Molecule properties: {str(calculate_properties(initial_smiles))}"}
    # ]
    
    messages = curr_messages
    
    # crossovers (2 molecules)
    # user_goal = (
    #     f"Goal: {prompt}\n"
    #     f"Initial SMILES:\n1. {initial_smiles[0]}\n2. {initial_smiles[1]}"
    # )    

    # messages.append(
    #     {"role": "user", "content": f"Possible attachment points for ligand 1: {str(get_attachment_points(initial_smiles))}\nPossible attachment points for ligand 2: {str(get_attachment_points(initial_smiles))}"}
    # )
    modification_tools = ["add_atom", "replace_atom", "add_functional_group", "add_substructure", "remove_substructure", "replace_substructure", "crossover_molecules"]

    should_break = False
    for step in range(max_steps):
        print(f"\n[Index {str(index)}] CURRENT MESSAGES: {str(messages)}\n", flush=True)
        response = client.responses.create(
            model='gpt-oss-120b',
            input=messages,
            tools=TOOL_SCHEMAS
        )
        
        for msg in response.output:
            if msg.type == "reasoning":
                print(f"[Index {str(index)}] [Reasoning]: {msg.content}", flush=True) 
            elif msg.type == "function_call":
                print(f"[Index {str(index)}] [Tool Call]: {msg}")
                
                tool_name = msg.name
                args = json.loads(msg.arguments)

                # if "smiles" in args:
                #     args["smiles"] = state.current_smiles

                result = execute_tool(tool_name, args)
                
                print(f"[Index {str(index)}] [Tool Result]: {result}\n")
                
                if "new_smiles" in result:
                    state.current_smiles = result["new_smiles"]
                
                state.history.append({
                    "step": step,
                    "tool": tool_name,
                    "arguments": args,
                    "result": result
                })
                
                messages.append(msg)

                messages.append({
                    "type": "function_call_output",
                    "call_id": msg.call_id,
                    "output": json.dumps(result)
                })
                if state.current_smiles != current_smiles:
                    if tool_name in modification_tools:
                        for i, item in enumerate(messages):
                            if isinstance(item, dict) and "content" in item and "Possible attachment points" in item["content"]:
                                messages.pop(i)
                                break
                        
                        messages.append({
                            "role": "user",
                            "content": (
                                "Output FINAL_ANSWER if you have made sufficient modifications (at most 3). Ensure that desired properties are maintained.\n"
                                f"Current SMILES: {state.current_smiles}\n"
                                f"Possible attachment points: {str(get_attachment_points(state.current_smiles))}"
                            )
                        })              
            else:
                content = msg.content[0].text
                messages.append({"role": "assistant", "content": content})
                print(f"[Index {str(index)}] [Message]: {content}")
                state.final_answer = content
                should_break = True
                break
        if should_break:
            break
        
    return (state, copy.deepcopy(messages))

def query_LLM(messages, index):
    print(f"[Index {str(index)}] [Summary Input]: {str(messages)}", flush=True)

    response = client.responses.create(
        model='gpt-oss-120b',
        input=messages,
    )

    for msg in response.output:
        if msg.type == "reasoning":
            continue
        else:
            content = msg.content[0].text
            print(f"[Index {str(index)}] [Summary]: {content}", flush=True)
            return content

def calculate_boltz_nautilus(protein_name, ligand_smiles, idx=0):
    print(f"\n[Worker {str(idx)}] Sending job for {ligand_smiles}...", flush=True)
    
    query_params = {
        "protein_name": protein_name,
        "ligand": ligand_smiles
    }
    worker_lifetime = time.time()
    num_errors = 0
    while True:
        try:
            before = time.time()
            response = requests.post(API_URL, params=query_params, headers={'Connection': 'close'})
            after = time.time()
            print(response)
            if response.status_code == 200:
                result = response.json()
                print(f"\n[Worker {str(idx)}] Success")
                print(result)
                print(f"[Worker {str(idx)}]Took: {str(after-before)} seconds", flush=True)
                time.sleep(3)
                return result["affinity"]
            elif response.status_code == 429 or response.status_code == 503:
                sleep_time = random.uniform(0.5, 2.0)
                print(f"[Worker {str(idx)}] Busy: Retrying", flush=True)
                current_lifetime = time.time()
                if worker_lifetime - current_lifetime > 3600:
                    print(f"[Worker {str(idx)}] Worker has been stalled for {str(worker_lifetime-current_lifetime)} seconds. Exiting.", flush=True)
                    return 0
                time.sleep(sleep_time)
            else:
                print(f"[Worker {str(idx)}] Error {response.status_code}: {response.text}", flush=True)
                num_errors += 1
                if num_errors >= 100:
                    return 0
                time.sleep(10)
        except Exception as e:
            print(f"\n[Worker {str(idx)}] Connection failed: {str(e)}", flush=True)        
            time.sleep(3)
            return 0

def calculate_fitness(smiles, affinity):
    mol = Chem.MolFromSmiles(smiles)
    qed = QED.qed(mol)
    sa = 1 - ((sascorer.calculateScore(mol) - 1) / 9)
    affin = -(affinity / 13)
    
    qed = qed * 1
    sa = sa * 1
    affin = affin * 5
    
    return affin + qed + sa

def run_trajectory(protein, initial_smiles, initial_affinity, idx=0, num_steps=50):
    system_context = (
        "You are a molecular design agent.\n"
        "You may ONLY modify molecules using tools.\n"
        "Only make one modification at a time.\n"
        "Read the parameter descriptions for the tools very carefully.\n"
        "Always ensure that your modifications don't break valence rules and does not result in a fragmented molecule."
    )        
    mols = {}
    best_ligand = initial_smiles
    best_fitness = calculate_fitness(initial_smiles, initial_affinity)
    best_affinity = initial_affinity
    summary = ""
    for step in range(num_steps): 
        print(f"[Index {str(idx)}]: [===== STEP {str(step+1)} =====] {best_ligand}: {str(best_fitness)}")
        current_mol = Chem.MolFromSmiles(best_ligand)
        if summary:
            goal = (
                f"We will collaborate on modifying a ligand to maximize binding affinity, maximize QED, and minimize SA to the protein {protein}. I will give you the output from docking software after each of your attempts\n."
                f"Here is our current accumulated knowledge about the target:\n{summary}\n\n"
                f"First, understand our current accumulated knowledge about the target. Then modify the current ligand (provided below). Assume that there are always more improvements to be made to the current ligand. Do not let molecular weight exceed 700.\n"
                f"Current ligand: {best_ligand}\n"
                f"Binding affinity (kcal/mol): {str(best_affinity)}\n"
                f"QED: {str(round(QED.qed(current_mol), 2))}\n"
                f"SA: {str(round(sascorer.calculateScore(current_mol), 2))}\n"
        )
        else:
            goal = (
                f"We will collaborate on modifying a ligand to maximize binding affinity, maximize QED, and minimize SA to the protein {protein}. I will give you the output from docking software after each of your attempts. Assume that there are always more improvements to be made to the current ligand. Do not let molecular weight exceed 700.\n"
                f"Current ligand: {best_ligand}\n"
                f"Binding affinity (kcal/mol): {str(best_affinity)}\n"
                f"QED: {str(round(QED.qed(current_mol), 2))}\n"
                f"SA: {str(round(sascorer.calculateScore(current_mol), 2))}\n"
            )
        conversation_messages = [{"role": "system", "content": system_context}, {"role": "user", "content": goal}]
        conversation_messages.append({"role": "user", "content":f"Possible attachment points for ligand: {str(get_attachment_points(best_ligand))}"})
        
        # print(f"[Index {str(idx)}]: [===== STEP {str(step+1)} =====]\n{conversation_messages}")
        message_input = copy.deepcopy(conversation_messages)
        
        for i in range(10):
            try:
                # state = run_agent(parent_smiles[0], task_objective, max_steps=20, index=idx)
                (state, message_output) = run_agent(best_ligand, message_input, index=idx)
                current_smiles = state.current_smiles
                
                affinity = calculate_boltz_nautilus(protein, current_smiles, idx)

                mols[current_smiles] = [idx, step, affinity]
                
                current_fitness = calculate_fitness(current_smiles, affinity)
                if current_fitness > best_fitness:
                    best_ligand = current_smiles
                    best_fitness = current_fitness
                    best_affinity - affinity
               
                current_mol = Chem.MolFromSmiles(current_smiles)
                summary_prompt = f"The new ligand {current_smiles} has a binding affinity of {str(affinity)}. It has a QED of {str(QED.qed(current_mol))} and a SA of {str(sascorer.calculateScore(current_mol))}.\n"
                summary_prompt += "Based on the results of this past trial, update our current accumulated knowledge of the important details and information about the binding target. Describe what has been effective and what has not. Keep your summary brief."
                message_output.append({"role": "user", "content": summary_prompt})
                summary = query_LLM(message_output, idx)
                break
            except Exception as e:
                print(e)
                continue
    print(f"Trajectory {str(idx)} produced: {str(mols)}")
    return mols

if __name__ == "__main__":
    with open(f"/home/ubuntu/MOLLEO/init_caches/c-met_0.yaml", 'r') as file:
        initial_pool = yaml.safe_load(file)
    with ThreadPoolExecutor(max_workers=20) as pool:
        inputs = [("c-met", list(initial_pool.keys())[idx], initial_pool[list(initial_pool.keys())[idx]], idx, 50) for idx in range(20)]
        output_smiles = list(pool.map(lambda x: run_trajectory(*x), inputs))
    print(f"Output molecule pool: {str(output_smiles)}")
    result = {k: v for d in output_smiles for k, v in d.items()}
    with open(f"/home/ubuntu/MOLLEO/conversation_framework/results/weighted_objective.yaml", 'w') as f:
            yaml.dump(result, f, sort_keys=False)
    
    # run_trajectory("c-met", "C[NH+](Cc1cccs1)C[C@H](O)COc1c(Cl)cccc1Cl", -9.02)
    # state = run_agent(
    #     initial_smiles=["Cc1ccc(-c2cc(N)c(=O)n([C@H](C)C(=O)NC3CCCCC3)n2)o1", "COc1cc(F)c(NC(C)=O)c(NC(=O)c2cc(F)ccc2O)c1"],
    #     prompt="I have given you two candidate ligands. Please propose a new molecule that binds better to c-MET. You are encouraged to make a crossover between the molecules on the first step, then mutate the resulting molecule. Only make a few modifications (at most 3), then respond with FINAL_ANSWER.\n"
    # )

    # print("Final SMILES:", state.current_smiles)
    # print("Final Answer:", state.final_answer)
    # print("Steps taken:", len(state.history))
    