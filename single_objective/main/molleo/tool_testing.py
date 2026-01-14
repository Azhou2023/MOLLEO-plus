import json
from dataclasses import dataclass, field
import sys
from typing import Dict, Any, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

from openai import OpenAI
from dotenv import load_dotenv
import os

from rdkit.Chem import QED
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

load_dotenv()

client = OpenAI(api_key=os.getenv("GPT_KEY"))

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    Chem.SanitizeMol(mol)
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


#Tools start here
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
        return {"attachment_points": pts, "success": True}
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
        
        # Validate target atom index
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        # Check if target atom has available hydrogens
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

        # Add new atom and bond
        new_idx = rw.AddAtom(Chem.Atom(new_atom))
        rw.AddBond(target_atom_index, new_idx, bond_map[bond_type])
        
        # Remove one implicit hydrogen from target
        target = rw.GetAtomWithIdx(target_atom_index)
        target.SetNumExplicitHs(max(0, target.GetNumExplicitHs() - 1))

        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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

FRAGMENTS = {
    **ALKYL_FRAGMENTS,
    **HALOGEN_FRAGMENTS,
    **HETEROATOM_FRAGMENTS,
    **CARBONYL_FRAGMENTS,
    **POLAR_FRAGMENTS
}


def add_functional_group(
    smiles: str,
    target_atom_index: int,
    group: str
) -> dict:
    """
    Adds a functional group by replacing a hydrogen.
    """
    try:
        if group not in FRAGMENTS:
            available = ", ".join(sorted(FRAGMENTS.keys()))
            return {"success": False, "error": f"Unknown functional group: {group}. Available: {available}"}

        mol = mol_from_smiles(smiles)
        
        # Validate target atom index
        if target_atom_index < 0 or target_atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {target_atom_index}"}
        
        target_atom = mol.GetAtomWithIdx(target_atom_index)
        
        # Check if target has available hydrogen
        if target_atom.GetTotalNumHs() == 0:
            return {
                "success": False,
                "error": f"Atom {target_atom_index} ({target_atom.GetSymbol()}) has no available hydrogens for substitution"
            }

        frag = Chem.MolFromSmiles(FRAGMENTS[group])
        if frag is None:
            return {"success": False, "error": f"Invalid fragment SMILES for group: {group}"}

        # Find dummy atom in fragment
        dummy_atoms = [
            atom for atom in frag.GetAtoms()
            if atom.GetSymbol() == "*" and atom.GetAtomMapNum() == 1
        ]
        if len(dummy_atoms) != 1:
            return {"success": False, "error": "Fragment must contain exactly one [*:1] dummy atom"}

        dummy_atom = dummy_atoms[0]
        dummy_idx = dummy_atom.GetIdx()

        # The real attachment atom is the neighbor of the dummy
        neighbors = list(dummy_atom.GetNeighbors())
        if len(neighbors) != 1:
            return {"success": False, "error": "Dummy atom must have exactly one neighbor"}

        attach_idx_frag = neighbors[0].GetIdx()

        # Combine parent + fragment
        combo = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combo)

        mol_n_atoms = mol.GetNumAtoms()
        dummy_idx_combo = mol_n_atoms + dummy_idx
        attach_idx_combo = mol_n_atoms + attach_idx_frag

        # Create bond between parent and fragment
        rw.AddBond(
            target_atom_index,
            attach_idx_combo,
            Chem.BondType.SINGLE
        )

        # Remove dummy atom AFTER bonding
        rw.RemoveAtom(dummy_idx_combo)

        Chem.SanitizeMol(rw)
        return {
            "success": True,
            "new_smiles": canonical_smiles(rw),
            "group_added": group
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
    WARNING: This can still create invalid molecules if valence doesn't match.
    """
    try:
        mol = mol_from_smiles(smiles)
        
        # Validate atom index
        if atom_index < 0 or atom_index >= mol.GetNumAtoms():
            return {"success": False, "error": f"Invalid atom index: {atom_index}"}
        
        rw = Chem.RWMol(mol)
        old_atom = rw.GetAtomWithIdx(atom_index)
        old_element = old_atom.GetSymbol()
        
        # Get atomic number for new element
        try:
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_element)
        except:
            return {"success": False, "error": f"Invalid element symbol: {new_element}"}

        # Set new element
        old_atom.SetAtomicNum(atomic_num)
        
        # Try to sanitize - this will fail if valence is wrong
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


def replace_substructure(
    smiles: str,
    query_smarts: str,
    replacement_smiles: str,
    replace_all: bool = False
) -> dict:
    """
    Replaces a substructure pattern with a new fragment.
    FIX: Added replace_all parameter and better match reporting.
    """
    try:
        mol = mol_from_smiles(smiles)
        query = Chem.MolFromSmarts(query_smarts)
        replacement = mol_from_smiles(replacement_smiles)

        if query is None:
            return {"success": False, "error": "Invalid SMARTS pattern"}

        # Find matches first
        matches = mol.GetSubstructMatches(query)
        if not matches:
            return {"success": False, "error": "No matching substructure found"}

        replaced = Chem.ReplaceSubstructs(
            mol,
            query,
            replacement,
            replaceAll=replace_all
        )

        if not replaced:
            return {"success": False, "error": "Replacement failed"}

        # Check for disconnected fragments
        frags = Chem.GetMolFrags(replaced[0], asMols=True)
        if len(frags) > 1:
            return {
                "success": False, 
                "error": "Replacement resulted in disconnected fragments. The query may not include proper attachment points."
            }
        
        Chem.SanitizeMol(replaced[0])
        return {
            "success": True,
            "new_smiles": canonical_smiles(replaced[0]),
            "num_matches": len(matches),
            "replaced_all": replace_all,
            "warning": "Only first match replaced" if not replace_all and len(matches) > 1 else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_properties(smiles: str) -> dict:
    """Calculates common molecular descriptors."""
    try:
        mol = mol_from_smiles(smiles)
        return {
            "success": True,
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
    
def remove_substructure(
    smiles: str,
    substructure_smarts: str
) -> dict:
    """
    Removes all instances of a substructure match from the molecule.
    Fails if the removal causes the molecule to break into disconnected fragments.
    """
    mol = Chem.MolFromSmiles(smiles)
    query = Chem.MolFromSmarts(substructure_smarts)

    if not mol:
        return {"success": False, "error": "Invalid base SMILES"}
    if not query:
        return {"success": False, "error": "Invalid SMARTS pattern"}

    # 1. Identify atoms to remove
    matches = mol.GetSubstructMatches(query)
    
    if not matches:
        return {"success": False, "error": "Substructure not found in molecule"}

    indices_to_remove = set()
    for match in matches:
        indices_to_remove.update(match)

    # 2. Perform Removal
    try:
        rw = Chem.RWMol(mol)
        
        # Sort descending to prevent index shifting errors during deletion
        sorted_indices = sorted(list(indices_to_remove), reverse=True)
        
        for idx in sorted_indices:
            rw.RemoveAtom(idx)

        # 3. Validation: Check for Fragmentation
        # GetMolFrags returns a tuple of sub-molecules
        frags = Chem.GetMolFrags(rw, asMols=True)
        
        if len(frags) > 1:
            return {
                "success": False, 
                "error": "Operation failed: Removal resulted in disconnected fragments (e.g., broken linker)."
            }
            
        final_mol = rw

        # 4. Sanitize and Return
        Chem.SanitizeMol(final_mol)
        return {
            "success": True,
            "new_smiles": Chem.MolToSmiles(final_mol, isomericSmiles=True),
            "removed_count": len(sorted_indices)
        }

    except Exception as e:
        return {"success": False, "error": f"Removal failed: {str(e)}"}


# ------------------------------------------------------------
# Tool registry
# ------------------------------------------------------------
TOOLS = {
    "validate_smiles": validate_smiles,
    "get_attachment_points": get_attachment_points,
    "add_atom": add_atom,
    "add_functional_group": add_functional_group,
    "replace_atom": replace_atom,
    "replace_substructure": replace_substructure,
    "remove_substructure": remove_substructure,
    "calculate_properties": calculate_properties,
}


# ------------------------------------------------------------
# Tool schemas (OpenAI Responses API)
# ------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "validate_smiles",
        "description": "Validate a SMILES string",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"}
            },
            "required": ["smiles"]
        }
    },
    {
        "type": "function",
        "name": "get_attachment_points",
        "description": "Find atoms that can accept substituents",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"}
            },
            "required": ["smiles"]
        }
    },
    {
        "type": "function",
        "name": "add_atom",
        "description": "Add a single atom to a target atom",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "target_atom_index": {"type": "integer"},
                "new_atom": {"type": "string"},
                "bond_type": {
                    "type": "string",
                    "enum": ["SINGLE", "DOUBLE", "TRIPLE"]
                }
            },
            "required": ["smiles", "target_atom_index", "new_atom"]
        }
    },
    {
        "type": "function",
        "name": "add_functional_group",
        "description": "Attach a functional group to a molecule",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "target_atom_index": {"type": "integer"},
                "group": {
                    "type": "string",
                    "enum": [
                         "ethyl",
                        "propyl",
                        "isopropyl",
                        "tert_butyl",
                        "cyclopropyl",
                        "cyclobutyl",
                        "cyclopentyl",
                        "cyclohexyl",
                        "fluoro",
                        "chloro",
                        "bromo",
                        "iodo",
                        "hydroxyl",
                        "methoxy",
                        "ethoxy",
                        "amine",
                        "methylamine",
                        "dimethylamine",
                        "thiol",
                        "methylthio",
                        "aldehyde",
                        "ketone_methyl",
                        "carboxylic_acid",
                        "ester_methyl",
                        "amide",
                        "amide_methyl",
                        "urea",
                        "carbamate",
                        "hydroxymethyl",
                        "aminoethyl",
                        "dimethylaminoethyl",
                        "morpholine",
                        "piperazine",
                        "piperidine"
                    ]
                }
            },
            "required": ["smiles", "target_atom_index", "group"]
        }
    },
    {
        "type": "function",
        "name": "replace_atom",
        "description": "Replace an atom with another element",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "atom_index": {"type": "integer"},
                "new_element": {"type": "string"}
            },
            "required": ["smiles", "atom_index", "new_element"]
        }
    },
    {
        "type": "function",
        "name": "replace_substructure",
        "description": "Replace a substructure using SMARTS. Fails if replacement results in fragments.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "query_smarts": {"type": "string"},
                "replacement_smiles": {"type": "string"}
            },
            "required": ["smiles", "query_smarts", "replacement_smiles"]
        }
    },
    {
        "type": "function",
        "name": "calculate_properties",
        "description": "Compute RDKit physicochemical properties. Includes molecular weight, QED, SA, etc",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"}
            },
            "required": ["smiles"]
        }
    },
    {
        "type": "function",
        "name": "remove_substructure",
        "description": "Removes atoms matching a specific SMARTS pattern. Fails if removal results in fragments.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "The SMILES string of the molecule to modify."
                },
                "substructure_smarts": {
                    "type": "string",
                    "description": "The SMARTS pattern representing the substructure to remove (e.g., '[N+](=O)[O-]' for nitro group)."
                },
            },
            "required": [
                "smiles",
                "substructure_smarts"
            ]
        }
}
]


# ------------------------------------------------------------
# Agent state
# ------------------------------------------------------------
@dataclass
class AgentState:
    original_smiles: str
    current_smiles: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None


# ------------------------------------------------------------
# Tool executor
# ------------------------------------------------------------
def execute_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"success": False, "error": "Unknown tool"}
    return TOOLS[name](**args)


# ------------------------------------------------------------
# Agent loop (OpenAI Responses API)
# ------------------------------------------------------------
def run_agent(
    initial_smiles: str,
    user_goal: str,
    max_steps: int = 20
) -> AgentState:

    state = AgentState(
        original_smiles=initial_smiles,
        current_smiles=initial_smiles
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a molecular design agent.\n"
                "You may ONLY modify molecules using tools.\n"
                "Never edit SMILES directly.\n"
                "Always call get_attachment_points() after every SMILES modification to obtain valid attachment indices.\n"
                "When finished, respond with FINAL_ANSWER."
            )
        },
        {
            "role": "user",
            "content": (
                f"Goal: {user_goal}\n"
                f"Initial SMILES: {initial_smiles}"
            )
        }
    ]

    for _ in range(max_steps):
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=messages,
            tools=TOOL_SCHEMAS
        )
        messages += response.output
        for msg in response.output:
            print(msg)

            # Tool call
            if msg.type == "function_call":
                tool_name = msg.name
                args = json.loads(msg.arguments)

                # Enforce current SMILES
                if "smiles" in args:
                    args["smiles"] = state.current_smiles

                result = execute_tool(tool_name, args)
                print(result)

                if "new_smiles" in result:
                    state.current_smiles = result["new_smiles"]

                state.history.append({
                    "tool": tool_name,
                    "arguments": args,
                    "result": result
                })

                messages.append({
                    "type": "function_call_output",
                    "call_id": msg.call_id,
                    "output": json.dumps(result)
                })
                continue

        # Final or intermediate message
        if response.output[0].type == "message":
            content = response.output[0].content[0].text
            messages.append({
                "role": "assistant",
                "content": content
            })
            if "FINAL_ANSWER" in content:
                state.final_answer = content
                
                messages.append({"role": "user", "content": "Summarize the change(s) you made."})
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=messages,
                    tools=TOOL_SCHEMAS
                )
                print("SUMMARY: " + response.output[0].content[0].text)
                break

    return state

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    state = run_agent(
        initial_smiles="Cc1ccc(-c2cc(N)c(=O)n([C@H](C)C(=O)NC3CCCCC3)n2)o1",
        user_goal="Improve binding affinity to the protein kinase c-MET. Ensure that your modified SMILES does not grow too much; do not let molecular weight exceed 700.\n"

    )

    print("Final SMILES:", state.current_smiles)
    print("Final Answer:", state.final_answer)
    print("Steps taken:", len(state.history))