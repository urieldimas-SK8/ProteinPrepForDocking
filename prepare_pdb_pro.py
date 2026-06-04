#!/usr/bin/env python3

import sys
import os
import numpy as np

from pdbfixer import PDBFixer
import openmm.app as app
import openmm as mm
from Bio.PDB import *

# ==============================
# CONFIG
# ==============================

PH = 7.4
MIN_STEPS = 500

REMOVE_HIS_TAGS = True
REMOVE_LIGAND = True

OUTPUT_SUFFIX = "_PRO_diffdock_ready.pdb"

STANDARD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY",
    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
    "THR","TRP","TYR","VAL"
}

# ==============================
# FUNCIONES
# ==============================

def load_structure(pdb_file):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("prot", pdb_file)

# ------------------------------

def remove_ligand_and_non_protein(structure):
    removed = 0
    for model in structure:
        for chain in model:
            for res in list(chain):
                # elimina HETATM (ligandos, solventes, etc.)
                if res.id[0] != " ":
                    chain.detach_child(res.id)
                    removed += 1
    return removed

# ------------------------------

def remove_terminal_his_tags(structure):
    removed = 0

    for model in structure:
        for chain in model:
            residues = list(chain)

            # N-terminal
            n_block = []
            for res in residues[:10]:
                if res.resname == "HIS":
                    n_block.append(res)
                else:
                    break

            if len(n_block) >= 4:
                for res in n_block:
                    chain.detach_child(res.id)
                    removed += 1

            # C-terminal
            c_block = []
            for res in reversed(residues[-10:]):
                if res.resname == "HIS":
                    c_block.append(res)
                else:
                    break

            if len(c_block) >= 4:
                for res in c_block:
                    chain.detach_child(res.id)
                    removed += 1

    return removed

# ------------------------------

def save_structure(structure, filename):
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)

# ------------------------------

def prepare_with_pdbfixer(input_pdb, output_pdb):
    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    missing_res = len(fixer.missingResidues)

    fixer.findMissingAtoms()
    missing_atoms = len(fixer.missingAtoms)

    fixer.findNonstandardResidues()
    nonstandard = len(fixer.nonstandardResidues)

    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=PH)

    with open(output_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return missing_res, missing_atoms, nonstandard

# ------------------------------

def minimize_structure(pdb_file, output_file):
    pdb = app.PDBFile(pdb_file)

    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    integrator = mm.LangevinIntegrator(
        300 * mm.unit.kelvin,
        1 / mm.unit.picosecond,
        0.002 * mm.unit.picoseconds
    )

    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    simulation.minimizeEnergy(maxIterations=MIN_STEPS)

    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    final_energy = final_state.getPotentialEnergy()

    with open(output_file, "w") as f:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)

    return initial_energy, final_energy

# ------------------------------

def calculate_rmsd_aligned(pdb1, pdb2):
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure("s1", pdb1)
    s2 = parser.get_structure("s2", pdb2)

    atoms1 = [a for a in s1.get_atoms() if a.get_id() == "CA"]
    atoms2 = [a for a in s2.get_atoms() if a.get_id() == "CA"]

    n = min(len(atoms1), len(atoms2))

    atoms1 = atoms1[:n]
    atoms2 = atoms2[:n]

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)

    return sup.rms

# ------------------------------

def analyze_basic(structure):
    residues = len(list(structure.get_residues()))
    chains = len(set(c.id for c in structure.get_chains()))
    return residues, chains

# ==============================
# MAIN
# ==============================

def main():

    if len(sys.argv) < 2:
        print("Uso: python prepare_pdb_pro.py archivo.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]

    base = input_pdb.replace(".pdb", "")
    step1 = base + "_clean.pdb"
    step2 = base + "_fixed.pdb"
    final_pdb = base + OUTPUT_SUFFIX

    print("\n🔍 Cargando estructura...")
    structure = load_structure(input_pdb)

    r0, c0 = analyze_basic(structure)

    print("🧹 Eliminando ligandos (HETATM)...")
    removed_lig = remove_ligand_and_non_protein(structure)

    print("✂ Eliminando His-tags...")
    removed_his = remove_terminal_his_tags(structure)

    save_structure(structure, step1)

    print("🔧 Reparando con pdbfixer...")
    miss_res, miss_atoms, nonstd = prepare_with_pdbfixer(step1, step2)

    print("⚛️ Minimización energética...")
    e0, e1 = minimize_structure(step2, final_pdb)

    print("📏 Calculando RMSD (alineado)...")
    rmsd = calculate_rmsd_aligned(step2, final_pdb)

    final_structure = load_structure(final_pdb)
    r1, c1 = analyze_basic(final_structure)

    # ==========================
    # REPORTE
    # ==========================

    print("\n==============================")
    print("🧬 REPORTE PRO DE PREPARACIÓN")
    print("==============================\n")

    print("📊 Estructura:")
    print(f"  Residuos: {r0} → {r1}")
    print(f"  Cadenas: {c0} → {c1}")

    print("\n🧹 Limpieza:")
    print(f"  Ligandos eliminados: {removed_lig}")
    print(f"  His-tags eliminados: {removed_his}")

    print("\n🔧 Reparación:")
    print(f"  Missing residues: {miss_res}")
    print(f"  Missing atoms: {miss_atoms}")
    print(f"  No estándar: {nonstd}")

    print("\n⚛️ Energía:")
    print(f"  Inicial: {e0}")
    print(f"  Final:   {e1}")

    print("\n📏 RMSD:")
    print(f"  Backbone CA RMSD (alineado): {rmsd:.3f} Å")

    print("\n📁 Output:")
    print(f"  {final_pdb}")

    print("\n✅ Evaluación:")

    if rmsd < 2.0:
        print("  ✔ Estructura estable")
    else:
        print("  ⚠ Cambios estructurales grandes")

    if miss_res > 10:
        print("  ⚠ Muchos residuos reconstruidos")

    print("\n==============================\n")


if __name__ == "__main__":
    main()
