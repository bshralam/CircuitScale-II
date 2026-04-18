from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
from pyscf import gto, scf, mcscf, fci
from openfermion import MolecularData
from openfermion.transforms import jordan_wigner
from openfermion.utils import count_qubits
from openfermionpyscf import run_pyscf

from .systems import MoleculeSpec

def build_mol(spec: MoleculeSpec):
    mol = gto.Mole()
    mol.atom = [(sym, *xyz) for sym, xyz in spec.geometry]
    mol.basis = spec.basis
    mol.charge = spec.charge
    mol.spin = spec.multiplicity - 1
    mol.unit = "Angstrom"
    mol.build()
    return mol

def run_rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf

def run_fci_energy(mf) -> float:
    cisolver = fci.FCI(mf.mol, mf.mo_coeff)
    e, _ = cisolver.kernel()
    return float(e)

def run_casscf_noons(mf, ncas: int, nelecas):
    mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
    mc.conv_tol = 1e-10
    mc.kernel()
    rdm1 = mc.make_rdm1()
    a0 = mc.ncore
    a1 = mc.ncore + mc.ncas
    rdm1_act = rdm1[a0:a1, a0:a1]
    noons = np.linalg.eigvalsh(rdm1_act)[::-1]
    return noons, float(mc.e_tot)

def mr_score_from_noons(noons: np.ndarray) -> float:
    noons = np.array(noons, dtype=float)
    return float(np.sum(np.minimum(noons, 2.0 - noons)))

def compute_reference(spec: MoleculeSpec, ncas: int, nelecas) -> Dict[str, Any]:
    mol = build_mol(spec)
    mf = run_rhf(mol)
    ehf = float(mf.e_tot)
    efci = run_fci_energy(mf)
    noons, ecas = run_casscf_noons(mf, ncas=ncas, nelecas=nelecas)
    mr = mr_score_from_noons(noons)
    return {
        "E_HF": ehf,
        "E_FCI": efci,
        "E_CASSCF": ecas,
        "noons": noons,
        "mr_score": mr,
        "n_electrons": mol.nelectron,
    }

def molecular_qubit_hamiltonian(spec: MoleculeSpec) -> Tuple[object, int]:
    geometry = [(sym, tuple(xyz)) for sym, xyz in spec.geometry]
    mol = MolecularData(
        geometry=geometry,
        basis=spec.basis,
        multiplicity=spec.multiplicity,
        charge=spec.charge,
        filename=None,
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False)
    fermion_ham = mol.get_molecular_hamiltonian()
    qubit_ham = jordan_wigner(fermion_ham)
    n_qubits = count_qubits(qubit_ham)
    return qubit_ham, n_qubits
