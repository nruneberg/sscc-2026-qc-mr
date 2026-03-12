"""
utils.py — shared helpers for SSCC 2026 Multireference QC notebooks
--------------------------------------------------------------------
Usage in notebooks:
    from utils import setup_workdir, run_orca, get_energy, get_no_occupations

Each notebook calls setup_workdir('<name>') at the top to create and
validate its working directory (e.g. 'h2', 'n2', 'cyc').
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────

HARTREE_TO_EV    = 27.2114
HARTREE_TO_KCAL  = 627.509
HARTREE_TO_KJMOL = 2625.5

# ── ORCA executable ──────────────────────────────────────────────────────

def _get_orca():
    """Return path to ORCA executable, with a clear error if not found."""
    orca_home = os.environ.get('ORCA_HOME', '')
    if not orca_home:
        raise EnvironmentError(
            "ORCA_HOME is not set.\n"
            "Make sure the orca-env module is loaded:\n"
            "  module load orca-env"
        )
    orca = Path(orca_home) / 'orca'
    if not orca.exists():
        raise EnvironmentError(f"ORCA executable not found at {orca}")
    return str(orca)

ORCA = _get_orca()


# ── Working directory management ─────────────────────────────────────────

# File extensions that indicate stale calculation output
_STALE_PATTERNS = [
    '*.gbw', '*.out', '*.inp', '*.densities', '*.densitiesinfo',
    '*.tmp', '*.bas', '*.prop', '*.engrad', '*.hess', '*.trj',
    '*.bibtex', '*.property.txt',
]

def _count_stale(work_dir: Path) -> list:
    """Return list of stale files in work_dir."""
    stale = []
    for pattern in _STALE_PATTERNS:
        stale.extend(work_dir.glob(pattern))
    return stale


def setup_workdir(name: str, auto_clean: bool = False) -> Path:
    """
    Create and validate the working directory for a notebook.

    Parameters
    ----------
    name : str
        Subdirectory name, e.g. 'h2', 'n2', 'cyc'.
    auto_clean : bool
        If True, silently remove stale files without asking.
        Default False — warn and let the student decide.

    Returns
    -------
    Path
        The working directory path (already created).

    Examples
    --------
    >>> work_dir = setup_workdir('cyc')
    >>> # Then prefix all file paths: work_dir / 'label.inp'
    """
    work_dir = Path(name)
    work_dir.mkdir(exist_ok=True)

    stale = _count_stale(work_dir)

    if not stale:
        print(f"✓  Working directory '{name}/' is clean and ready.")
    elif auto_clean:
        for f in stale:
            f.unlink()
        print(f"✓  Removed {len(stale)} stale files from '{name}/'.")
    else:
        print(f"⚠️   Found {len(stale)} files from a previous run in '{name}/'.")
        print( "    Stale .gbw files in particular can cause CASSCF to read")
        print( "    wrong starting orbitals and converge to an incorrect state.")
        print(f"    Run  clean_workdir('{name}')  to remove them before proceeding.")

    return work_dir


def clean_workdir(name: str) -> None:
    """
    Remove all stale calculation files from a working directory.

    The directory itself is preserved. Use this when setup_workdir()
    warns about stale files from a previous run.

    Parameters
    ----------
    name : str
        Subdirectory name, e.g. 'h2', 'n2', 'cyc'.
    """
    work_dir = Path(name)
    if not work_dir.exists():
        work_dir.mkdir()
        print(f"✓  '{name}/' did not exist — created fresh.")
        return

    stale = _count_stale(work_dir)
    if not stale:
        print(f"✓  '{name}/' is already clean.")
        return

    for f in stale:
        f.unlink()
    print(f"✓  Removed {len(stale)} files from '{name}/'. Ready for a fresh run.")


# ── ORCA job runner ───────────────────────────────────────────────────────

def run_orca(label: str, input_text: str, work_dir: Path = Path('.')) -> Path:
    """
    Write an ORCA input file, run ORCA, return path to the .out file.

    Parameters
    ----------
    label : str
        Job label, used as the base filename (no extension).
    input_text : str
        Full ORCA input file contents.
    work_dir : Path
        Directory in which to write input/output files.
        Defaults to current directory.

    Returns
    -------
    Path
        Path to the .out file.
    """
    inp = work_dir / f"{label}.inp"
    out = work_dir / f"{label}.out"
    inp.write_text(input_text)
    with open(out, 'w') as fh:
        subprocess.run(
            [ORCA, str(inp)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(work_dir),
        )
    return out


# ── Output parsers ────────────────────────────────────────────────────────

def get_energy(out_file) -> float:
    """
    Return the last 'FINAL SINGLE POINT ENERGY' from an ORCA output file.

    Works for RHF, UHF, DFT, and CASSCF outputs.
    Returns float('nan') if not found.
    """
    text = Path(out_file).read_text()
    m = re.findall(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)', text)
    return float(m[-1]) if m else float('nan')


def get_casscf_energy(out_file) -> float:
    """
    Return the CASSCF total energy from an ORCA output file.

    Returns float('nan') if not found.
    """
    text = Path(out_file).read_text()
    m = re.findall(r'CASSCF TOTAL ENERGY:\s+(-?\d+\.\d+)', text)
    return float(m[-1]) if m else float('nan')


def get_nevpt2_energy(out_file) -> float:
    """
    Return the NEVPT2 total energy from an ORCA output file.

    Returns float('nan') if not found.
    """
    text = Path(out_file).read_text()
    m = re.findall(r'NEVPT2 Total Energy:\s+(-?\d+\.\d+)', text)
    return float(m[-1]) if m else float('nan')


def get_s2(out_file) -> float:
    """
    Return the <S²> expectation value from a UHF ORCA output file.

    Returns float('nan') if not found.
    """
    text = Path(out_file).read_text()
    m = re.findall(r'<S\*\*2>\s*=\s*(-?\d+\.\d+)', text)
    return float(m[-1]) if m else float('nan')


def get_no_occupations(out_file) -> list:
    """
    Extract CASSCF natural orbital occupation numbers, sorted descending.

    Returns a list of floats, or an empty list if not found.
    """
    text = Path(out_file).read_text()
    blocks = re.findall(
        r'Natural Orbital Occupation Numbers.*?\n((?:\s*\d+\.\d+\s*\n?)+)',
        text, re.IGNORECASE
    )
    if not blocks:
        return []
    return sorted([float(x) for x in blocks[-1].split()], reverse=True)


def terminated_normally(out_file) -> bool:
    """Return True if ORCA output contains TERMINATED NORMALLY."""
    text = Path(out_file).read_text()
    return 'TERMINATED NORMALLY' in text


# ── Geometry helpers ──────────────────────────────────────────────────────

def parse_xyz(filename) -> list:
    """
    Parse an XYZ file.

    Returns
    -------
    list of (element, x, y, z) tuples
    """
    lines = Path(filename).read_text().strip().split('\n')
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            atoms.append((parts[0],
                          float(parts[1]),
                          float(parts[2]),
                          float(parts[3])))
    return atoms


def bond_length(atoms: list, i: int, j: int) -> float:
    """Return the distance between atoms i and j (0-indexed)."""
    xi, yi, zi = atoms[i][1:]
    xj, yj, zj = atoms[j][1:]
    return float(np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2))


def stretch_bond(atoms: list, i: int, j: int, new_dist: float) -> list:
    """
    Return a new atom list with bond i–j stretched to new_dist.

    The midpoint of the bond is kept fixed; both atoms move symmetrically.
    """
    atoms = [list(a) for a in atoms]
    xi, yi, zi = atoms[i][1:]
    xj, yj, zj = atoms[j][1:]
    mx, my, mz = (xi + xj) / 2, (yi + yj) / 2, (zi + zj) / 2
    dx, dy, dz = xi - mx, yi - my, zi - mz
    norm  = np.sqrt(dx**2 + dy**2 + dz**2)
    scale = (new_dist / 2) / norm
    atoms[i][1] = mx + dx * scale
    atoms[i][2] = my + dy * scale
    atoms[i][3] = mz + dz * scale
    atoms[j][1] = mx - dx * scale
    atoms[j][2] = my - dy * scale
    atoms[j][3] = mz - dz * scale
    return [tuple(a) for a in atoms]


def atoms_to_xyz_block(atoms: list) -> str:
    """
    Format atom list as a geometry block for an ORCA input file.

    Returns a string with one 'El  x  y  z' line per atom,
    suitable for embedding between '* xyz charge mult' and '*'.
    """
    lines = [f"{el:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}"
             for el, x, y, z in atoms]
    return '\n'.join(lines)
