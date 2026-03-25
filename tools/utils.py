# tools/utils.py — SSCC 2026 Multireference QC shared utilities
"""
utils.py — shared helpers for SSCC 2026 Multireference QC notebooks
--------------------------------------------------------------------
Usage in notebooks:
    from utils import setup_workdir, run_orca, get_energy, get_no_occupations

Each notebook calls setup_workdir('<name>') at the top to create and
validate its working directory (e.g. 'h2', 'n2', 'ethylene').

Geometry helpers
----------------
parse_xyz(filename)
    Parse an xyz file into a list of (element, x, y, z) tuples.
bond_length(atoms, i, j)
    Distance between atoms i and j in an atom list.
get_distance(xyz_file, i, j)
    Distance between atoms i and j directly from an xyz file.
stretch_bond(atoms, i, j, new_dist)
    Return a new atom list with bond i-j stretched to new_dist.
atoms_to_xyz_block(atoms)
    Format atom list as an ORCA geometry block string.
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

# Number of CPU cores available for ORCA jobs.
# On Mahti (sinteractive, OOD, sbatch) this is set by SLURM automatically.
# On a local workstation without SLURM it falls back to 1.
NPROCS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

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
    '*.tmp', '*.bas', '*.bas0', '*.bas1', '*.bas2', '*.bas3', '*.bas4', '*.bas5',
    '*.prop', '*.engrad', '*.hess', '*.trj',
    '*.bibtex', '*.property.txt', '*.property.json', '*.json',
    '*.hostnames', '*.cube', '*.xyz', '*.pdf',
    '*.casinp.tmp.VEC.tmp.0', '*.cpscfdata.tmp.0', '*.propint.tmp.0',
    '*.mo*a.cube', '*.mo*b.cube',
    '*.casci*', '*.cpcm*',
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
        Subdirectory name, e.g. 'h2', 'n2', 'ethylene'.
    auto_clean : bool
        If True, silently remove stale files without asking.
        Default False — warn and let the student decide.

    Returns
    -------
    Path
        The working directory path (already created).

    Examples
    --------
    >>> work_dir = setup_workdir('ethylene')
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
        Subdirectory name, e.g. 'h2', 'n2', 'ethylene'.
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

def run_orca(label: str, input_text: str, work_dir: Path = Path('.'),
             nprocs: int = None) -> Path:
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
    nprocs : int, optional
        Number of MPI processes. If > 1 and no %pal block is present in
        input_text, a '%pal nprocs N end' block is prepended automatically.
        If None (default), no %pal block is injected — single-core safe.
        Use NPROCS from this module for jobs that should use all available
        cores, or pass 1 explicitly for small systems where parallelism
        would cause ORCA to abort (e.g. H2 with a minimal basis).

    Returns
    -------
    Path
        Path to the .out file.

    Examples
    --------
    >>> # Single-core (safe for small systems)
    >>> outfile = run_orca('h2_scan', inp, work_dir)

    >>> # Parallel (use all Slurm-allocated cores)
    >>> from utils import NPROCS
    >>> outfile = run_orca('ethylene_scan', inp, work_dir, nprocs=NPROCS)
    """
    if nprocs is not None and nprocs > 1 and '%pal' not in input_text.lower():
        input_text = f'%pal nprocs {nprocs} end\n\n' + input_text

    inp = work_dir / f"{label}.inp"
    out = work_dir / f"{label}.out"
    inp.write_text(input_text)
    with open(out, 'w') as fh:
        subprocess.run(
            [ORCA, f"{label}.inp"],
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
    # ORCA 6.x: "Final CASSCF energy       : -154.823760436 Eh"
    m = re.findall(r'Final CASSCF energy\s*:\s*(-?\d+\.\d+)', text)
    if m:
        return float(m[-1])
    # fallback for older ORCA versions
    m = re.findall(r'CASSCF TOTAL ENERGY:\s+(-?\d+\.\d+)', text)
    return float(m[-1]) if m else float('nan')


def get_nevpt2_energy(out_file) -> float:
    """
    Return the NEVPT2 total energy from an ORCA output file.

    Returns float('nan') if not found.
    """
    text = Path(out_file).read_text()
    # ORCA 6.x: "Total Energy (E0+dE)    : E  = -155.32997444957246"
    m = re.findall(r'Total Energy \(E0\+dE\)\s*:\s*E\s*=\s*(-?\d+\.\d+)', text)
    if m:
        return float(m[-1])
    # fallback for older ORCA versions
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

    Parses the ORBITAL ENERGIES table in ORCA 6.x CASSCF output:
        NO   OCC          E(Eh)            E(eV)
         0   1.9764      -0.584955       -15.9174
         1   0.0236       0.619446        16.8560

    Returns only non-zero occupations (active space), sorted descending.
    Returns an empty list if not found.
    """
    text = Path(out_file).read_text()

    # ORCA 6.x: parse ORBITAL ENERGIES table after CASSCF RESULTS
    blocks = list(re.finditer(
        r'ORBITAL ENERGIES\s*\n-+\n\s*NO\s+OCC\s+E\(Eh\).*?\n'
        r'((?:\s*\d+\s+\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)',
        text, re.IGNORECASE
    ))
    if blocks:
        block = blocks[-1].group(1)
        occs = [float(line.split()[1])
                for line in block.strip().split('\n')
                if line.strip()]
        return sorted([o for o in occs if o > 1e-6], reverse=True)

    # Fallback: N(occ)= line from CASSCF iterations
    m = re.findall(r'N\(occ\)=\s*([\d\.\s]+)', text)
    if m:
        return sorted([float(x) for x in m[-1].split()], reverse=True)

    return []


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
    """
    Return the distance between atoms i and j (0-indexed) in an atom list.

    Parameters
    ----------
    atoms : list of (element, x, y, z) tuples
        As returned by parse_xyz().
    i, j : int
        0-based atom indices.

    Returns
    -------
    float
        Distance in Å.

    See Also
    --------
    get_distance : convenience wrapper that reads directly from an xyz file.
    """
    xi, yi, zi = atoms[i][1:]
    xj, yj, zj = atoms[j][1:]
    return float(np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2))


def get_distance(xyz_file, i: int, j: int) -> float:
    """
    Return the distance between atoms i and j (0-indexed) in an xyz file.

    Convenience wrapper around parse_xyz() + bond_length() for use as a
    sort key or in list comprehensions without loading the full atom list.

    Parameters
    ----------
    xyz_file : str or Path
        Path to the xyz file.
    i, j : int
        0-based atom indices.

    Returns
    -------
    float
        Distance in Å.

    Examples
    --------
    >>> # O-C1 distance in an ethylene+O scan step (atoms 0 and 2)
    >>> r = get_distance('ethylene/scan_nevpt2.001.xyz', 0, 2)

    >>> # Sort scan xyz files by O-C1 distance
    >>> files = sorted(work_dir.glob('scan_nevpt2.0*.xyz'),
    ...                key=lambda f: get_distance(f, 0, 2))
    """
    return bond_length(parse_xyz(xyz_file), i, j)


def stretch_bond(atoms: list, i: int, j: int, new_dist: float) -> list:
    """
    Return a new atom list with bond i–j stretched to new_dist.

    The midpoint of the bond is kept fixed; both atoms move symmetrically.

    Parameters
    ----------
    atoms : list of (element, x, y, z) tuples
        As returned by parse_xyz().
    i, j : int
        0-based atom indices defining the bond to stretch.
    new_dist : float
        Target bond length in Å.

    Returns
    -------
    list of (element, x, y, z) tuples
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

    Parameters
    ----------
    atoms : list of (element, x, y, z) tuples
        As returned by parse_xyz().

    Returns
    -------
    str
        Multi-line geometry block.
    """
    lines = [f"{el:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}"
             for el, x, y, z in atoms]
    return '\n'.join(lines)


# ── Orbital visualization ─────────────────────────────────────────────────

def plot_orbital(basename: str, mo_index: int, resolution: int = 60,
                 work_dir: Path = None) -> str:
    """
    Generate a cube string for a given MO using OPI's Output.plot_mo().

    Parameters
    ----------
    basename : str
        ORCA job basename (without extension), e.g. 'h2_seed_casscf'.
    mo_index : int
        0-based MO index to plot.
    resolution : int
        Grid resolution per axis (default 60).
    work_dir : Path, optional
        Working directory containing the .gbw and .out files.

    Returns
    -------
    str
        Cube file contents as a string, ready for py3Dmol.
    """
    try:
        from opi.output.core import Output
    except ImportError:
        raise ImportError("orca-pi (OPI) is required for orbital visualization.")

    output = Output(basename, working_dir=work_dir, parse=True)
    cube_output = output.plot_mo(mo_index, resolution=resolution)
    return cube_output.cube


def show_orbital(cube_file, isoval: float = 0.05,
                 label: str = '',
                 width: int = 500, height: int = 400):
    """
    Visualize an orbital from a Gaussian cube file using py3Dmol.

    Shows positive phase in blue and negative phase in red.

    Parameters
    ----------
    cube_file : str or Path
        Path to the .cube file, or cube file contents as a string.
    isoval : float
        Isovalue for the orbital surface (default 0.05).
    label : str
        Optional label printed above the viewer.
    width, height : int
        Viewer dimensions in pixels.
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError("py3Dmol is required for orbital visualization.")

    if isinstance(cube_file, str) and cube_file.strip().startswith('CUBE'):
        cube_text = cube_file
    elif isinstance(cube_file, str) and len(cube_file) > 5 and '\n' in cube_file:
        cube_text = cube_file
    else:
        cube_text = Path(cube_file).read_text()

    if label:
        print(label)

    view = py3Dmol.view(width=width, height=height)
    view.addVolumetricData(cube_text, 'cube',
                           {'isoval':  isoval,
                            'color':   'blue',
                            'opacity': 0.75})
    view.addVolumetricData(cube_text, 'cube',
                           {'isoval': -isoval,
                            'color':   'red',
                            'opacity': 0.75})
    view.addModel(cube_text, 'cube')
    view.setStyle({'stick': {'colorscheme': 'grayCarbon', 'radius': 0.1}})
    view.zoomTo()
    return view


def orbital_gallery(basename: str, mo_indices: list, occupations: list = None,
                    labels: list = None, isoval: float = 0.08,
                    resolution: int = 60, work_dir=None):
    """
    Display a gallery of orbitals using py3Dmol, one per cell output.

    Parameters
    ----------
    basename : str
        ORCA job basename (without extension), e.g. 'h2_seed_casscf'.
    mo_indices : list of int
        List of 0-based MO indices to display.
    occupations : list of float, optional
        Occupation numbers to annotate each orbital.
    labels : list of str, optional
        Labels for each orbital (e.g. ['σ_g', 'σ_u*']).
        Defaults to 'MO N'.
    isoval : float
        Isovalue for orbital surfaces (default 0.08).
    resolution : int
        Grid resolution per axis (default 60).
    work_dir : Path, optional
        Working directory.
    """
    if labels is None:
        labels = [f'MO {i}' for i in mo_indices]
    if occupations is None:
        occupations = [None] * len(mo_indices)

    views = []
    for idx, (mo, occ, lbl) in enumerate(zip(mo_indices, occupations, labels)):
        cube = plot_orbital(basename, mo, resolution=resolution, work_dir=work_dir)
        if cube is None:
            print(f"Skipping MO {mo} — cube generation failed")
            continue
        ann = f"{lbl}  (occ = {occ:.4f})" if occ is not None else lbl
        view = show_orbital(cube, isoval=isoval, label=ann)
        view.show()
        views.append(view)

    return views
