# ORCA / OPI Diagnostic Notebook
# CSC Spring School on Computational Chemistry 2026 - Mahti
#
# Purpose: verify that the full software stack (ORCA, OPI, nglview, ASE, ...)
# is correctly wired up before the actual course exercises.
# Run this cell-by-cell top to bottom; each section is self-contained.

import os
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. Environment sanity checks
# ============================================================
print("=== 1. Environment ===")

orca_home = os.environ.get("ORCA_HOME")
assert orca_home, "ORCA_HOME is not set — was the orca-env/1.0 module loaded?"
orca_exe = Path(orca_home) / "orca"
assert orca_exe.exists(), f"ORCA executable not found at {orca_exe}"
print("ORCA_HOME      :", orca_home)
print("ORCA executable:", orca_exe)

# Verify the mpirun wrapper is on PATH and is OUR wrapper, not the Spack stub
mpirun_on_path = subprocess.run(["which", "mpirun"], capture_output=True, text=True).stdout.strip()
print("mpirun on PATH :", mpirun_on_path)
orca_wrappers = os.environ.get("ORCA_WRAPPERS", "")
if orca_wrappers and mpirun_on_path.startswith(orca_wrappers):
    print("mpirun wrapper : OK (from ORCA_WRAPPERS)")
else:
    print("WARNING: mpirun is NOT from ORCA_WRAPPERS — parallel ORCA may fail.")
    print("         Expected prefix:", orca_wrappers)

# OOD_PORTAL tells us we are inside an Open OnDemand session
print("OOD_PORTAL     :", os.environ.get("OOD_PORTAL", "(not set — not an OOD session)"))
print("SLURM_JOB_ID   :", os.environ.get("SLURM_JOB_ID", "(not set)"))

# Determine the correct CPU count.
# OOD Jupyter for Courses allocates cores as Slurm *tasks* (--ntasks=N),
# not as threads per task. SLURM_CPUS_PER_TASK will therefore be 1.
# Use SLURM_NTASKS to get the actual number of cores available to ORCA.
ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
ncpus_to_use = ntasks * cpus_per_task
print(f"SLURM_NTASKS={ntasks}, SLURM_CPUS_PER_TASK={cpus_per_task} → using nprocs={ncpus_to_use}")

# Scratch directory: use LOCAL_SCRATCH (NVMe) if available, else cwd
scratch = os.environ.get("LOCAL_SCRATCH") or os.environ.get("TMPDIR") or str(Path.cwd())
print("Scratch dir    :", scratch)

# ============================================================
# 2. OPI import and version check
# ============================================================
print("\n=== 2. OPI ===")
import opi
print("opi version    :", opi.__version__)

# Verify ORCA is on PATH
import shutil
resolved = shutil.which("orca")
print("ORCA on PATH   :", resolved)
assert resolved, "orca not found on PATH — was orca-env/1.0 module loaded?"

# ============================================================
# 3. Supporting package imports
# ============================================================
print("\n=== 3. Supporting packages ===")
import numpy;      print("numpy          :", numpy.__version__)
import scipy;      print("scipy          :", scipy.__version__)
import pandas;     print("pandas         :", pandas.__version__)
import matplotlib; print("matplotlib     :", matplotlib.__version__)
import ase;        print("ASE            :", ase.__version__)
import cclib;      print("cclib          :", cclib.__version__)
import nglview;    print("nglview        :", nglview.__version__)
import ipywidgets; print("ipywidgets     :", ipywidgets.__version__)
try:
    import py3Dmol
    print("py3Dmol        : OK (import as py3Dmol)")
except ImportError as e:
    print("py3Dmol        : MISSING —", e)
try:
    import pubchempy
    print("pubchempy      : OK")
except ImportError as e:
    print("pubchempy      : MISSING —", e)
try:
    import openbabel
    print("openbabel      : OK")
except ImportError as e:
    print("openbabel      : MISSING —", e)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    mol = Chem.MolFromSmiles("O")
    assert mol is not None, "RDKit failed to parse water SMILES"
    print("rdkit          : OK (parsed water SMILES)")
except ImportError as e:
    print("rdkit          : MISSING —", e)

try:
    from openbabel import openbabel as ob
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "xyz")
    mol = ob.OBMol()
    conv.ReadString(mol, "O")
    xyz = conv.WriteString(mol)
    assert "O" in xyz, "OpenBabel failed to convert water SMILES to xyz"
    print("openbabel API  : OK (SMILES → xyz conversion)")
except Exception as e:
    print("openbabel API  : FAILED —", e)
    print("               → fallback: use obabel CLI via subprocess")

# ============================================================
# 4. Run a minimal ORCA job (raw input file)
# ============================================================
print("\n=== 4. ORCA test job ===")

import subprocess

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(scratch) / f"orca_diag_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)
print("Run directory  :", run_dir)

inp_file = run_dir / "h2o.inp"
out_file = run_dir / "h2o.out"

inp_file.write_text(f"""! RHF STO-3G
%pal nprocs 1 end
* xyz 0 1
O  0.000  0.000  0.000
H  0.757  0.000  0.586
H -0.757  0.000  0.586
*
""")

orca_exe = str(Path(orca_home) / "orca")
with open(out_file, "w") as fout:
    result = subprocess.run([orca_exe, str(inp_file)], stdout=fout, stderr=subprocess.STDOUT)

print("Return code    :", result.returncode)

output = out_file.read_text()
normal = "ORCA TERMINATED NORMALLY" in output
print("Normal termination:", normal)

if not normal:
    print("\n--- ORCA output (last 30 lines) ---")
    print("\n".join(output.splitlines()[-30:]))
    raise RuntimeError("ORCA did not terminate normally — check output above.")

# Parse SCF energy from output
for line in output.splitlines():
    if "FINAL SINGLE POINT ENERGY" in line:
        print("SCF energy     :", line.split()[-1], "Eh")
        break

print("\nAll checks passed. The environment is ready for the course exercises.")

# ============================================================
# 5. OPI Calculator interface (input builder + output parsing)
# ============================================================
print("\n=== 5. OPI Calculator interface ===")

from opi.core import Calculator
from opi.output.core import Output
from opi.input.simple_keywords import Dft, Task
from opi.input.structures.structure import Structure

opi_dir = Path(scratch) / f"orca_opi_{timestamp}"
opi_dir.mkdir(parents=True, exist_ok=True)

xyz_data = """3

O      0.00000   -0.00000    0.00000
H      0.00000    0.96899    0.00000
H      0.93966   -0.23409    0.03434
"""
with open(opi_dir / "struc.xyz", "w") as f:
    f.write(xyz_data)

structure = Structure.from_xyz(opi_dir / "struc.xyz")

calc = Calculator(basename="h2o_opi", working_dir=opi_dir)
calc.structure = structure
calc.input.add_simple_keywords(Dft.R2SCAN_3C, Task.SP)
calc.input.ncores = 1

calc.write_input()
print("OPI input written:", (opi_dir / "h2o_opi.inp").exists())
calc.run()

output = calc.get_output()
if not output.terminated_normally():
    raise RuntimeError("OPI ORCA job did not terminate normally")
output.parse()

energy = output.results_properties.geometries[0].single_point_data.finalenergy
print("OPI SCF energy   :", energy, "Eh")
print("OPI Calculator   : OK")

# ============================================================
# 6. Visualise H2O with nglview (run in its own cell)
# ============================================================
print("\n=== 6. Visualisation (run separately in notebook) ===")
print("Run the cell below to visualise H2O:")
print()
print("  from ase.build import molecule")
print("  import nglview as nv")
print("  view = nv.show_ase(molecule('H2O'))")
print("  view")
