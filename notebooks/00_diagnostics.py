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
import numpy;     print("numpy          :", numpy.__version__)
import ase;       print("ASE            :", ase.__version__)
import nglview;   print("nglview        :", nglview.__version__)
import ipywidgets; print("ipywidgets     :", ipywidgets.__version__)
import cclib;     print("cclib          :", cclib.__version__)
try:
    import openbabel
    print("openbabel      : OK")
except ImportError as e:
    print("openbabel      : MISSING —", e)

# ============================================================
# 4. Run a minimal ORCA job via OPI
# ============================================================
print("\n=== 4. ORCA test job via OPI ===")

from opi.input.core import OrcaInput
from opi.input.simple_keywords.method import HF
from opi.input.simple_keywords.basis_set import BasisSet
from opi.input.structures.structure import Structure
from opi.execution.core import run_orca

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(scratch) / f"orca_diag_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)
print("Run directory  :", run_dir)

# Build a minimal H2O RHF/STO-3G single point via OPI
inp = (
    OrcaInput()
    .add_keywords(HF(), BasisSet.STO_3G)
    .set_structure(
        Structure.from_xyz_string(
            "3\nwater\nO  0.000  0.000  0.000\nH  0.757  0.000  0.586\nH -0.757  0.000  0.586"
        )
    )
    .set_nprocs(ncpus_to_use)
)

result = run_orca(inp, run_dir=run_dir)

print("Return code    :", result.returncode)
normal = result.normal_termination
print("Normal termination:", normal)
if not normal:
    print("\n--- ORCA output (last 30 lines) ---")
    lines = result.output.splitlines()
    print("\n".join(lines[-30:]))
    raise RuntimeError("ORCA did not terminate normally — check output above.")

print("SCF energy     :", result.properties.scf_energy.value, "Eh")
print("\nAll checks passed. The environment is ready for the course exercises.")

# ============================================================
# 5. Visualise H2O with nglview (run in its own cell)
# ============================================================
print("\n=== 5. Visualisation (run separately in notebook) ===")
print("Run the cell below to visualise H2O:")
print()
print("  from ase.build import molecule")
print("  import nglview as nv")
print("  view = nv.show_ase(molecule('H2O'))")
print("  view")
