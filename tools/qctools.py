# tools/qctools.py
# Static copy of Antti Karttunen's qctools.py with local additions.
# Original: https://github.com/anttikarttunen/sscc2026-qc/blob/main/tools/qctools.py
# Last synced: 2026-03-17
# Contact: antti.karttunen@aalto.fi
#
# Local additions (SSCC 2026 MR course):
#   build_xyz_trajectory() — sort and concatenate xyz files into a trajectory


############## Functions to facilitate printing ##############

def print_info(info):
    banner = "-----------------------------------------------------------"
    print(f"{banner}\n{info}\n{banner}")


def print_error(error):
    err = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    print(f"{err}\n{error}\n{err}")


############## Functions for loading molecules ##############

def load_xyz(xyzfile, silent=False):
    """
    Load an XYZ file.

    Parameters
    ----------
    xyzfile : str or Path
        Path to the xyz file.
    silent : bool
        If True, suppress info/error messages.

    Returns
    -------
    ase.Atoms or None
    """
    from ase.io import read
    try:
        atoms = read(xyzfile, format='xyz')
        if not silent:
            print_info(f"Molecule was loaded from file {xyzfile}\n"
                       f"Atoms: {len(atoms)}\n"
                       f"Formula: {str(atoms.symbols)}")
    except OSError:
        if not silent:
            print_error(f"Failed to load file {xyzfile}")
        return None
    else:
        atoms.info['id'] = xyzfile
        return atoms


def load_xyz_as_traj(xyztraj, silent=False):
    """
    Convert a multi-frame XYZ file to an ASE trajectory.

    Writes a temporary .traj file alongside the input and returns a
    TrajectoryReader. Suitable for use with show_molecule().

    Parameters
    ----------
    xyztraj : str or Path
        Path to the multi-frame xyz file.
    silent : bool
        If True, suppress info/error messages.

    Returns
    -------
    ase.io.trajectory.Trajectory or None
    """
    from ase.io import read, Trajectory
    try:
        frames = read(xyztraj, index=':')
        if not silent:
            print_info(f"XYZ trajectory loaded from {xyztraj}\n"
                       f"Frames: {len(frames)}")
    except OSError:
        if not silent:
            print_error(f"Failed to load file {xyztraj}")
        return None
    trajfile = xyztraj + '.traj'
    with Trajectory(trajfile, mode='w') as traj:
        for frame in frames:
            traj.write(frame)
    return Trajectory(trajfile)


############## Functions for building trajectories ##############

def build_xyz_trajectory(xyz_files, output_path, key=None, reverse=False):
    """
    Concatenate xyz files into a single multi-frame trajectory file.

    Optionally sorts the files by a key function before concatenating,
    which is useful for assembling scan steps into a reaction path
    ordered by a geometric coordinate (e.g. a bond distance).

    Parameters
    ----------
    xyz_files : list of str or Path
        Input xyz files to concatenate.
    output_path : str or Path
        Path to write the combined trajectory.
    key : callable, optional
        Sort key function applied to each file path.
        If None, files are written in the order given.
        Example: ``key=lambda f: get_distance(f, 0, 2)``
    reverse : bool
        If True, sort in descending order. Default False.
        Use ``reverse=True`` to produce an approach trajectory
        (large R first).

    Returns
    -------
    list of Path
        File paths in the order they were written.

    Examples
    --------
    >>> from utils import get_distance
    >>> from qctools import build_xyz_trajectory
    >>>
    >>> # Assemble O-attack scan in approach order (large R -> small R)
    >>> all_xyz = list(work_dir.glob('scan_nevpt2.0*.xyz')) + \\
    ...           list(work_dir.glob('scan_dense.0*.xyz'))
    >>> frames = build_xyz_trajectory(
    ...     all_xyz,
    ...     work_dir / 'full_path.xyz',
    ...     key=lambda f: get_distance(f, 0, 2),
    ...     reverse=True
    ... )
    """
    from pathlib import Path as _Path
    files = [_Path(f) for f in xyz_files]
    if key is not None:
        files = sorted(files, key=key, reverse=reverse)
    with open(output_path, 'w') as fh:
        for f in files:
            fh.write(f.read_text())
    print(f'Written {len(files)} frames to {output_path}')
    return files


############## Functions for visualizing molecules with nglview ##############

def show_molecule(molecule, size=(500, 400), style='ball+stick',
                  unitcell=None, labels=None, bg='black'):
    """
    Show a molecule or trajectory using NGLWidget.

    Parameters
    ----------
    molecule : ase.Atoms or ase.io.trajectory.TrajectoryReader
        Single structure or trajectory as returned by load_xyz_as_traj().
    size : tuple of (width, height)
        Viewer dimensions in pixels.
    style : str
        NGL representation style, e.g. 'ball+stick', 'spacefill', 'licorice'.
    unitcell : None or str
        If not None, add a unit cell representation with this color.
    labels : None or str
        Atom label type: None, 'atomname', 'atomindex', or 'element'.
    bg : str
        Background color (default 'black').

    Returns
    -------
    nglview.NGLWidget

    Examples
    --------
    >>> traj = load_xyz_as_traj('ethylene/full_path.xyz', silent=True)
    >>> view = show_molecule(traj, size=(500, 400), bg='white')
    >>> view
    """
    import nglview
    from ase import Atoms
    from ase.io.trajectory import TrajectoryReader
    if isinstance(molecule, Atoms):
        nv = nglview.show_ase(molecule)
    elif isinstance(molecule, TrajectoryReader):
        nv = nglview.show_asetraj(molecule)
    else:
        print_error("Invalid molecule!")
        return None
    nv._set_size(f"{size[0]}px", f"{size[1]}px")
    nv.clear_representations()
    nv.add_representation(style)
    if style == 'spacefill':
        nv.update_representation(component=0, repr_index=0, radiusType='covalent')
    if unitcell is not None:
        nv.add_representation('unitcell')
    if labels is not None:
        nv.add_representation('label', labelType=labels)
    nv.parameters = dict(backgroundColor=bg, clipDist=-100)
    nv.camera = 'orthographic'
    nv.display()
    return nv


############## Functions for orbital phase alignment ##############

def phase_align_cube(cube_str, reference_cube_str):
    """
    Align the phase of a cube file to a reference cube file.

    Natural orbital phases are arbitrary — ORCA may flip the sign between
    geometry steps even with warm-starting. This function attempts to ensure
    visual continuity in orbital galleries by flipping the phase of
    ``cube_str`` if it appears misaligned with ``reference_cube_str``.

    .. warning::
        This function is unreliable when the orbital character changes
        significantly between geometries. The heuristic used (sign of the
        target cube at the position of the reference maximum) can fail when
        the reference maximum falls near a nodal region of the target, or
        when the dominant lobe shifts region between steps. In practice this
        causes incorrect flips for some orbital rows in the 4-column gallery.

        A robust implementation requires computing the overlap between MO
        coefficient vectors from the .gbw files — this is the planned
        ``match_orbitals()`` function in qctools.py. Until that is
        implemented, phase_align_cube should not be used in production.
        The gallery notebook uses no phase alignment and notes that orbital
        sign is arbitrary and carries no physical meaning.

    Parameters
    ----------
    cube_str : str
        Cube file content to be phase-aligned.
    reference_cube_str : str
        Reference cube file content to align against.

    Returns
    -------
    str
        Cube file content, possibly with flipped phase.
    """
    import numpy as np

    def _parse_cube(cube):
        lines = cube.strip().splitlines()
        # Cube format: 2 comment lines, 1 natoms+origin line, 3 axis lines,
        # natoms atom lines, then volumetric data
        natoms = abs(int(lines[2].split()[0]))
        data_start = 6 + natoms
        values = np.array([
            float(v)
            for line in lines[data_start:]
            for v in line.split()
        ])
        return lines, data_start, values

    lines1, data_start1, v1 = _parse_cube(cube_str)
    _,      _,           v2 = _parse_cube(reference_cube_str)

    # Phase alignment strategy: find where the reference orbital has its
    # largest amplitude, then check if the target has the same sign there.
    # This is more robust than comparing global max-voxel signs or dot products
    # when the orbital redistributes weight between geometry steps.
    # The only failure case is if the reference maximum falls in a nodal
    # region of the target — unlikely for adjacent warm-started geometries.
    ref_max_idx = np.argmax(np.abs(v2))          # where is reference largest?
    sign_ref    = np.sign(v2[ref_max_idx])        # reference sign there
    sign_target = np.sign(v1[ref_max_idx])        # target sign at same position

    if sign_target == sign_ref:
        return cube_str  # phases already aligned

    # Flip phase: negate all volumetric data
    neg_values = -v1
    header = '\n'.join(lines1[:data_start1])
    data_lines = []
    for i in range(0, len(neg_values), 6):
        chunk = neg_values[i:i+6]
        data_lines.append('  '.join(f'{v:12.5E}' for v in chunk))
    return header + '\n' + '\n'.join(data_lines) + '\n'


############## Functions for PES scan (NB02 O(3P)+ethylene) ##############

def _extract_relaxed_geom(outfile):
    """Extract last relaxed geometry from ORCA output file."""
    from pathlib import Path
    text = Path(outfile).read_text()
    idx = text.rfind('CARTESIAN COORDINATES (ANGSTROEM)')
    if idx < 0:
        return None
    block = text[idx:].splitlines()
    gl = []
    for line in block[2:]:
        parts = line.split()
        if len(parts) == 4 and parts[0].isalpha():
            gl.append(f'{parts[0]:2s}  {float(parts[1]):12.6f}  {float(parts[2]):12.6f}  {float(parts[3]):12.6f}')
        elif gl:
            break
    return '\n'.join(gl) if gl else None


def run_casscf_scan(R_grid, start_geometry, work_dir, nprocs=1,
                    nel=6, norb=6, mult=3, basis='cc-pVDZ',
                    maxiter=200):
    """
    Run a CASSCF constrained relaxed scan on a custom R grid.

    Each point is a geometry optimisation with the O-C1 distance
    (atoms 0 and 2) fixed at the target R. Points are computed inward
    (large R first) with warm-starting: orbitals and relaxed geometry
    from each step are passed to the next via MORead.

    Already-converged points are skipped on re-run — the function is
    safe to call multiple times if the calculation was interrupted.

    Parameters
    ----------
    R_grid : list of float
        O-C1 distances in Angstrom, ordered large R first.
    start_geometry : str
        XYZ coordinate block for the starting geometry.
    work_dir : Path
        Working directory for ORCA input/output files.
    nprocs : int
        Number of MPI processes (default 1).
    nel, norb, mult : int
        CASSCF active space parameters.
    basis : str
        Basis set keyword.
    maxiter : int
        Maximum CASSCF macro-iterations.

    Returns
    -------
    list of (float, float)
        List of (R, energy_Eh) pairs in R_grid order.
    """
    from pathlib import Path
    from utils import run_orca, get_energy, terminated_normally

    work_dir = Path(work_dir)
    results  = []
    prev_gbw  = None
    prev_geom = start_geometry

    for R in R_grid:
        tag     = f'casscf_R{R:.2f}'.replace('.', 'p')
        gbw_out = (work_dir / tag).with_suffix('.gbw').resolve()
        out_file = work_dir / f'{tag}.out'

        # Skip if already successfully completed
        if out_file.exists() and terminated_normally(out_file):
            E = get_energy(out_file)
            print(f'R = {R:.2f} A ... skipped (already done)  E = {E:.6f} Eh')
            results.append((R, E))
            if gbw_out.exists():
                prev_gbw = gbw_out
            geom = _extract_relaxed_geom(out_file)
            if geom:
                prev_geom = geom
            continue

        moread = f'! MORead\n%moinp "{prev_gbw}"\n\n' if prev_gbw else ''

        inp = f"""{moread}! CASSCF {basis} TightSCF Opt

%maxcore 4000

%geom
  Constraints
    {{B 0 2 {R:.3f} C}}
  end
end

%casscf
  nel {nel}
  norb {norb}
  mult {mult}
  nroots 1
  MaxIter {maxiter}
end

* xyz 0 3
{prev_geom}
*
"""
        print(f'R = {R:.2f} A ...', flush=True)
        outfile = run_orca(tag, inp, work_dir, nprocs=nprocs)
        E  = get_energy(outfile)
        ok = terminated_normally(outfile)
        print(f'  E = {E:.6f} Eh  ({"OK" if ok else "FAILED"})', flush=True)
        results.append((R, E))

        if gbw_out.exists():
            prev_gbw = gbw_out
        geom = _extract_relaxed_geom(outfile)
        if geom:
            prev_geom = geom

    print(f'\nAll CASSCF points done. {len(results)} geometries.')
    return results


def run_nevpt2_scan(casscf_results, work_dir, nprocs=1,
                    nel=6, norb=6, mult=3, basis='cc-pVDZ'):
    """
    Run NEVPT2 single points on CASSCF-relaxed geometries.

    Uses the CASSCF .gbw file from each point as orbital guess via MORead.
    Already-converged points are skipped on re-run.

    Parameters
    ----------
    casscf_results : list of (float, float)
        Output from run_casscf_scan: list of (R, E_casscf) pairs.
    work_dir : Path
        Working directory.
    nprocs : int
        Number of MPI processes.
    nel, norb, mult : int
        CASSCF active space parameters (must match the scan).
    basis : str
        Basis set keyword.

    Returns
    -------
    list of (float, float)
        List of (R, E_nevpt2_Eh) pairs.
    """
    from pathlib import Path
    from utils import run_orca, get_nevpt2_energy, terminated_normally

    work_dir = Path(work_dir)
    results  = []

    for R, _ in casscf_results:
        cas_tag  = f'casscf_R{R:.2f}'.replace('.', 'p')
        gbw_file = (work_dir / cas_tag).with_suffix('.gbw').resolve()
        tag      = f'nevpt2_R{R:.2f}'.replace('.', 'p')
        out_file = work_dir / f'{tag}.out'

        if out_file.exists() and terminated_normally(out_file):
            E = get_nevpt2_energy(out_file)
            print(f'R = {R:.2f} A ... skipped (already done)  E = {E:.6f} Eh')
            results.append((R, E))
            continue

        geom = _extract_relaxed_geom(work_dir / f'{cas_tag}.out')
        if geom is None:
            xyz = work_dir / f'{cas_tag}.xyz'
            lines = xyz.read_text().splitlines()
            geom  = '\n'.join(lines[2:])

        inp = f"""! NEVPT2 {basis} TightSCF MORead

%maxcore 4000

%moinp "{gbw_file}"

%casscf
  nel {nel}
  norb {norb}
  mult {mult}
  nroots 1
end

* xyz 0 3
{geom}
*
"""
        print(f'R = {R:.2f} A ...', flush=True)
        outfile = run_orca(tag, inp, work_dir, nprocs=nprocs)
        E = get_nevpt2_energy(outfile)
        print(f'  E = {E:.6f} Eh', flush=True)
        results.append((R, E))

    print('All NEVPT2 done.')
    return results


def run_b3lyp_scan(casscf_results, work_dir, nprocs=1, basis='cc-pVDZ'):
    """
    Run B3LYP single points on CASSCF-relaxed geometries.

    Runs outward (small R first) with warm-starting via MORead for
    better convergence through the barrier region. Already-converged
    points are skipped on re-run.

    Parameters
    ----------
    casscf_results : list of (float, float)
        Output from run_casscf_scan: list of (R, E_casscf) pairs.
    work_dir : Path
        Working directory.
    nprocs : int
        Number of MPI processes.
    basis : str
        Basis set keyword.

    Returns
    -------
    list of (float, float)
        List of (R, E_b3lyp_Eh) pairs, sorted large R first.
    """
    from pathlib import Path
    from utils import run_orca, get_energy, terminated_normally

    work_dir = Path(work_dir)
    results  = []
    prev_gbw = None

    for R, _ in sorted(casscf_results, key=lambda x: x[0]):  # small R first
        cas_tag  = f'casscf_R{R:.2f}'.replace('.', 'p')
        tag      = f'b3lyp_R{R:.2f}'.replace('.', 'p')
        gbw_out  = (work_dir / tag).with_suffix('.gbw').resolve()
        out_file = work_dir / f'{tag}.out'

        if out_file.exists() and terminated_normally(out_file):
            E = get_energy(out_file)
            print(f'R = {R:.2f} A ... skipped (already done)  E = {E:.6f} Eh')
            results.append((R, E))
            if gbw_out.exists():
                prev_gbw = gbw_out
            continue

        geom = _extract_relaxed_geom(work_dir / f'{cas_tag}.out')
        if geom is None:
            xyz = work_dir / f'{cas_tag}.xyz'
            lines = xyz.read_text().splitlines()
            geom  = '\n'.join(lines[2:])

        moread = f'! MORead\n%moinp "{prev_gbw}"\n\n' if prev_gbw else ''

        inp = f"""{moread}! B3LYP {basis} TightSCF SlowConv

%maxcore 4000

%scf
  MaxIter 500
  STABPerform true
end

* xyz 0 3
{geom}
*
"""
        print(f'R = {R:.2f} A ...', flush=True)
        outfile = run_orca(tag, inp, work_dir, nprocs=nprocs)
        E = get_energy(outfile)
        print(f'  E = {E:.6f} Eh', flush=True)
        results.append((R, E))
        if gbw_out.exists():
            prev_gbw = gbw_out

    results.sort(key=lambda x: -x[0])  # large R first
    print(f'B3LYP done. Convergence failures: {sum(1 for _, e in results if e != e)}')
    return results
