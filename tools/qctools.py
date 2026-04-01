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
    geometry steps even with warm-starting. This function ensures visual
    continuity in orbital galleries by flipping the phase of ``cube_str``
    if its overlap integral with ``reference_cube_str`` is negative.

    Parameters
    ----------
    cube_str : str
        Cube file content (as returned by plot_orbital) to be phase-aligned.
    reference_cube_str : str
        Reference cube file content to align against. Typically the orbital
        at the first geometry in a sequence (e.g. R=2.2 Å).

    Returns
    -------
    str
        Cube file content with phase aligned to reference. If the overlap
        is already positive, the input is returned unchanged.

    Notes
    -----
    Uses the dot product of the volumetric data arrays as a proxy for the
    overlap integral. This is valid when both cubes share the same grid,
    which is guaranteed when both come from plot_orbital on the same system.

    Examples
    --------
    >>> cubes = []
    >>> for col, (tag, mo_idx) in enumerate(zip(tags_diag, mo_indices)):
    ...     cube = plot_orbital(tag, mo_index=mo_idx, work_dir=work_dir)
    ...     if col == 0:
    ...         ref_cube = cube
    ...     else:
    ...         cube = phase_align_cube(cube, ref_cube)
    ...     cubes.append(cube)
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

    # Truncate to same length in case of minor grid differences
    n = min(len(v1), len(v2))
    overlap = np.dot(v1[:n], v2[:n])

    if overlap >= 0:
        return cube_str  # phases already aligned

    # Flip phase: negate all volumetric data
    neg_values = -v1
    header = '\n'.join(lines1[:data_start1])
    data_lines = []
    for i in range(0, len(neg_values), 6):
        chunk = neg_values[i:i+6]
        data_lines.append('  '.join(f'{v:12.5E}' for v in chunk))
    return header + '\n' + '\n'.join(data_lines) + '\n'
