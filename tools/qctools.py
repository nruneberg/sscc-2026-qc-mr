# tools/qctools.py
# Static copy of Antti Karttunen's qctools.py
# Original: https://github.com/anttikarttunen/sscc2026-qc/blob/main/tools/qctools.py
# Last synced: 2026-03-17
# Contact: antti.karttunen@aalto.fi

############## Functions to facilitate printing ##############

def print_info(info):
    banner = "-----------------------------------------------------------"
    print(f"{banner}\n{info}\n{banner}")

def print_error(error):
    err = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    print(f"{err}\n{error}\n{err}")

############## Functions for loading molecules ##############

def load_xyz(xyzfile, silent=False):
    """Load an XYZ file. Returns ase.Atoms or None."""
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
    """Convert multi-XYZ file to ASE trajectory. Returns Trajectory or None."""
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

############## Functions for visualizing molecules with nglview ##############

def show_molecule(molecule, size=(500, 400), style='ball+stick',
                  unitcell=None, labels=None, bg='black'):
    """Show a molecule using NGLWidget.

    Parameters
    ----------
    molecule : ase.Atoms or ase.io.Trajectory
    size : tuple of (width, height) in pixels
    style : nglview representation style
    unitcell : None or color string
    labels : None or 'atomname', 'atomindex', 'element'
    bg : background color string

    Returns
    -------
    NGLWidget
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
