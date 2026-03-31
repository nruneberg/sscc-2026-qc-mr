# decisions.md — SSCC 2026 Multireference QC course
Design decisions, technical choices, and pinned items for future work.
Add a dated entry after each significant session.

---

## General workflow

- One canonical notebook per topic, named `NN_descriptive_name.ipynb`
- Scratch versions live in `notebooks/scratch/` — not committed unless useful
- Commit after every session, even WIP, with a descriptive message
- `decisions.md` + end-of-session commit = lowest-friction context capture
- Tool split: Copilot M365 for boilerplate and Office tasks;
  Claude for scientific reasoning, CASSCF/NEVPT2/ORCA, notebook pedagogy

---

## NB01 — H2 single-reference fails

- NOON diagnostic scan works; `get_no_occupations()` returns all MO occupations
  (not just active) — slice `[-n_active:]` to get active orbitals only

---

## NB03 — O(³P) + ethylene: why transition states are multireference

### Session 2026-03-31

#### Calculation strategy
- Single CASSCF constrained-opt scan with **non-uniform R grid** (15 points)
  via `{B 0 2 R C}` in `%geom Constraints` block — gives full control over
  point spacing without needing two scan jobs or a merge step
- Grid: coarse in flat tail (3.5–2.9 Å), medium in approach (2.6–2.2 Å),
  fine through barrier and product (2.05–1.22 Å)
- Warm-started **inward** (large R → small R): each step passes `.gbw` and
  relaxed geometry forward via `MORead` — mirrors what ORCA does internally
- `MaxIter 200` for convergence robustness near barrier
- Progress reporting: use two separate `print()` calls, not `end=' '`
  (JupyterLab buffers incomplete lines from blocking calls)

- B3LYP single points run **outward** (small R → large R) with warm-starting
  On triplet surface ORCA uses ROB3LYP by default (not UB3LYP)
  Forcing `! UKS` gives identical results — the convergence failures are
  physical, not a technical limitation of the restricted framework
  B3LYP gaps in the barrier region are the message: single-reference breaks
  down exactly where multiconfigurational character is strongest

- All three methods normalised to their own largest-R point:
  CASSCF: `scan_dat[0, 1]`; NEVPT2/B3LYP: `energies[0]`
  (scan_xyzs sorted large R first throughout)

#### Section 3 structure (PENDING implementation)
- Split into **3a** (motivation from chemical intuition, before calculations)
  and **3b** (verification from NOs, after diagnostic single points)
- 3a: ~4 sentences motivating (6,6) from electron counting alone —
  two O p SOMOs, π/π* of C=C, σ/σ* for forming bond
- 3b: full orbital evolution table + 3-point mechanistic narrative + gallery
- Apply section 3 edits via Python JSON API, not str_replace
  (repeated str_replace introduced literal newlines inside JSON strings)

#### Orbital gallery — known pitfall (CRITICAL)
- NO pairing across geometries MUST use overlap-based matching, not MO index
- NOs are sorted by occupation not character — index-based pairing fails
  when occupations reorder between geometries (confirmed in this session
  by expert review; the error was not caught by visual inspection alone)
- Even for O(³P) + C₂H₄ — a textbook-simple reaction — getting the orbital
  assignments right required careful validation against raw ORCA output.
  For larger or less symmetric systems, chemical intuition alone is
  insufficient and semi-automatic methods become essential.

#### Overlap matcher — IMPLEMENT IN qctools.py
```python
def match_orbitals(gbw_file1, gbw_file2, n_active=6, work_dir=None):
    """
    Match active NOs between two geometries by maximum overlap.
    Returns permutation array: matched_indices[i] is the MO index
    in gbw_file2 corresponding to active MO i in gbw_file1.
    Uses orca_2mkl or OPI to extract MO coefficients,
    scipy.optimize.linear_sum_assignment (Hungarian algorithm)
    on the 6x6 overlap matrix.
    """
```

#### Section 3 future additions (collapsible details)
- Validation table: CASSCF(8,8) vs (6,6) NOON comparison
  Confirm additional orbitals stay near 0 or 2 — if not, expand active space
- Paragraph on semi-automatic active space selection:
  - AVAS (Atomic Valence Active Space) — available in ORCA 6.x
    Check exact syntax: `%casscf AVAS true end` + reference AO specification
  - Closing sentence: "Even for this relatively simple system, the orbital
    analysis required careful validation against raw ORCA output. For larger
    or less symmetric systems, semi-automatic methods such as AVAS become
    not just convenient but essential."
  - Also mention: DMRG-based active space selection for large systems

#### Orbital assignments (confirmed from Loewdin analysis)
At R=2.2 Å (MO index → character):
- MO 9:  O py, occ ≈ 2.000 — doubly occupied lone pair (becomes σ at R=1.5)
- MO 10: C1+C2 pz symmetric, occ ≈ 1.907 — π(C=C)
- MO 11: O px/pz, occ ≈ 1.000 — p(O, in-plane) SOMO
- MO 12: O pz/px mix, occ ≈ 0.999 — σ/p(O) mix SOMO
- MO 13: C1+C2 pz antisymmetric, occ ≈ 0.092 — π*(C=C)
- MO 14: C-O pz mix, occ ≈ 0.003 — σ*(O–C)

At R=1.5 Å (MO index → character):
- MO 9:  O px, occ ≈ 1.997 — σ(O–C) bond (formed)
- MO 10: C1 py, occ ≈ 1.988 — π/O mixed bonding (3-centre interaction)
- MO 11: C2 pz, occ ≈ 1.000 — C2 radical SOMO
- MO 12: O py, occ ≈ 1.000 — persistent O-centred SOMO
- MO 13: C1 py, occ ≈ 0.010 — near-doubly-occupied (NO reordering of π*)
- MO 14: C1+O pz mix, occ ≈ 0.003 — σ*(O–C)

Note: MO 13 at R=1.5 Å maps to an O lone pair by index, NOT to π*.
The true π* character is distributed — overlap-based matching needed.

#### PES results (v4/v5 scan, 15-point non-uniform grid)
- B3LYP: barrierless, attractive throughout; gaps at R≈2.9–2.4 Å
- CASSCF(6,6): barrier ≈ 68 kJ/mol at R≈1.9 Å
- CASSCF + NEVPT2: barrier ≈ 35 kJ/mol
- Local minimum at R≈1.4 Å visible on B3LYP and NEVPT2 curves

#### Files
- Canonical notebook: `notebooks/03_oxygen_ethylene_reaction.ipynb`
- Figures: `notebooks/figures/o3p_ethylene_scheme.svg` (from Antti)
- Old scheme: `notebooks/scratch/reaction_scheme_nb03.svg` (superseded)
- v3–v5 intermediates: `notebooks/scratch/` (keep for reference)
