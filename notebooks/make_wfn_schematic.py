"""
Generates a schematic figure showing how RHF enforces 50% ionic character
at dissociation, while CASSCF(2,2) correctly reaches 0% ionic.

Saves: h2_wfn_schematic.pdf and h2_wfn_schematic.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ── Colour palette ──────────────────────────────────────────────────────────
C_COVAL  = '#2980b9'   # blue  — covalent
C_IONIC  = '#c0392b'   # red   — ionic
C_CASSCF = '#27ae60'   # green — CASSCF
C_RHF    = '#e74c3c'   # red   — RHF label
C_BOX    = '#f8f9fa'   # light grey box fill
C_BORDER = '#dee2e6'   # box border
GRAY     = '#6c757d'

fig = plt.figure(figsize=(13, 7.5))

# ── Title ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         r'Why RHF enforces 50% ionic character at dissociation',
         ha='center', va='top', fontsize=14, fontweight='bold', color='#2c3e50')

# ═══════════════════════════════════════════════════════════════════════════
# LEFT PANEL — MO expansion of RHF
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.03, 0.08, 0.44, 0.84])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

def box(ax, x, y, w, h, text, fontsize=11, color=C_BOX, border=C_BORDER,
        textcolor='black', bold=False, alpha=1.0):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=border, linewidth=1.2, alpha=alpha,
        zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight=weight, zorder=4,
            linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, color='#495057', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=14))

def eq(ax, x, y, text, fontsize=13, color='#495057'):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=color, zorder=4)

# ── Section label ─────────────────────────────────────────────────────────
ax1.text(5, 9.7, 'RHF wavefunction — MO expansion',
         ha='center', va='center', fontsize=12, fontweight='bold',
         color=C_RHF)

# Row 1: RHF wavefunction
box(ax1, 0.3, 8.5, 9.4, 0.9,
    r'$\Psi_\mathrm{RHF} = |\sigma_g\,\bar\sigma_g\rangle$',
    fontsize=12, bold=True, color='#fff3cd', border='#f0ad4e')

# Arrow down
arrow(ax1, 5, 8.5, 5, 7.85)
ax1.text(5.3, 8.15, r'expand $\sigma_g = \frac{1}{\sqrt{2}}(1s_A + 1s_B)$',
         ha='center', va='center', fontsize=9.5, color=GRAY, style='italic')

# Row 2: Expanded form
box(ax1, 0.3, 7.0, 9.4, 0.75,
    r'$\Psi_\mathrm{RHF} \propto\; |1s_A\,\overline{1s_A}\rangle\;+\;|1s_B\,\overline{1s_B}\rangle\;+\;|1s_A\,\overline{1s_B}\rangle\;+\;|1s_B\,\overline{1s_A}\rangle$',
    fontsize=10.5, color=C_BOX)

# Arrow down
arrow(ax1, 5, 7.0, 5, 6.35)
ax1.text(5.3, 6.65, 'group by physical character',
         ha='center', va='center', fontsize=9.5, color=GRAY, style='italic')

# Row 3: Two groups
box(ax1, 0.3, 5.2, 4.2, 1.0,
    'Ionic\n'
    r'$|H^-\,H^+\rangle + |H^+\,H^-\rangle$',
    fontsize=10, color='#fdecea', border=C_IONIC, textcolor=C_IONIC)

box(ax1, 5.5, 5.2, 4.2, 1.0,
    'Covalent\n'
    r'$|H\!\cdot\quad\cdot\!H\rangle\times 2$',
    fontsize=10, color='#e8f4fd', border=C_COVAL, textcolor=C_COVAL)

ax1.text(5.0, 5.7, '+', ha='center', va='center', fontsize=16, color=GRAY)

# Weights
ax1.text(2.4, 4.9, '50%', ha='center', va='center',
         fontsize=13, fontweight='bold', color=C_IONIC)
ax1.text(7.6, 4.9, '50%', ha='center', va='center',
         fontsize=13, fontweight='bold', color=C_COVAL)

# Arrow down to punchline
arrow(ax1, 5, 4.85, 5, 4.1)
ax1.text(5.3, 4.5, r'at $r\to\infty$, should be 0% ionic',
         ha='center', va='center', fontsize=9.5, color=GRAY, style='italic')

# Punchline box
box(ax1, 0.3, 2.9, 9.4, 1.05,
    'RHF is STUCK at 50% ionic at dissociation\n'
    r'because the coefficient of $|\sigma_g\bar\sigma_g\rangle$ is fixed at 1.0',
    fontsize=10.5, color='#fdecea', border=C_IONIC, textcolor='#7b241c', bold=False)

# Correct answer
box(ax1, 0.3, 1.55, 9.4, 1.1,
    'Correct dissociation: H· + ·H\n'
    r'requires 0% ionic  →  $E = E_H + E_H$ exactly',
    fontsize=10.5, color='#e8f8f5', border=C_COVAL, textcolor='#1a5276')

# Energy scale bar at bottom
ax1.text(5.0, 0.8,
         'RHF dissociation error ≈ +3.8 eV  (overestimates bond energy)',
         ha='center', va='center', fontsize=9.5, color=C_RHF, style='italic')


# ═══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — CASSCF fix
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.52, 0.08, 0.46, 0.84])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(5, 9.7, 'CASSCF(2,2) — variational coefficients',
         ha='center', va='center', fontsize=12, fontweight='bold',
         color=C_CASSCF)

# CASSCF wavefunction
box(ax2, 0.2, 8.5, 9.6, 0.9,
    r'$\Psi_\mathrm{CASSCF} = c_1|\sigma_g\bar\sigma_g\rangle - c_2|\sigma_u\bar\sigma_u\rangle$',
    fontsize=12, bold=True, color='#d5f5e3', border=C_CASSCF)

# Arrow
arrow(ax2, 5, 8.5, 5, 7.85)
ax2.text(5.3, 8.15, r'expand both configurations in AO basis',
         ha='center', va='center', fontsize=9.5, color=GRAY, style='italic')

# Expanded CASSCF — showing cancellation
box(ax2, 0.2, 6.65, 9.6, 1.1,
    r'$\Psi_\mathrm{CASSCF} \propto\;(c_1-c_2)\,[|1s_A\bar{1s}_B\rangle+|1s_B\bar{1s}_A\rangle]$'
    '    [covalent]\n'
    r'$+\;(c_1+c_2)\,[|1s_A\bar{1s}_A\rangle+|1s_B\bar{1s}_B\rangle]$'
    '        [ionic]',
    fontsize=10, color=C_BOX)

# Arrow
arrow(ax2, 5, 6.65, 5, 5.95)

# Two geometry regimes
# ── Equilibrium ──
box(ax2, 0.2, 4.65, 4.5, 1.2,
    'At equilibrium\n'
    r'$c_1 \approx 0.99,\;c_2 \approx 0.11$'
    '\n'
    r'ionic weight $\approx 10\%$',
    fontsize=9.5, color='#fef9e7', border='#f39c12', textcolor='#7d6608')

# ── Dissociation ──
box(ax2, 5.3, 4.65, 4.5, 1.2,
    'At dissociation\n'
    r'$c_1 = c_2 = \frac{1}{\sqrt{2}}$'
    '\n'
    r'ionic weight $= 0\%$ ✓',
    fontsize=9.5, color='#e8f8f5', border=C_CASSCF, textcolor='#1d6a47')

ax2.text(5.0, 5.25, '→', ha='center', va='center', fontsize=18, color=GRAY)

# Arrow down
arrow(ax2, 5, 4.65, 5, 3.9)
ax2.text(5.3, 4.25, r'$(c_1+c_2)\to 0$ as $r\to\infty$',
         ha='center', va='center', fontsize=9.5, color=GRAY, style='italic')

# Ionic cancellation box
box(ax2, 0.2, 2.75, 9.6, 1.05,
    r'Ionic terms cancel exactly when $c_1 = c_2$'
    '\n'
    r'CASSCF optimises $c_1,c_2$ variationally at every geometry',
    fontsize=10.5, color='#d5f5e3', border=C_CASSCF, textcolor='#1a5276')

# Correct answer
box(ax2, 0.2, 1.55, 9.6, 1.1,
    'Correct dissociation: H· + ·H,  ⟨S²⟩ = 0  (pure singlet)\n'
    r'$E = E_H + E_H$ exactly  →  dissociation error = 0',
    fontsize=10.5, color='#e8f4fd', border=C_COVAL, textcolor='#1a5276')

ax2.text(5.0, 0.8,
         'CASSCF(2,2) dissociation error ≈ 0  (within basis set limit)',
         ha='center', va='center', fontsize=9.5, color=C_CASSCF, style='italic')

# ── Vertical divider ────────────────────────────────────────────────────────
fig.add_artist(
    plt.Line2D([0.505, 0.505], [0.06, 0.97],
               transform=fig.transFigure,
               color=C_BORDER, lw=1.5, linestyle='--'))

plt.savefig('h2_wfn_schematic.pdf', dpi=150, bbox_inches='tight')
plt.savefig('h2_wfn_schematic.png', dpi=150, bbox_inches='tight')
print("Saved h2_wfn_schematic.pdf and h2_wfn_schematic.png")
plt.show()
