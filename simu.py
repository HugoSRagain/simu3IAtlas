import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import (Sun, Mercury, Venus, Earth, Mars,
                              Jupiter, Saturn, Uranus, Neptune)
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit

# =========================
# PARAMÈTRES GÉNÉRAUX
# =========================
epoch = Time("2025-07-17", scale="tdb")  # époque de référence
obs_times = [
    Time("2025-07-14T21:16:09", scale="tdb"),
    Time("2025-07-14T21:32:05", scale="tdb"),
    Time("2025-07-17T20:55:55", scale="tdb"),
    Time("2025-07-17T21:11:44", scale="tdb"),
]

# =========================
# ORBITE HYPERBOLIQUE 3I/ATLAS (VAL. EXEMPLE)
# =========================
a = -108.541 * u.AU         # demi-grand axe (négatif hyperbole)
e = 1.10350                 # excentricité
inc = np.deg2rad(171.2656)  # inclinaison
raan = np.deg2rad(298.5940) # Ω
argp = np.deg2rad(175.3664) # ω

# Domaine global d'anomalie vraie (pour figure 3D longue)
nu_full_deg = np.linspace(-160, 160, 800)
nu_full = np.deg2rad(nu_full_deg)

# -------------------------
# Fonctions rotation & hyperbole
# -------------------------
def Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def Rx(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

R = Rz(raan) @ Rx(inc) @ Rz(argp)

def hyperbola_xyz(a, e, nu):  # nu en radians (array)
    # Conversion via anomalie hyperbolique H
    fac = np.sqrt((e - 1)/(e + 1))
    tan_half_nu = np.tan(nu/2)
    tanh_half_H = fac * tan_half_nu
    # éviter dépassement
    tanh_half_H = np.clip(tanh_half_H, -0.999999999, 0.999999999)
    H = 2 * np.arctanh(tanh_half_H)
    a_val = a.to(u.AU).value
    r = a_val * (e * np.cosh(H) - 1)   # distance >0
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = np.zeros_like(x_orb)
    coords = R @ np.vstack((x_orb, y_orb, z_orb))
    return coords, r

# Trajectoire hyperbolique globale
coords_full, r_full = hyperbola_xyz(a, e, nu_full)
X3I, Y3I, Z3I = coords_full

# =========================
# POSITIONS APPROX OBJET AUX OBSERVATIONS (méthode fractionnelle)
# =========================
def approx_position_from_fraction(frac):
    idx = int((frac+1)/2 * (len(nu_full)-1))
    return X3I[idx], Y3I[idx], Z3I[idx]

frac_times = [-0.4, -0.35, 0.05, 0.1]  # heuristique
obs_positions = [approx_position_from_fraction(f) for f in frac_times]

# =========================
# ORBITES PLANÉTAIRES (par éléments osculateurs à epoch)
# =========================
planets = [Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]
names   = ["Mercure","Vénus","Terre","Mars","Jupiter","Saturne","Uranus","Neptune"]
colors  = ["gray","orange","deepskyblue","red","purple","saddlebrown","teal","navy"]

planet_orbits_coords = []
planet_positions_epoch = []
N_ellipse = 800
nu_plan = np.linspace(0, 2*np.pi, N_ellipse)

for body in planets:
    eph = Ephem.from_body(body, epoch)
    orb = Orbit.from_ephem(Sun, eph, epoch)
    a_p   = orb.a.to(u.AU).value
    e_p   = orb.ecc.value
    inc_p = orb.inc.to(u.rad).value
    raan_p= orb.raan.to(u.rad).value
    argp_p= orb.argp.to(u.rad).value

    # ellipse dans son plan
    r_p = a_p * (1 - e_p**2) / (1 + e_p * np.cos(nu_plan))
    x_p = r_p * np.cos(nu_plan)
    y_p = r_p * np.sin(nu_plan)
    z_p = np.zeros_like(x_p)
    Rp = Rz(raan_p) @ Rx(inc_p) @ Rz(argp_p)
    xyz_p = Rp @ np.vstack((x_p, y_p, z_p))
    planet_orbits_coords.append(xyz_p)

    pos = orb.r.to(u.AU).value
    planet_positions_epoch.append(pos)

# =========================
# FIGURE PRINCIPALE 3D (optionnelle)
# =========================
def plot_main_3d(limit=15):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Hyperbole
    ax.plot(X3I, Y3I, Z3I, color='crimson', lw=2, label='3I/ATLAS (hyperbolique)')
    # Points observation
    for (xp, yp, zp), t in zip(obs_positions, obs_times):
        ax.scatter(xp, yp, zp, color='black', marker='x', s=60)
        ax.text(xp, yp, zp, t.iso.split('T')[0], fontsize=8, color='black')
    # Orbites planétaires
    for (xyz_p, pos_p, name, col) in zip(planet_orbits_coords, planet_positions_epoch, names, colors):
        ax.plot(xyz_p[0], xyz_p[1], xyz_p[2], color=col, lw=1, alpha=0.9)
        ax.scatter(pos_p[0], pos_p[1], pos_p[2], color=col, s=40)
        ax.text(pos_p[0], pos_p[1], pos_p[2], name, fontsize=8, color=col)
    # Soleil
    ax.scatter(0,0,0, color='gold', s=140, edgecolors='k', label='Soleil')
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    ax.set_title("Trajectoire hyperbolique de 3I/ATLAS et orbites planétaires\n(époque 2025-07-17)")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit*0.6, limit*0.6)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================
# PANNEAUX MULTI-ÉCHELLES (géométriques)
# =========================
def plot_three_scales(mode='2d'):
    """
    mode = '2d' : projection sur plan X-Y
    mode = '3d' : 3 sous-graphiques 3D (plus lourd visuellement)
    """
    ranges = {
        "zoom_peri": (-5, 5),
        "mid":       (-40, 40),
        "wide":      (-170, 170)
    }
    titles = {
        "zoom_peri": "Zoom péricentre (|ν| ≤ 5°)",
        "mid":       "Zone intermédiaire (|ν| ≤ 40°)",
        "wide":      "Vue large (|ν| ≤ 170°)"
    }

    if mode == '2d':
        fig, axes = plt.subplots(1, 3, figsize=(17,5))
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(18,5))
        axes = [fig.add_subplot(1,3,i+1, projection='3d') for i in range(3)]

    keys = list(ranges.keys())
    for ax, key in zip(axes, keys):
        nu_min, nu_max = ranges[key]
        nu_sub_deg = np.linspace(nu_min, nu_max, 600)
        nu_sub = np.deg2rad(nu_sub_deg)
        coords_sub, r_sub = hyperbola_xyz(a, e, nu_sub)
        Xs, Ys, Zs = coords_sub

        # Orbites planétaires complètes
        for (xyz_p, name, col) in zip(planet_orbits_coords, names, colors):
            if mode == '2d':
                ax.plot(xyz_p[0], xyz_p[1], color=col, lw=0.7, alpha=0.7)
            else:
                ax.plot(xyz_p[0], xyz_p[1], xyz_p[2], color=col, lw=0.8, alpha=0.7)

        # Trajectoire hyperbolique portion
        if mode == '2d':
            ax.plot(Xs, Ys, color='black', lw=2, label='3I/ATLAS')
            # Périhélie (nu=0) si dans range
            if nu_min < 0 < nu_max:
                peri_coord, _ = hyperbola_xyz(a, e, np.array([0.0]))
                ax.scatter(peri_coord[0,0], peri_coord[1,0], color='gold', s=60, edgecolors='k', zorder=5)
        else:
            ax.plot(Xs, Ys, Zs, color='black', lw=2, label='3I/ATLAS')
            if nu_min < 0 < nu_max:
                peri_coord, _ = hyperbola_xyz(a, e, np.array([0.0]))
                ax.scatter(peri_coord[0,0], peri_coord[1,0], peri_coord[2,0], color='gold', s=60, edgecolors='k')

        # Soleil
        if mode == '2d':
            ax.scatter(0,0, color='orange', s=70, edgecolors='k')
        else:
            ax.scatter(0,0,0, color='orange', s=90, edgecolors='k')

        ax.set_title(titles[key])
        if mode == '2d':
            ax.set_aspect('equal','box')
            ax.set_xlabel("X (UA)")
            ax.set_ylabel("Y (UA)")
        else:
            ax.set_xlabel("X (UA)")
            ax.set_ylabel("Y (UA)")
            ax.set_zlabel("Z (UA)")
            # Limites adaptatives
            span = np.max(np.sqrt(Xs**2 + Ys**2))
            m = min(12, span*1.2)
            ax.set_xlim(-m, m)
            ax.set_ylim(-m, m)
            ax.set_zlim(-m*0.6, m*0.6)
            ax.view_init(elev=25, azim=130)

        # Limites spécifiques pour lisibilité (2D)
        if mode == '2d':
            if key == 'zoom_peri':
                ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
            elif key == 'mid':
                ax.set_xlim(-10,10); ax.set_ylim(-10,10)
            else:
                ax.set_xlim(-25,25); ax.set_ylim(-25,25)

        if key == 'zoom_peri':
            ax.legend(loc='upper right', fontsize=7, frameon=False)

        ax.grid(alpha=0.25, ls=':')

    fig.suptitle("3I/ATLAS – Trois échelles (portions d'hyperbole)", fontsize=14)
    plt.tight_layout()
    plt.show()

# =========================
# EXÉCUTION
# =========================
if __name__ == "__main__":
    # Figure principale (si tu veux la garder)
    plot_main_3d(limit=15)

    # Figure multi-échelles (choisis '2d' ou '3d')
    plot_three_scales(mode='2d')
