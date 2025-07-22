import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
import plotly.graph_objects as go
import os

# -- Paramètres
epoch = Time("2025-07-17", scale="tdb")
obs_times = [
    Time("2025-07-14T21:16:09", scale="tdb"),
    Time("2025-07-14T21:32:05", scale="tdb"),
    Time("2025-07-17T20:55:55", scale="tdb"),
    Time("2025-07-17T21:11:44", scale="tdb"),
]

a = -108.541 * u.AU
e = 1.10350
inc = np.deg2rad(171.2656)
raan = np.deg2rad(298.5940)
argp = np.deg2rad(175.3664)

nu_full_deg = np.linspace(-160, 160, 800)
nu_full = np.deg2rad(nu_full_deg)

def Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def Rx(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

R = Rz(raan) @ Rx(inc) @ Rz(argp)

def hyperbola_xyz(a, e, nu):
    fac = np.sqrt((e - 1)/(e + 1))
    tan_half_nu = np.tan(nu/2)
    tanh_half_H = fac * tan_half_nu
    tanh_half_H = np.clip(tanh_half_H, -0.999999999, 0.999999999)
    H = 2 * np.arctanh(tanh_half_H)
    a_val = a.to(u.AU).value
    r = a_val * (e * np.cosh(H) - 1)
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = np.zeros_like(x_orb)
    coords = R @ np.vstack((x_orb, y_orb, z_orb))
    return coords, r

coords_full, r_full = hyperbola_xyz(a, e, nu_full)
X3I, Y3I, Z3I = coords_full

def approx_position_from_fraction(frac):
    idx = int((frac+1)/2 * (len(nu_full)-1))
    return X3I[idx], Y3I[idx], Z3I[idx]

frac_times = [-0.4, -0.35, 0.05, 0.1]
obs_positions = [approx_position_from_fraction(f) for f in frac_times]

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
    a_p = orb.a.to(u.AU).value
    e_p = orb.ecc.value
    inc_p = orb.inc.to(u.rad).value
    raan_p = orb.raan.to(u.rad).value
    argp_p = orb.argp.to(u.rad).value

    r_p = a_p * (1 - e_p**2) / (1 + e_p * np.cos(nu_plan))
    x_p = r_p * np.cos(nu_plan)
    y_p = r_p * np.sin(nu_plan)
    z_p = np.zeros_like(x_p)
    Rp = Rz(raan_p) @ Rx(inc_p) @ Rz(argp_p)
    xyz_p = Rp @ np.vstack((x_p, y_p, z_p))
    planet_orbits_coords.append(xyz_p)

    pos = orb.r.to(u.AU).value
    planet_positions_epoch.append(pos)

# -- Création figure Plotly
fig = go.Figure()

# Hyperbole 3I/ATLAS
fig.add_trace(go.Scatter3d(
    x=X3I, y=Y3I, z=Z3I,
    mode='lines',
    line=dict(color='crimson', width=4),
    name='3I/ATLAS (hyperbolique)'
))

# Observations
for (xp, yp, zp), t in zip(obs_positions, obs_times):
    fig.add_trace(go.Scatter3d(
        x=[xp], y=[yp], z=[zp],
        mode='markers+text',
        marker=dict(color='black', size=5, symbol='x'),
        text=[t.iso.split("T")[0]],
        textposition="top center",
        name='Observation'
    ))

# Orbites planétaires
for (xyz_p, pos_p, name, col) in zip(planet_orbits_coords, planet_positions_epoch, names, colors):
    fig.add_trace(go.Scatter3d(
        x=xyz_p[0], y=xyz_p[1], z=xyz_p[2],
        mode='lines',
        line=dict(color=col, width=1),
        name=f'Orbite de {name}'
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos_p[0]], y=[pos_p[1]], z=[pos_p[2]],
        mode='markers+text',
        marker=dict(size=4, color=col),
        text=[name],
        textposition="top center",
        name=name
    ))

# Soleil
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=10, color='gold', line=dict(color='black', width=1)),
    name="Soleil"
))

fig.update_layout(
    scene=dict(
        xaxis_title="X (UA)",
        yaxis_title="Y (UA)",
        zaxis_title="Z (UA)",
        xaxis=dict(range=[-15, 15]),
        yaxis=dict(range=[-15, 15]),
        zaxis=dict(range=[-10, 10]),
    ),
    title="Trajectoire hyperbolique de 3I/ATLAS et orbites planétaires<br>(époque 2025-07-17)",
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=True
)

# Export HTML
save_path = os.path.join(os.path.expanduser("~"), "3iatlas_orbit.html")
fig.write_html(save_path)
print(f"✅ Fichier HTML enregistré ici : {save_path}")
