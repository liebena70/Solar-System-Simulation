import numpy as np
from vpython import (canvas, sphere, vector, color, label, rate, button,
                     slider, wtext, distant_light, local_light, points)

# ============================================================
# 1. UNITS
# ============================================================
GM_SUN = 4.0 * np.pi ** 2                      # AU^3 / yr^2
SEC_PER_YEAR = 365.25 * 24 * 3600
AU_KM = 149_597_870.7
C_LIGHT = 299_792.458 * SEC_PER_YEAR / AU_KM   # speed of light, AU/yr
C2 = C_LIGHT ** 2

DEG = np.pi / 180.0

# ============================================================
# 2. PLANETARY DATA (NASA fact-sheet, J2000 mean ecliptic)
#    a    : semi-major axis (AU)
#    e    : eccentricity
#    inc  : inclination to ecliptic (deg)
#    Omega: longitude of ascending node (deg)
#    omega: argument of perihelion (deg)
#    mass : in solar masses
# ============================================================
PLANETS = [
    # name        a            e        inc      Omega     omega     mass(M_sun)    colour                    rvis
    ("Mercury",  0.387098,  0.205630,   7.005,   48.331,   29.124,   1.66014e-7,   vector(0.70,0.70,0.70),  0.20),
    ("Venus",    0.723332,  0.006772,   3.395,   76.680,   54.884,   2.44783e-6,   vector(0.95,0.78,0.40),  0.30),
    ("Earth",    1.000000,  0.016709,   0.000,  -11.260,  114.208,   3.00348e-6,   vector(0.20,0.55,1.00),  0.32),
    ("Mars",     1.523679,  0.093400,   1.850,   49.558,  286.502,   3.22715e-7,   vector(0.85,0.35,0.20),  0.26),
    ("Jupiter",  5.204267,  0.048775,   1.303,  100.464,  273.867,   9.54792e-4,   vector(0.90,0.75,0.55),  0.90),
    ("Saturn",   9.582017,  0.055723,   2.485,  113.665,  339.392,   2.85886e-4,   vector(0.95,0.85,0.60),  0.80),
    ("Uranus",  19.229411,  0.044405,   0.773,   74.006,   96.998,   4.36624e-5,   vector(0.55,0.85,0.90),  0.55),
    ("Neptune", 30.103660,  0.011214,   1.770,  131.784,  276.336,   5.15139e-5,   vector(0.30,0.45,0.95),  0.55),
]

# ============================================================
# 3. DISPLAY SCALING
#    Real distances span 0.39 to 50+ AU. A log stretch keeps inner
#    planets visible while showing the Kuiper belt on the same canvas.
# ============================================================
DISTANCE_SCALE_MODE = "log"   # "linear" or "log"

def scale_factor(au_norm):
    """Return display-units / AU for a given true AU radius."""
    if DISTANCE_SCALE_MODE == "linear":
        return 1.0
    # f(a) / a where f(a) = 4 ln(1 + a/0.1)
    return 4.0 * np.log(1.0 + au_norm / 0.1) / au_norm

def scale_vec3(r_vec):
    """Apply the scaling to a 3D numpy vector, preserving direction."""
    n = np.linalg.norm(r_vec)
    if n == 0.0:
        return vector(0, 0, 0)
    s = scale_factor(n)
    return vector(r_vec[0]*s, r_vec[1]*s, r_vec[2]*s)

# ============================================================
# 4. ORBITAL-ELEMENT to STATE VECTOR
#    Start each body at perihelion. State (r,v) in heliocentric
#    inertial frame is obtained by rotating the in-plane perihelion
#    state by  R_z(Ω) · R_x(i) · R_z(ω).
# ============================================================
def state_from_elements(a, e, inc_deg, Om_deg, w_deg, nu_deg=0.0,
                        mu=GM_SUN):
    """
    Returns (r, v) numpy arrays (AU, AU/yr) at true anomaly nu (default 0
    = perihelion).
    """
    i  = inc_deg * DEG
    Om = Om_deg  * DEG
    w  = w_deg   * DEG
    nu = nu_deg  * DEG

    # In perifocal frame (P points to perihelion, Q to +90° in orbit plane)
    p = a * (1 - e * e)
    r_mag = p / (1 + e * np.cos(nu))
    r_pf = np.array([r_mag * np.cos(nu),
                     r_mag * np.sin(nu),
                     0.0])
    factor = np.sqrt(mu / p)
    v_pf = np.array([-factor * np.sin(nu),
                      factor * (e + np.cos(nu)),
                      0.0])

    # Rotation: R = Rz(Om) · Rx(i) · Rz(w)
    cO, sO = np.cos(Om), np.sin(Om)
    ci, si = np.cos(i),  np.sin(i)
    cw, sw = np.cos(w),  np.sin(w)
    R = np.array([
        [cO*cw - sO*sw*ci,  -cO*sw - sO*cw*ci,   sO*si],
        [sO*cw + cO*sw*ci,  -sO*sw + cO*cw*ci,  -cO*si],
        [sw*si,              cw*si,              ci   ],
    ])
    return R @ r_pf, R @ v_pf

# ============================================================
# 5. SCENE
# ============================================================
scene = canvas(title="<b>Solar System Simulator</b> (N-body + Einstein GR + Asteroid &amp; Kuiper belts)",
               width=1280, height=720,
               background=vector(0.02, 0.02, 0.05),
               forward=vector(0, -0.45, -1), up=vector(0, 1, 0))
scene.ambient = vector(0.15, 0.15, 0.2)
scene.lights = []
distant_light(direction=vector(0, 0.3, 1), color=vector(0.2, 0.2, 0.25))
local_light(pos=vector(0, 0, 0), color=vector(1, 0.95, 0.8))
scene.range = 35

# Sun
sun = sphere(pos=vector(0, 0, 0), radius=1.2,
             color=vector(1.0, 0.85, 0.2), emissive=True)
label(pos=sun.pos, text="Sun", xoffset=10, yoffset=20, height=12,
      color=vector(1, 0.9, 0.4), box=False, opacity=0, font='sans')

# ============================================================
# 6. PLANETS
# ============================================================
bodies = []
for name, a, e, inc, Om, om, mass, col, rvis in PLANETS:
    r0, v0 = state_from_elements(a, e, inc, Om, om)
    sph = sphere(pos=scale_vec3(r0), radius=rvis, color=col,
                 make_trail=True, trail_type="curve",
                 trail_radius=rvis * 0.05, retain=4000,
                 interval=1, shininess=0.3)
    lab = label(pos=sph.pos, text=name, xoffset=12, yoffset=12, height=11,
                color=col, box=False, opacity=0, font='sans')
    bodies.append({
        "name": name, "a": a, "e": e, "mass": mass,
        "r": r0.astype(float), "v": v0.astype(float),
        "acc": np.zeros(3),
        "sphere": sph, "label": lab,
    })

GM_PLANETS = np.array([b["mass"] for b in bodies]) * GM_SUN

# ============================================================
# 7. ASTEROID BELT + KUIPER BELT  (test particles) doesn't have gravity effect to another celestial object
# ============================================================
rng = np.random.default_rng(42)

def make_belt(n, a_lo, a_hi, e_max, inc_max_deg):
    """
    Returns (R, V) for n test particles randomly populating an orbital-element shell.
    """
    a   = rng.uniform(a_lo, a_hi, n)
    e   = rng.uniform(0.0, e_max, n)
    inc = rng.uniform(-inc_max_deg, inc_max_deg, n)
    Om  = rng.uniform(0.0, 360.0, n)
    om  = rng.uniform(0.0, 360.0, n)
    nu  = rng.uniform(0.0, 360.0, n)  # spread along orbit
    R = np.zeros((n, 3))
    V = np.zeros((n, 3))
    for k in range(n):
        r, v = state_from_elements(a[k], e[k], inc[k], Om[k], om[k], nu[k])
        R[k] = r
        V[k] = v
    return R, V

N_ASTEROIDS = 350
N_KUIPER    = 350
R_ast, V_ast = make_belt(N_ASTEROIDS, 2.1, 3.3, 0.20, 15.0)
R_kui, V_kui = make_belt(N_KUIPER,   30.0, 50.0, 0.20, 25.0)

def belt_initial_points(R):
    return [scale_vec3(r) for r in R]

ast_points = points(pos=belt_initial_points(R_ast),
                    radius=2.5, color=vector(0.75, 0.65, 0.45))
kui_points = points(pos=belt_initial_points(R_kui),
                    radius=2.5, color=vector(0.55, 0.75, 0.95))

# ============================================================
# 8. PHYSICS
# ============================================================
def planet_accelerations(R, V):
    """N-body (Sun + GR + planet-planet) acceleration for the planets."""
    r2 = np.einsum('ij,ij->i', R, R)
    r  = np.sqrt(r2)
    L  = np.cross(R, V)
    L2 = np.einsum('ij,ij->i', L, L)
    gr = 1.0 + 3.0 * L2 / (C2 * r2)
    A  = -(GM_SUN * gr / (r2 * r))[:, None] * R                 # Sun + GR
    diff = R[None, :, :] - R[:, None, :]
    d2   = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    inv  = 1.0 / (d2 * np.sqrt(d2))
    np.fill_diagonal(inv, 0.0)
    A += np.einsum('j,ij,ijk->ik', GM_PLANETS, inv, diff)
    return A

def test_accelerations(R):
    """Sun-only Newtonian acceleration for many test particles (vectorised)."""
    r2 = np.einsum('ij,ij->i', R, R)
    r  = np.sqrt(r2)
    return -(GM_SUN / (r2 * r))[:, None] * R

# Initial accelerations
R_pl = np.array([b["r"] for b in bodies])
V_pl = np.array([b["v"] for b in bodies])
A_pl = planet_accelerations(R_pl, V_pl)
A_ast = test_accelerations(R_ast)
A_kui = test_accelerations(R_kui)

# ============================================================
# 9. UI CONTROLS
# ============================================================
running     = {"on": True}
show_trails = {"on": True}
show_ast    = {"on": True}
show_kui    = {"on": True}

def toggle_pause(btn):
    running["on"] = not running["on"]
    btn.text = "Resume" if not running["on"] else "Pause"

def toggle_trails(btn):
    show_trails["on"] = not show_trails["on"]
    for b in bodies:
        b["sphere"].make_trail = show_trails["on"]
        if not show_trails["on"]:
            b["sphere"].clear_trail()
    btn.text = "Show Paths" if not show_trails["on"] else "Hide Paths"

def toggle_asteroids(btn):
    show_ast["on"] = not show_ast["on"]
    ast_points.visible = show_ast["on"]
    btn.text = "Show Asteroids" if not show_ast["on"] else "Hide Asteroids"

def toggle_kuiper(btn):
    show_kui["on"] = not show_kui["on"]
    kui_points.visible = show_kui["on"]
    btn.text = "Show Kuiper" if not show_kui["on"] else "Hide Kuiper"

def reset_view(btn):
    scene.center = vector(0, 0, 0)
    scene.range = 35
    scene.forward = vector(0, -0.45, -1)
    scene.up = vector(0, 1, 0)

scene.append_to_caption("\n")
button(text="Pause",          bind=toggle_pause);     scene.append_to_caption("  ")
button(text="Hide Paths",     bind=toggle_trails);    scene.append_to_caption("  ")
button(text="Hide Asteroids", bind=toggle_asteroids); scene.append_to_caption("  ")
button(text="Hide Kuiper",    bind=toggle_kuiper);    scene.append_to_caption("  ")
button(text="Reset View",     bind=reset_view);       scene.append_to_caption("\n\n")

ratio_state = {"days_per_sec": 10.0}
def set_ratio(s):
    ratio_state["days_per_sec"] = s.value
    ratio_text.text = (f"  Time ratio: 1 real second = "
                       f"<b>{s.value:.2f}</b> simulated days\n")

scene.append_to_caption("Simulation speed (sim days per real second):  ")
slider(min=0.1, max=5000.0, value=10.0, length=420, bind=set_ratio, right=15)
ratio_text = wtext(text=f"  Time ratio: 1 real second = "
                       f"<b>{ratio_state['days_per_sec']:.2f}</b> simulated days\n")
scene.append_to_caption("\n")
info_text = wtext(text="")
scene.append_to_caption("\n\n<i>Mouse wheel = zoom &nbsp;|&nbsp; right-drag = rotate &nbsp;|&nbsp; shift-drag = pan</i>\n")

# ============================================================
# 10. MAIN LOOP – velocity Verlet
# ============================================================
sim_time_years = 0.0
FPS = 60
SUB_STEPS_MAX = 4000
DT_BASE = 1.0 / 365.25 / 4.0     # 6-hour base step (inner-planet stable)

while True:
    rate(FPS)

    if running["on"]:
        sim_years_this_frame = (ratio_state["days_per_sec"] / FPS) / 365.25
        n_steps = max(1, int(np.ceil(sim_years_this_frame / DT_BASE)))
        n_steps = min(n_steps, SUB_STEPS_MAX)
        dt = sim_years_this_frame / n_steps

        for _ in range(n_steps):
            # --- planets ---
            R_new = R_pl + V_pl * dt + 0.5 * A_pl * dt * dt
            A_new = planet_accelerations(R_new, V_pl + 0.5 * A_pl * dt)
            V_pl  = V_pl + 0.5 * (A_pl + A_new) * dt
            R_pl, A_pl = R_new, A_new

            # --- asteroid belt (test particles) ---
            R_ast_new = R_ast + V_ast * dt + 0.5 * A_ast * dt * dt
            A_ast_new = test_accelerations(R_ast_new)
            V_ast = V_ast + 0.5 * (A_ast + A_ast_new) * dt
            R_ast, A_ast = R_ast_new, A_ast_new

            # --- Kuiper belt (test particles) ---
            R_kui_new = R_kui + V_kui * dt + 0.5 * A_kui * dt * dt
            A_kui_new = test_accelerations(R_kui_new)
            V_kui = V_kui + 0.5 * (A_kui + A_kui_new) * dt
            R_kui, A_kui = R_kui_new, A_kui_new

        sim_time_years += dt * n_steps

        # Write planet state back into body dicts (needed nowhere else,
        # but kept for clarity / future use)
        for i, b in enumerate(bodies):
            b["r"], b["v"], b["acc"] = R_pl[i], V_pl[i], A_pl[i]

    # --- update visuals ---
    for i, b in enumerate(bodies):
        p = scale_vec3(R_pl[i])
        b["sphere"].pos = p
        b["label"].pos  = p

    if show_ast["on"]:
        ast_points.clear()
        ast_points.append(pos=[scale_vec3(r) for r in R_ast])
    if show_kui["on"]:
        kui_points.clear()
        kui_points.append(pos=[scale_vec3(r) for r in R_kui])

    # HUD
    years = sim_time_years
    days  = years * 365.25
    info_text.text = (
        f"<b>Elapsed simulated time:</b> {years:,.4f} yr ({days:,.2f} days)\n"
        f"<b>Speed:</b> 1 real second = {ratio_state['days_per_sec']:.2f} simulated days "
        f"(= {ratio_state['days_per_sec']/365.25:.5f} sim years / sec)\n"
        f"<b>Integrator:</b> velocity-Verlet | "
        f"Planets: N-body + GR (Schwarzschild) | "
        f"Belts: {N_ASTEROIDS} asteroids + {N_KUIPER} Kuiper objects (test particles)\n"
    )

