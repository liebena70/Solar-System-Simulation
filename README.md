N-Body Solar System Simulation with Numerical Integration and Relativistic Corrections

A computational physics simulation of the Solar System using Newtonian N-body gravitational dynamics. The system models planetary motion, asteroid belt, and Kuiper belt objects using numerical integration and includes simplified relativistic corrections for improved orbital accuracy.

🎯 Project Objective

The goal of this project is to simulate the dynamics of the Solar System using computational methods, with emphasis on:

Numerical modeling of gravitational N-body systems
Stable long-term orbital integration
Visualization of planetary motion and orbital structures
Exploration of chaotic behavior in multi-body gravitational systems
🧠 Key Concepts Implemented

This project demonstrates understanding of:

Newtonian gravitation (inverse-square law)
N-body interaction dynamics
Velocity-Verlet numerical integration method
Orbital mechanics using Keplerian elements
Simplified relativistic correction (Schwarzschild precession approximation)
Large-scale particle simulation (test particles for asteroid and Kuiper belts)


⚙️ System Features
Full Solar System planetary simulation (Sun + 8 planets)
Mutual gravitational interaction between planets (N-body system)
Asteroid belt simulation (2.1–3.3 AU)
Kuiper belt simulation (30–50 AU)
Real-time interactive 3D visualization (VPython)
Adjustable simulation speed
Trail visualization of orbital paths
Toggleable asteroid and Kuiper belt display

🔬 Scientific Insights
This simulation demonstrates several important physical behaviors:

Long-term orbital stability under Newtonian gravity
Sensitivity of multi-body systems to initial conditions (chaos)
Orbital precession effects due to relativistic correction
Structural separation between inner and outer Solar System dynamics

🛠️ Tech Stack
Python
NumPy
VPython (3D visualization)
📊 Numerical Method

The system uses the velocity-Verlet integration scheme, which provides improved stability compared to Euler methods for long-term orbital simulations:

Position update based on current velocity and acceleration
Acceleration recalculated at new positions
Symmetric update for improved energy conservation behavior
🚀 Future Improvements
Implementation of higher-order integrators (Runge-Kutta 4)
Energy conservation error analysis
Detection of orbital resonances (e.g., Jupiter–asteroid belt interactions)
GPU acceleration for large-scale particle systems
Improved relativistic modeling (post-Newtonian expansion)
📌 Note

This project is developed for educational purposes to explore computational physics and numerical methods in astrophysical systems.
