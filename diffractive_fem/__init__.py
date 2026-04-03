"""
Diffractive Beam Shaper Design via Finite Element Method
=========================================================

A Python re-implementation of the methods described in:

  P. Wocjan, "Entwurf diffraktiver Strahlformer mit der Methode der finiten Elemente"
  Studienarbeit, Institut für Algorithmen und Kognitive Systeme,
  Universität Karlsruhe (TH), Sommersemester 1997
  Supervised by Prof. Dr. Th. Beth and Dipl.-Inform. M. Schmid

The method designs diffractive phase elements (DPEs) by:
  1. Generating topologically equivalent meshes over input and output planes
  2. Optimizing meshes so corresponding finite elements contain equal energy
  3. Recovering the phase function via stationary phase approximation
  4. Optionally refining with the Iterative Fourier Transform Algorithm (IFTA)
"""

from .mesh import RectMesh, TriMesh
from .phase import recover_phase_polynomial, evaluate_phase
from .propagation import propagate, apply_dpe
from .metrics import compute_snr, compute_efficiency
from .ifta import ifta

__version__ = "1.0.0"
__author__ = "Pawel Wocjan (original 1997 C++), re-implemented in Python 2026"
