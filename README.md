# Diffractive Beam Shaper Design via Finite Element Method

A Python re-implementation of the methods described in:

> **P. Wocjan**, *"Entwurf diffraktiver Strahlformer mit der Methode der finiten Elemente"*
> Studienarbeit, Institut für Algorithmen und Kognitive Systeme,
> Universität Karlsruhe (TH), Sommersemester 1997.
> Supervised by Prof. Dr. Th. Beth and Dipl.-Inform. M. Schmid.

The original implementation was in C++ and integrated into **DigiOpt**, a diffractive optics design system developed at the IAKS. This Python version recreates the complete pipeline from scratch.

## The Problem

Design a *diffractive phase element* (DPE) that transforms an input light beam $f_1$ into a desired output intensity pattern $|f_2|^2$. The DPE is a thin optical element described by a transmission function

$$T_\text{DPE}(x,y) = \exp(i\,\phi(x,y))$$

where $\phi(x,y)$ is the phase function we need to compute. Wave propagation between the input and output planes is modeled by the 2D Fourier transform (Fraunhofer approximation).

## The Method

The key insight is to use **finite element meshes** to find a geometric transformation that redistributes the input energy into the desired output pattern, then recover the phase function via the **stationary phase approximation**.

### Pipeline

1. **Mesh generation**: Create topologically equivalent meshes over the input plane $E$ and output plane $A$. Nodes are placed using interpolation techniques (4-point for rectangular topology, 6-point for triangular).

2. **Mesh optimization**: Iteratively move mesh nodes so that corresponding cells in the input and output meshes contain equal energy. For constant-intensity signals, this reduces to equal-area optimization. For general signals, the area conditions are weighted by the local intensity.

3. **Phase recovery**: The mesh correspondences define a discrete geometric transformation $T$. By the stationary phase condition, $\nabla\phi(x,y) = c \cdot T(x,y)$. We approximate $\phi$ as a bivariate polynomial and solve for its coefficients via least squares on the gradient conditions.

4. **IFTA refinement**: The phase from step 3 serves as an excellent starting point for the Iterative Fourier Transform Algorithm (Gerchberg-Saxton), which further optimizes the SNR and diffraction efficiency.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Circle-to-square beam shaping (main example from the Studienarbeit)
python examples/circle_to_square.py

# Gaussian beam to cross pattern (general intensity distributions)
python examples/gaussian_to_cross.py
```

## Package Structure

```
diffractive_fem/
├── __init__.py
├── mesh.py            # Mesh generation & optimization (Sections 4.1-4.3)
├── phase.py           # Phase polynomial recovery (Section 4.4)
├── propagation.py     # Fourier propagation model (Section 1)
├── metrics.py         # SNR and diffraction efficiency (Section 5)
├── ifta.py            # Iterative Fourier Transform Algorithm
└── visualization.py   # Plotting utilities
```

## Mathematical Details

### Mesh Optimization (8-Point Algorithm)

For an interior node $P_0$ surrounded by 8 neighbors $P_1, \ldots, P_8$, the ideal position $C(x_c, y_c)$ that equalizes the four surrounding cell areas satisfies:

$$x_c = \frac{bf - de}{bc - ad}, \quad y_c = \frac{af - ce}{bc - ad}$$

where $a, b, c, d, e, f$ are linear combinations of the neighbor coordinates (see Section 4.1 of the Studienarbeit for the full formulas).

### Phase Recovery

The phase function is approximated as:

$$\phi(x,y) = \sum_{k+l \leq D-1} a_{kl} \, x^k y^l$$

The stationary phase condition $\nabla\phi(x_{ij}, y_{ij}) = c \cdot (\phi_x^{ij}, \phi_y^{ij})$ at each mesh node gives a linear system for the coefficients $a_{kl}$, solved via least squares.

### Quality Metrics

- **Signal-to-Noise Ratio**: $\text{SNR}_I(f_2, f) = \frac{\|f\|^2}{\| |f_2| - \alpha|f| \|^2}$, where $\alpha$ is the optimal scaling factor.
- **Diffraction Efficiency**: $\eta = \frac{\|f\|^2_{W_\text{Signal}}}{\|f_1\|^2_{W_\text{DE}}}$, the fraction of input energy reaching the signal window.

## The Original Studienarbeit

The `studienarbeit/` directory contains the original LaTeX source from 1997, ported to modern LaTeX (pdfLaTeX-compatible with AMS packages). Compile with:

```bash
cd studienarbeit
pdflatex ausarb.tex
pdflatex ausarb.tex  # twice for cross-references
```

## References

- H. Aagedal, *Optische Implementierung linearer Transformationen mittels diffraktiver Elemente*, Diplomarbeit, IAKS, Universität Karlsruhe, 1993.
- H. Aagedal, M. Schmid, S. Egner, J. Müller-Quade, Th. Beth, F. Wyrowski, *Analytical beam shaping with application to laser diode arrays*, JOSA A, Vol. 14, No. 7, 1997.
- T. Dresel, M. Beyerlein, J. Schwider, *Design of computer-generated beam-shaping holograms by iterative finite-element mesh adaptation*, Applied Optics, Vol. 35, 1996.
- R.W. Gerchberg, W.O. Saxton, *A practical algorithm for the determination of phase from image and diffraction plane pictures*, Optik, Vol. 35, 1972.
- F. Wyrowski, O. Bryngdahl, *Iterative Fourier-transform algorithm applied to computer holography*, JOSA A, Vol. 5, No. 7, 1988.

## License

MIT

## Author

**Pawel Wocjan** — original C++ implementation (1997), Python re-implementation (2026)
