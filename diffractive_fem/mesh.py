"""
Mesh generation and optimization for diffractive element design.

This module implements the finite element mesh approach described in Sections 4
of the Studienarbeit. The key idea:

  - Lay topologically equivalent meshes over input plane E and output plane A
  - Optimize node positions so corresponding mesh cells contain equal energy
  - The resulting node correspondences define a geometric transformation T

Two mesh topologies are supported:
  - Rectangular (4 neighbors per interior node, 8-point area equalization)
  - Triangular (6 neighbors per interior node, 9-point area equalization)
"""

import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Rectangular Mesh
# =============================================================================

class RectMesh:
    """
    A mesh with rectangular topology over a 2D domain.

    Nodes are indexed by (i, j) with i, j in {0, ..., s-1}.
    Interior nodes have 4 direct neighbors and 8 surrounding nodes.

    Parameters
    ----------
    s : int
        Number of nodes per side (total nodes = s*s).
    """

    def __init__(self, s: int):
        self.s = s
        # Node coordinates: x[i,j], y[i,j] in [0, 1]
        self.x = np.zeros((s, s))
        self.y = np.zeros((s, s))

    def copy(self):
        """Return a deep copy of this mesh."""
        m = RectMesh(self.s)
        m.x = self.x.copy()
        m.y = self.y.copy()
        return m

    # ---- Initialization methods ----

    def init_uniform_square(self):
        """
        Initialize as a uniform square grid on [0, 1] x [0, 1].
        This is the natural starting configuration.
        """
        s = self.s
        for i in range(s):
            for j in range(s):
                self.x[i, j] = i / (s - 1)
                self.y[i, j] = j / (s - 1)

    def init_uniform_circle(self, radius: float = 0.25, cx: float = 0.5, cy: float = 0.5):
        """
        Initialize boundary nodes uniformly on a circle, then interpolate
        interior nodes using the 4-point interpolation technique.

        This corresponds to the mesh generation algorithm in Section 4.1:
          Step 1: Place boundary nodes on the circle
          Step 2: Interpolate interior using 4-point formula
        """
        s = self.s

        # Step 1: Place boundary nodes on the circle
        # Map the boundary of the (i,j)-square to the circle
        # We traverse the boundary of the [0,s-1]x[0,s-1] square
        # and map each boundary node to the circle.
        boundary_nodes = self._get_boundary_order()
        n_boundary = len(boundary_nodes)

        for idx, (i, j) in enumerate(boundary_nodes):
            angle = 2 * np.pi * idx / n_boundary
            self.x[i, j] = cx + radius * np.cos(angle)
            self.y[i, j] = cy + radius * np.sin(angle)

        # Step 2: 4-point interpolation for interior nodes
        self._four_point_interpolation()

    def _get_boundary_order(self):
        """Return boundary nodes in counterclockwise traversal order."""
        s = self.s
        nodes = []
        # Bottom: (i, 0) for i = 0 .. s-1
        for i in range(s):
            nodes.append((i, 0))
        # Right: (s-1, j) for j = 1 .. s-1
        for j in range(1, s):
            nodes.append((s - 1, j))
        # Top: (i, s-1) for i = s-2 .. 0
        for i in range(s - 2, -1, -1):
            nodes.append((i, s - 1))
        # Left: (0, j) for j = s-2 .. 1
        for j in range(s - 2, 0, -1):
            nodes.append((0, j))
        return nodes

    def _four_point_interpolation(self):
        """
        Interpolate interior nodes from their 4 associated boundary nodes
        using the formula from Atkin (1994), Eq. (1) in the Studienarbeit.

        For each interior node (i0, j0), the 4 associated boundary nodes are:
          P1 = boundary node on same row (j=j0) at i=0
          P2 = boundary node on same row (j=j0) at i=s-1
          P3 = boundary node on same column (i=i0) at j=0
          P4 = boundary node on same column (i=i0) at j=s-1

        The weights g_l depend on the Euclidean distances d_l in (i,j)-space.
        """
        s = self.s
        for i0 in range(1, s - 1):
            for j0 in range(1, s - 1):
                # The 4 associated boundary nodes in (i,j) space
                boundary_ij = [
                    (0, j0),       # P1: left boundary
                    (s - 1, j0),   # P2: right boundary
                    (i0, 0),       # P3: bottom boundary
                    (i0, s - 1),   # P4: top boundary
                ]

                # Euclidean distances in (i,j)-space
                d = np.array([
                    np.sqrt((i0 - bi) ** 2 + (j0 - bj) ** 2)
                    for bi, bj in boundary_ij
                ])

                # Weights from the 4-point formula
                d1, d2, d3, d4 = d
                denom12 = d1 + d2
                prod12 = d1 * d2
                prod34 = d3 * d4

                g = np.zeros(4)
                g[0] = d2 * d3 * d4 / (denom12 * (prod12 + prod34))
                g[1] = d1 * d3 * d4 / (denom12 * (prod12 + prod34))
                g[2] = d1 * d2 * d4 / ((d3 + d4) * (prod34 + prod12))
                g[3] = d1 * d2 * d3 / ((d3 + d4) * (prod34 + prod12))

                # Interpolate
                self.x[i0, j0] = sum(
                    g[k] * self.x[boundary_ij[k]]
                    for k in range(4)
                )
                self.y[i0, j0] = sum(
                    g[k] * self.y[boundary_ij[k]]
                    for k in range(4)
                )

    # ---- Smoothing and optimization ----

    def laplace_smooth(self, n_iter: int = 50, alpha: float = 0.5):
        """
        Smooth interior nodes using the discrete Laplace operator.

        Each interior node is moved toward the centroid of its 4 direct
        neighbors: P0' = P0 + alpha * (C - P0), where C = mean of neighbors.
        """
        s = self.s
        for _ in range(n_iter):
            x_new = self.x.copy()
            y_new = self.y.copy()

            for i in range(1, s - 1):
                for j in range(1, s - 1):
                    cx = 0.25 * (
                        self.x[i - 1, j] + self.x[i + 1, j] +
                        self.x[i, j - 1] + self.x[i, j + 1]
                    )
                    cy = 0.25 * (
                        self.y[i - 1, j] + self.y[i + 1, j] +
                        self.y[i, j - 1] + self.y[i, j + 1]
                    )
                    x_new[i, j] = self.x[i, j] + alpha * (cx - self.x[i, j])
                    y_new[i, j] = self.y[i, j] + alpha * (cy - self.y[i, j])

            self.x = x_new
            self.y = y_new

    def optimize_equal_area(self, n_iter: int = 200, alpha: float = 0.5,
                            move_boundary: bool = False,
                            boundary_shape: str = 'circle',
                            boundary_radius: float = 0.25,
                            boundary_center: Tuple[float, float] = (0.5, 0.5)):
        """
        Optimize node positions so all mesh cells have equal area,
        using the 8-point algorithm from Section 4.1.

        For each interior node P0 with 8 neighbors P1..P8, the ideal
        position C(xc, yc) is computed such that the 4 surrounding
        quadrilateral areas satisfy A1=A2=A3=A4.

        Parameters
        ----------
        n_iter : int
            Number of optimization iterations.
        alpha : float
            Step size for node movement (0 < alpha <= 1).
        move_boundary : bool
            Whether to also optimize boundary nodes.
        boundary_shape : str
            Shape of the boundary ('circle' or 'square').
        boundary_radius : float
            Radius if boundary is a circle.
        boundary_center : tuple
            Center of the boundary shape.
        """
        s = self.s
        for iteration in range(n_iter):
            x_new = self.x.copy()
            y_new = self.y.copy()

            # Optimize interior nodes with 8-point algorithm
            for i in range(1, s - 1):
                for j in range(1, s - 1):
                    xc, yc = self._ideal_position_8pt(i, j)
                    if xc is not None and np.isfinite(xc) and np.isfinite(yc):
                        dx = xc - self.x[i, j]
                        dy = yc - self.y[i, j]
                        # Clamp movement to avoid instability
                        max_step = 0.5 / s
                        dx = np.clip(dx, -max_step, max_step)
                        dy = np.clip(dy, -max_step, max_step)
                        x_new[i, j] = self.x[i, j] + alpha * dx
                        y_new[i, j] = self.y[i, j] + alpha * dy

            # Optionally optimize boundary nodes
            if move_boundary:
                self._optimize_boundary_nodes(
                    x_new, y_new, alpha,
                    boundary_shape, boundary_radius, boundary_center
                )

            self.x = x_new
            self.y = y_new

    def _ideal_position_8pt(self, i, j):
        """
        Compute the ideal position for interior node (i,j) using the
        8-point equal-area algorithm.

        The 8 neighbors are numbered as in the Studienarbeit (Fig. "acht"):
          P8  P1  P2
          P7  P0  P3
          P6  P5  P4

        Returns (xc, yc) or (None, None) if the system is degenerate.
        """
        # Extract the 8 neighbor coordinates
        x1, y1 = self.x[i, j + 1], self.y[i, j + 1]       # P1: top
        x2, y2 = self.x[i + 1, j + 1], self.y[i + 1, j + 1]  # P2: top-right
        x3, y3 = self.x[i + 1, j], self.y[i + 1, j]       # P3: right
        x4, y4 = self.x[i + 1, j - 1], self.y[i + 1, j - 1]  # P4: bottom-right
        x5, y5 = self.x[i, j - 1], self.y[i, j - 1]       # P5: bottom
        x6, y6 = self.x[i - 1, j - 1], self.y[i - 1, j - 1]  # P6: bottom-left
        x7, y7 = self.x[i - 1, j], self.y[i - 1, j]       # P7: left
        x8, y8 = self.x[i - 1, j + 1], self.y[i - 1, j + 1]  # P8: top-left

        # Coefficients from the Studienarbeit (Section 4.1)
        a = y2 - y8 - y6 + y4
        b = x2 - x8 - x6 + x4
        c = y8 - y6 - y4 + y2
        d = x8 - x6 - x4 + x2
        e = (x1 * (y2 - y8) + x5 * (y4 - y6) +
             y1 * (x8 - x2) + y5 * (x6 - x4))
        f = (x3 * (y2 - y4) + x7 * (y8 - y6) +
             y3 * (x4 - x2) + y7 * (x6 - x8))

        det = b * c - a * d
        if abs(det) < 1e-14:
            return None, None

        xc = (b * f - d * e) / det
        yc = (a * f - c * e) / det
        return xc, yc

    def _optimize_boundary_nodes(self, x_new, y_new, alpha,
                                 shape, radius, center):
        """
        Optimize boundary nodes (non-corner) using the 5-point algorithm
        from Section 4.1. The ideal position lies on the intersection of
        the equal-area line with the boundary curve.
        """
        s = self.s
        cx_b, cy_b = center

        # Process each boundary edge
        for idx in range(1, s - 1):
            for (i, j), neighbors_5 in self._boundary_node_neighbors(idx):
                (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5) = neighbors_5

                a = y1 - 2 * y3 + y5
                b = -x1 + 2 * x3 - x5
                e = (x3 * y4 - y3 * x4 + x4 * y5 - y4 * x5 -
                     (x1 * y2 - y1 * x2 + x2 * y3 - y2 * x3))

                if shape == 'circle':
                    # Find intersection of line ax+by=e with circle
                    xc, yc = self._line_circle_intersection(
                        a, b, e, radius, center,
                        self.x[i, j], self.y[i, j]
                    )
                    if xc is not None:
                        # Move along the boundary (use angular interpolation)
                        angle_old = np.arctan2(
                            self.y[i, j] - cy_b, self.x[i, j] - cx_b
                        )
                        angle_new = np.arctan2(yc - cy_b, xc - cx_b)
                        angle_interp = angle_old + alpha * _angle_diff(angle_new, angle_old)
                        x_new[i, j] = cx_b + radius * np.cos(angle_interp)
                        y_new[i, j] = cy_b + radius * np.sin(angle_interp)

                elif shape == 'square':
                    # Boundary nodes stay on axis-parallel edges
                    # Compute position along the edge
                    xc, yc = self._line_edge_intersection(
                        a, b, e, i, j
                    )
                    if xc is not None:
                        x_new[i, j] = self.x[i, j] + alpha * (xc - self.x[i, j])
                        y_new[i, j] = self.y[i, j] + alpha * (yc - self.y[i, j])

    def _boundary_node_neighbors(self, idx):
        """Yield (i,j) and 5 neighbors for non-corner boundary nodes."""
        s = self.s
        # Bottom edge: j=0, i=idx
        i, j = idx, 0
        yield (i, j), [
            (self.x[i - 1, 0], self.y[i - 1, 0]),      # P1
            (self.x[i - 1, 1], self.y[i - 1, 1]),      # P2
            (self.x[i, 1], self.y[i, 1]),               # P3
            (self.x[i + 1, 1], self.y[i + 1, 1]),      # P4
            (self.x[i + 1, 0], self.y[i + 1, 0]),      # P5
        ]
        # Top edge: j=s-1, i=idx
        i, j = idx, s - 1
        yield (i, j), [
            (self.x[i + 1, s - 1], self.y[i + 1, s - 1]),
            (self.x[i + 1, s - 2], self.y[i + 1, s - 2]),
            (self.x[i, s - 2], self.y[i, s - 2]),
            (self.x[i - 1, s - 2], self.y[i - 1, s - 2]),
            (self.x[i - 1, s - 1], self.y[i - 1, s - 1]),
        ]
        # Left edge: i=0, j=idx
        i, j = 0, idx
        yield (i, j), [
            (self.x[0, j + 1], self.y[0, j + 1]),
            (self.x[1, j + 1], self.y[1, j + 1]),
            (self.x[1, j], self.y[1, j]),
            (self.x[1, j - 1], self.y[1, j - 1]),
            (self.x[0, j - 1], self.y[0, j - 1]),
        ]
        # Right edge: i=s-1, j=idx
        i, j = s - 1, idx
        yield (i, j), [
            (self.x[s - 1, j - 1], self.y[s - 1, j - 1]),
            (self.x[s - 2, j - 1], self.y[s - 2, j - 1]),
            (self.x[s - 2, j], self.y[s - 2, j]),
            (self.x[s - 2, j + 1], self.y[s - 2, j + 1]),
            (self.x[s - 1, j + 1], self.y[s - 1, j + 1]),
        ]

    @staticmethod
    def _line_circle_intersection(a, b, e, radius, center, x_current, y_current):
        """Find the intersection of line ax+by=e with a circle, closest to current pos."""
        cx, cy = center
        # Substitute parametrically: points on line ax+by=e
        if abs(b) > abs(a):
            # y = (e - a*x) / b; substitute into circle equation
            # (x-cx)^2 + ((e-a*x)/b - cy)^2 = r^2
            A = 1 + (a / b) ** 2
            B = -2 * cx + 2 * (a / b) * (cy - e / b)
            C = cx ** 2 + (e / b - cy) ** 2 - radius ** 2
        else:
            if abs(a) < 1e-14:
                return None, None
            # x = (e - b*y) / a; substitute
            A = 1 + (b / a) ** 2
            B = -2 * cy + 2 * (b / a) * (cx - e / a)
            C = cy ** 2 + (e / a - cx) ** 2 - radius ** 2

        disc = B ** 2 - 4 * A * C
        if disc < 0:
            return None, None

        sqrt_disc = np.sqrt(disc)
        solutions = []
        for sign in [-1, 1]:
            val = (-B + sign * sqrt_disc) / (2 * A)
            if abs(b) > abs(a):
                x_sol = val
                y_sol = (e - a * x_sol) / b
            else:
                y_sol = val
                x_sol = (e - b * y_sol) / a
            solutions.append((x_sol, y_sol))

        # Pick the solution closest to current position
        dists = [
            (xs - x_current) ** 2 + (ys - y_current) ** 2
            for xs, ys in solutions
        ]
        best = solutions[np.argmin(dists)]
        return best

    def _line_edge_intersection(self, a, b, e, i, j):
        """Find intersection of line ax+by=e with the square boundary edge containing (i,j)."""
        s = self.s
        if j == 0:
            # Bottom edge: y is fixed at y[i,j]
            y_fixed = self.y[i, j]
            if abs(a) < 1e-14:
                return None, None
            xc = (e - b * y_fixed) / a
            return xc, y_fixed
        elif j == s - 1:
            y_fixed = self.y[i, j]
            if abs(a) < 1e-14:
                return None, None
            xc = (e - b * y_fixed) / a
            return xc, y_fixed
        elif i == 0:
            x_fixed = self.x[i, j]
            if abs(b) < 1e-14:
                return None, None
            yc = (e - a * x_fixed) / b
            return x_fixed, yc
        elif i == s - 1:
            x_fixed = self.x[i, j]
            if abs(b) < 1e-14:
                return None, None
            yc = (e - a * x_fixed) / b
            return x_fixed, yc
        return None, None

    # ---- Energy-weighted optimization for general signals ----

    def optimize_equal_energy(self, signal: np.ndarray, n_iter: int = 200,
                              alpha: float = 0.3):
        """
        Optimize node positions so all mesh cells contain equal energy,
        for a signal with arbitrary intensity distribution.

        This is the generalization from Section 4.3 of the Studienarbeit.

        Strategy: Energy-weighted Laplace smoothing. Each interior node is
        moved toward the energy-weighted centroid of its 4 direct neighbors.
        The weight for each neighbor is the total energy in the cells that
        would SHRINK if the node moves toward that neighbor. This naturally
        clusters nodes in high-energy regions and spreads them in low-energy
        regions.

        Boundary nodes (non-corner) are moved along their edge to equalize
        the energy in the two adjacent cells.

        Parameters
        ----------
        signal : ndarray, shape (size, size)
            2D amplitude distribution. Energy is computed as amplitude^2.
        n_iter : int
            Number of optimization iterations.
        alpha : float
            Step size for node movement (0 < alpha <= 1).
        """
        s = self.s

        for iteration in range(n_iter):
            # Compute energy in every cell ONCE per iteration
            energies = self.cell_energies(signal)  # shape (s-1, s-1)

            x_new = self.x.copy()
            y_new = self.y.copy()

            # --- Interior nodes: energy-weighted Laplace smoothing ---
            for i in range(1, s - 1):
                for j in range(1, s - 1):
                    # The 4 cells surrounding node (i,j):
                    #   cell (i-1, j)   = upper-left  (NW)
                    #   cell (i,   j)   = upper-right (NE)
                    #   cell (i-1, j-1) = lower-left  (SW)
                    #   cell (i,   j-1) = lower-right (SE)
                    e_nw = energies[i - 1, j] if (i > 0 and j < s - 1) else 0
                    e_ne = energies[i, j] if (i < s - 1 and j < s - 1) else 0
                    e_sw = energies[i - 1, j - 1] if (i > 0 and j > 0) else 0
                    e_se = energies[i, j - 1] if (i < s - 1 and j > 0) else 0

                    # Weight for each neighbor = energy of adjacent cells
                    # Moving toward left neighbor shrinks NW and SW cells
                    w_left = e_nw + e_sw
                    w_right = e_ne + e_se
                    w_up = e_nw + e_ne
                    w_down = e_sw + e_se

                    total_w = w_left + w_right + w_up + w_down
                    if total_w < 1e-14:
                        continue

                    # Energy-weighted centroid of the 4 direct neighbors
                    cx = (w_left * self.x[i - 1, j] +
                          w_right * self.x[i + 1, j] +
                          w_down * self.x[i, j - 1] +
                          w_up * self.x[i, j + 1]) / total_w
                    cy = (w_left * self.y[i - 1, j] +
                          w_right * self.y[i + 1, j] +
                          w_down * self.y[i, j - 1] +
                          w_up * self.y[i, j + 1]) / total_w

                    new_x = self.x[i, j] + alpha * (cx - self.x[i, j])
                    new_y = self.y[i, j] + alpha * (cy - self.y[i, j])

                    # Only accept if it doesn't tangle the mesh
                    if self._move_is_valid(i, j, new_x, new_y, x_new, y_new):
                        x_new[i, j] = new_x
                        y_new[i, j] = new_y

            # --- Boundary nodes: balance energy of 2 adjacent cells ---
            self._optimize_boundary_energy_simple(
                x_new, y_new, energies, alpha
            )

            self.x = x_new
            self.y = y_new

    def _move_is_valid(self, i, j, new_x, new_y, x_arr, y_arr):
        """
        Check that moving node (i,j) to (new_x, new_y) doesn't create
        any negative-area (tangled) cells.
        """
        s = self.s
        # Check each cell that shares node (i,j)
        for di, dj in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
            ci, cj = i + di, j + dj
            if ci < 0 or ci >= s - 1 or cj < 0 or cj >= s - 1:
                continue
            # Get the 4 corners of cell (ci, cj), with the updated position
            corners_i = [(ci, cj), (ci + 1, cj), (ci + 1, cj + 1), (ci, cj + 1)]
            vx, vy = [], []
            for ni, nj in corners_i:
                if ni == i and nj == j:
                    vx.append(new_x)
                    vy.append(new_y)
                else:
                    vx.append(x_arr[ni, nj])
                    vy.append(y_arr[ni, nj])
            # Signed area (shoelace)
            area = 0
            for k in range(4):
                area += vx[k] * vy[(k + 1) % 4] - vx[(k + 1) % 4] * vy[k]
            if area <= 0:  # negative = tangled
                return False
        return True

    def _optimize_boundary_energy_simple(self, x_new, y_new, energies, alpha):
        """
        Move boundary nodes along their edge to balance energy in adjacent cells.
        """
        s = self.s

        # Bottom edge (j=0): cells at (i-1, 0) and (i, 0)
        for i in range(1, s - 1):
            e_left = energies[i - 1, 0]
            e_right = energies[i, 0]
            total = e_left + e_right
            if total < 1e-14:
                continue
            # Target: fraction of energy to the left
            frac = e_left / total
            x_left = self.x[i - 1, 0]
            x_right = self.x[i + 1, 0]
            x_target = x_left + frac * (x_right - x_left)
            x_new[i, 0] += alpha * (x_target - self.x[i, 0])

        # Top edge (j=s-1)
        for i in range(1, s - 1):
            e_left = energies[i - 1, s - 2]
            e_right = energies[i, s - 2]
            total = e_left + e_right
            if total < 1e-14:
                continue
            frac = e_left / total
            x_left = self.x[i - 1, s - 1]
            x_right = self.x[i + 1, s - 1]
            x_target = x_left + frac * (x_right - x_left)
            x_new[i, s - 1] += alpha * (x_target - self.x[i, s - 1])

        # Left edge (i=0)
        for j in range(1, s - 1):
            e_below = energies[0, j - 1]
            e_above = energies[0, j]
            total = e_below + e_above
            if total < 1e-14:
                continue
            frac = e_below / total
            y_below = self.y[0, j - 1]
            y_above = self.y[0, j + 1]
            y_target = y_below + frac * (y_above - y_below)
            y_new[0, j] += alpha * (y_target - self.y[0, j])

        # Right edge (i=s-1)
        for j in range(1, s - 1):
            e_below = energies[s - 2, j - 1]
            e_above = energies[s - 2, j]
            total = e_below + e_above
            if total < 1e-14:
                continue
            frac = e_below / total
            y_below = self.y[s - 1, j - 1]
            y_above = self.y[s - 1, j + 1]
            y_target = y_below + frac * (y_above - y_below)
            y_new[s - 1, j] += alpha * (y_target - self.y[s - 1, j])

    def _ideal_position_energy_weighted(self, i, j, signal, size):
        """
        Compute ideal position for (i,j) using energy-weighted 8-point algorithm.

        Same geometry as _ideal_position_8pt, but area conditions are weighted
        by the energy content: a*A1 + b*A2 = c*A3 + d*A4, etc.
        """
        # Get the 4 surrounding cell energies
        energies = self._compute_cell_energies_around(i, j, signal, size)
        if energies is None:
            return None, None
        e_a, e_b, e_c, e_d = energies

        # Get neighbor coordinates (same numbering as 8-point)
        x1, y1 = self.x[i, j + 1], self.y[i, j + 1]
        x2, y2 = self.x[i + 1, j + 1], self.y[i + 1, j + 1]
        x3, y3 = self.x[i + 1, j], self.y[i + 1, j]
        x4, y4 = self.x[i + 1, j - 1], self.y[i + 1, j - 1]
        x5, y5 = self.x[i, j - 1], self.y[i, j - 1]
        x6, y6 = self.x[i - 1, j - 1], self.y[i - 1, j - 1]
        x7, y7 = self.x[i - 1, j], self.y[i - 1, j]
        x8, y8 = self.x[i - 1, j + 1], self.y[i - 1, j + 1]

        # Energy-weighted coefficients
        # Condition 1: e_a*A1 + e_b*A2 = e_c*A3 + e_d*A4
        # Condition 2: e_a*A1 + e_c*A3 = e_b*A2 + e_d*A4
        # These linearize to equations for (xc, yc)
        a_coeff = e_a * (y2 - y8) + e_d * (y4 - y6) - e_b * (y4 - y6) - e_c * (y2 - y8)
        b_coeff = e_a * (x8 - x2) + e_d * (x6 - x4) - e_b * (x6 - x4) - e_c * (x8 - x2)
        e_val = (e_a * (x1 * (y2 - y8) + y1 * (x8 - x2)) +
                 e_d * (x5 * (y4 - y6) + y5 * (x6 - x4)) -
                 e_b * (x5 * (y4 - y6) + y5 * (x6 - x4)) -
                 e_c * (x1 * (y2 - y8) + y1 * (x8 - x2)))

        c_coeff = e_a * (y2 - y8) + e_b * (y4 - y6) - e_c * (y2 - y8) - e_d * (y4 - y6)
        d_coeff = e_a * (x8 - x2) + e_b * (x6 - x4) - e_c * (x8 - x2) - e_d * (x6 - x4)
        f_val = (e_a * (x3 * (y2 - y4) + y3 * (x4 - x2)) +
                 e_b * (x7 * (y8 - y6) + y7 * (x6 - x8)) -
                 e_c * (x3 * (y2 - y4) + y3 * (x4 - x2)) -
                 e_d * (x7 * (y8 - y6) + y7 * (x6 - x8)))

        det = b_coeff * c_coeff - a_coeff * d_coeff
        if abs(det) < 1e-14:
            # Fall back to unweighted
            return self._ideal_position_8pt(i, j)

        xc = (b_coeff * f_val - d_coeff * e_val) / det
        yc = (a_coeff * f_val - c_coeff * e_val) / det
        return xc, yc

    def _compute_cell_energies_around(self, i, j, signal, size):
        """Compute the energy in the 4 cells surrounding interior node (i,j)."""
        try:
            # The 4 quadrilateral cells around (i,j):
            # Cell A (top-left): nodes (i-1,j), (i-1,j+1), (i,j+1), (i,j)
            # Cell B (top-right): nodes (i,j), (i,j+1), (i+1,j+1), (i+1,j)
            # Cell C (bottom-left): nodes (i-1,j-1), (i-1,j), (i,j), (i,j-1)
            # Cell D (bottom-right): nodes (i,j-1), (i,j), (i+1,j), (i+1,j-1)
            cells = [
                [(i - 1, j), (i - 1, j + 1), (i, j + 1), (i, j)],
                [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)],
                [(i - 1, j - 1), (i - 1, j), (i, j), (i, j - 1)],
                [(i, j - 1), (i, j), (i + 1, j), (i + 1, j - 1)],
            ]
            energies = []
            for cell in cells:
                poly_x = [self.x[ci, cj] for ci, cj in cell]
                poly_y = [self.y[ci, cj] for ci, cj in cell]
                e = _compute_polygon_energy(poly_x, poly_y, signal, size)
                energies.append(max(e, 1e-10))  # avoid zero
            return energies
        except (IndexError, ValueError):
            return None

    def _optimize_boundary_energy(self, i, j, x_new, y_new, signal, size, alpha):
        """Optimize a boundary node using energy-weighted condition."""
        s = self.s
        is_boundary = (i == 0 or i == s - 1 or j == 0 or j == s - 1)
        is_corner = (i in [0, s - 1]) and (j in [0, s - 1])
        if not is_boundary or is_corner:
            return

        # Determine which edge and get the 2 adjacent cells
        if j == 0 and 0 < i < s - 1:
            # Bottom edge
            cell_a = [(i - 1, 0), (i - 1, 1), (i, 1), (i, 0)]
            cell_b = [(i, 0), (i, 1), (i + 1, 1), (i + 1, 0)]
        elif j == s - 1 and 0 < i < s - 1:
            cell_a = [(i, s - 1), (i, s - 2), (i - 1, s - 2), (i - 1, s - 1)]
            cell_b = [(i + 1, s - 1), (i + 1, s - 2), (i, s - 2), (i, s - 1)]
        elif i == 0 and 0 < j < s - 1:
            cell_a = [(0, j), (1, j), (1, j - 1), (0, j - 1)]
            cell_b = [(0, j + 1), (1, j + 1), (1, j), (0, j)]
        elif i == s - 1 and 0 < j < s - 1:
            cell_a = [(s - 1, j - 1), (s - 2, j - 1), (s - 2, j), (s - 1, j)]
            cell_b = [(s - 1, j), (s - 2, j), (s - 2, j + 1), (s - 1, j + 1)]
        else:
            return

        try:
            e_a = _compute_polygon_energy(
                [self.x[ci, cj] for ci, cj in cell_a],
                [self.y[ci, cj] for ci, cj in cell_a],
                signal, size
            )
            e_b = _compute_polygon_energy(
                [self.x[ci, cj] for ci, cj in cell_b],
                [self.y[ci, cj] for ci, cj in cell_b],
                signal, size
            )
        except (IndexError, ValueError):
            return

        if abs(e_a + e_b) < 1e-14:
            return

        # Move along edge toward equal energy partition
        if j == 0 or j == s - 1:
            # Horizontal edge: move in x
            ratio = e_a / (e_a + e_b)
            x_left = self.x[i - 1, j] if i > 0 else self.x[i, j]
            x_right = self.x[i + 1, j] if i < s - 1 else self.x[i, j]
            x_target = x_left + ratio * (x_right - x_left)
            x_new[i, j] = self.x[i, j] + alpha * (x_target - self.x[i, j])
        else:
            # Vertical edge: move in y
            ratio = e_a / (e_a + e_b)
            y_low = self.y[i, j - 1] if j > 0 else self.y[i, j]
            y_high = self.y[i, j + 1] if j < s - 1 else self.y[i, j]
            y_target = y_low + ratio * (y_high - y_low)
            y_new[i, j] = self.y[i, j] + alpha * (y_target - self.y[i, j])

    # ---- Metrics ----

    def cell_areas(self) -> np.ndarray:
        """
        Compute the area of each mesh cell.
        Returns array of shape (s-1, s-1).
        """
        s = self.s
        areas = np.zeros((s - 1, s - 1))
        for i in range(s - 1):
            for j in range(s - 1):
                # Quadrilateral with vertices (i,j), (i+1,j), (i+1,j+1), (i,j+1)
                # Using shoelace formula
                vx = [self.x[i, j], self.x[i + 1, j],
                      self.x[i + 1, j + 1], self.x[i, j + 1]]
                vy = [self.y[i, j], self.y[i + 1, j],
                      self.y[i + 1, j + 1], self.y[i, j + 1]]
                areas[i, j] = _polygon_area(vx, vy)
        return areas

    def cell_energies(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the energy in each mesh cell for a given signal.
        Returns array of shape (s-1, s-1).
        """
        s = self.s
        size = signal.shape[0]
        energies = np.zeros((s - 1, s - 1))
        for i in range(s - 1):
            for j in range(s - 1):
                vx = [self.x[i, j], self.x[i + 1, j],
                      self.x[i + 1, j + 1], self.x[i, j + 1]]
                vy = [self.y[i, j], self.y[i + 1, j],
                      self.y[i + 1, j + 1], self.y[i, j + 1]]
                energies[i, j] = _compute_polygon_energy(vx, vy, signal, size)
        return energies


# =============================================================================
# Triangular Mesh (Section 4.2)
# =============================================================================

class TriMesh:
    """
    A mesh with triangular topology.

    Nodes indexed by (i, j) with j = 0..s-1, i = 0..j.
    Total nodes = s*(s+1)/2.
    """

    def __init__(self, s: int):
        self.s = s
        # Store as dict for sparse indexing
        self.x = {}
        self.y = {}
        for j in range(s):
            for i in range(j + 1):
                self.x[i, j] = 0.0
                self.y[i, j] = 0.0

    def init_uniform_triangle(self, side: float = 0.5, cx: float = 0.5, cy: float = 0.4):
        """Initialize with equilateral triangle nodes uniformly placed."""
        s = self.s
        # Equilateral triangle vertices
        h = side * np.sqrt(3) / 2
        v0 = np.array([cx, cy + 2 * h / 3])
        v1 = np.array([cx - side / 2, cy - h / 3])
        v2 = np.array([cx + side / 2, cy - h / 3])

        for j in range(s):
            for i in range(j + 1):
                if j == 0:
                    t = 0
                else:
                    t = i / j if j > 0 else 0
                # Barycentric-like interpolation
                u = j / (s - 1) if s > 1 else 0
                pt = (1 - u) * v0 + u * ((1 - t) * v1 + t * v2)
                self.x[i, j] = pt[0]
                self.y[i, j] = pt[1]

    def init_circle(self, radius: float = 0.25, cx: float = 0.5, cy: float = 0.5):
        """Initialize with boundary on circle, interior via 6-point interpolation."""
        s = self.s
        # First init as triangle
        self.init_uniform_triangle()

        # Place boundary nodes on circle
        boundary = self._get_boundary_nodes()
        n_b = len(boundary)
        for idx, (i, j) in enumerate(boundary):
            angle = 2 * np.pi * idx / n_b
            self.x[i, j] = cx + radius * np.cos(angle)
            self.y[i, j] = cy + radius * np.sin(angle)

        # 6-point interpolation for interior
        self._six_point_interpolation()

    def _get_boundary_nodes(self):
        """Return boundary nodes in order."""
        s = self.s
        nodes = []
        # Left edge: (0, j) for j=0..s-1
        for j in range(s):
            nodes.append((0, j))
        # Bottom edge: (i, s-1) for i=1..s-1
        for i in range(1, s):
            nodes.append((i, s - 1))
        # Right edge: (j, j) for j=s-2..0
        for j in range(s - 2, -1, -1):
            nodes.append((j, j))
        return nodes

    def is_interior(self, i, j):
        """Check if (i,j) is an interior node."""
        s = self.s
        return 0 < i < j and j < s - 1

    def _six_point_interpolation(self):
        """
        6-point interpolation for interior nodes (Eq. 5 in the Studienarbeit).
        """
        s = self.s
        for j in range(2, s - 1):
            for i in range(1, j):
                if not self.is_interior(i, j):
                    continue

                # 6 associated boundary nodes (3 pairs along the 3 triangle directions)
                boundary_pts = self._find_6_boundary_nodes(i, j)
                if boundary_pts is None or len(boundary_pts) < 6:
                    continue

                d = np.array([
                    np.sqrt((i - bi) ** 2 + (j - bj) ** 2)
                    for bi, bj in boundary_pts
                ])

                d1, d2, d3, d4, d5, d6 = d
                denom1 = (d1 + d2) * (d1 * d2 + d3 * d4) * (d1 * d2 + d5 * d6)
                denom2 = (d1 + d2) * (d1 * d2 + d3 * d4) * (d1 * d2 + d5 * d6)
                denom3 = (d3 + d4) * (d3 * d4 + d1 * d2) * (d3 * d4 + d5 * d6)
                denom4 = (d3 + d4) * (d3 * d4 + d1 * d2) * (d3 * d4 + d5 * d6)
                denom5 = (d5 + d6) * (d5 * d6 + d1 * d2) * (d5 * d6 + d3 * d4)
                denom6 = (d5 + d6) * (d5 * d6 + d1 * d2) * (d5 * d6 + d3 * d4)

                if any(abs(dd) < 1e-14 for dd in [denom1, denom2, denom3, denom4, denom5, denom6]):
                    continue

                g = [
                    d2 * d3 * d4 * d5 * d6 / denom1,
                    d1 * d3 * d4 * d5 * d6 / denom2,
                    d1 * d2 * d4 * d5 * d6 / denom3,
                    d1 * d2 * d3 * d5 * d6 / denom4,
                    d1 * d2 * d3 * d4 * d6 / denom5,
                    d1 * d2 * d3 * d4 * d5 / denom6,
                ]

                self.x[i, j] = sum(g[k] * self.x[boundary_pts[k]] for k in range(6))
                self.y[i, j] = sum(g[k] * self.y[boundary_pts[k]] for k in range(6))

    def _find_6_boundary_nodes(self, i, j):
        """
        Find 6 boundary nodes for the 6-point interpolation.
        These are the boundary intersections along 3 directions from (i,j).
        """
        s = self.s
        # Direction 1: along constant j (horizontal) → left and right boundary
        # P1: (0, j) left boundary
        # P2: (j, j) right boundary
        p1 = (0, j)
        p2 = (j, j)

        # Direction 2: along constant i (vertical) → top and bottom of column
        # P3: (i, i) diagonal boundary
        # P4: (i, s-1) bottom boundary
        p3 = (i, i)
        p4 = (i, s - 1)

        # Direction 3: along constant j-i → left and bottom boundary
        # P5: (0, j-i) left boundary
        # P6: (j-i...? need to trace to bottom)
        diff = j - i
        p5 = (0, diff) if (0, diff) in self.x else (0, j)
        p6_i = s - 1 - diff
        p6 = (p6_i, s - 1) if (p6_i, s - 1) in self.x and p6_i >= 0 else (j, j)

        return [p1, p2, p3, p4, p5, p6]

    def laplace_smooth(self, n_iter: int = 50, alpha: float = 0.5):
        """Smooth interior nodes using 6-neighbor Laplace."""
        s = self.s
        for _ in range(n_iter):
            x_new = dict(self.x)
            y_new = dict(self.y)

            for j in range(1, s - 1):
                for i in range(1, j):
                    # 6 neighbors in triangular topology
                    neighbors = [
                        (i - 1, j), (i + 1, j),           # horizontal
                        (i, j - 1), (i, j + 1),           # vertical
                        (i - 1, j - 1), (i + 1, j + 1),   # diagonal
                    ]
                    valid = [(ni, nj) for ni, nj in neighbors if (ni, nj) in self.x]
                    if len(valid) == 0:
                        continue
                    cx_val = np.mean([self.x[n] for n in valid])
                    cy_val = np.mean([self.y[n] for n in valid])
                    x_new[i, j] = self.x[i, j] + alpha * (cx_val - self.x[i, j])
                    y_new[i, j] = self.y[i, j] + alpha * (cy_val - self.y[i, j])

            self.x = x_new
            self.y = y_new


# =============================================================================
# Helper functions
# =============================================================================

def _polygon_area(vx, vy):
    """Compute area of a polygon using the shoelace formula."""
    n = len(vx)
    area = 0.0
    for k in range(n):
        area += vx[k] * vy[(k + 1) % n] - vx[(k + 1) % n] * vy[k]
    return abs(area) / 2.0


def _point_in_polygon(qx, qy, poly_x, poly_y):
    """
    Ray-casting algorithm for point-in-polygon test.
    Implements the half-ray method described in Section 4.3 of the Studienarbeit.
    """
    n = len(poly_x)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]

        if ((yi > qy) != (yj > qy)) and \
                (qx < (xj - xi) * (qy - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _compute_polygon_energy(poly_x, poly_y, signal, size):
    """
    Compute the total energy (sum of intensity) inside a polygon,
    using the pixel-scanning approach from Section 4.3.
    """
    # Bounding box in pixel coordinates
    x_min_pix = max(0, int(min(poly_x) * (size - 1)))
    x_max_pix = min(size - 1, int(max(poly_x) * (size - 1)) + 1)
    y_min_pix = max(0, int(min(poly_y) * (size - 1)))
    y_max_pix = min(size - 1, int(max(poly_y) * (size - 1)) + 1)

    energy = 0.0
    for xp in range(x_min_pix, x_max_pix + 1):
        for yp in range(y_min_pix, y_max_pix + 1):
            x = xp / (size - 1)
            y = yp / (size - 1)
            if _point_in_polygon(x, y, poly_x, poly_y):
                energy += signal[yp, xp] ** 2  # intensity = amplitude^2
    return energy


def _angle_diff(a, b):
    """Compute the signed angular difference, wrapping around 2*pi."""
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff
