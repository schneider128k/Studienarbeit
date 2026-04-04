"""
Triangular mesh topology for diffractive element design (Section 4.2).

Nodes indexed by (i, j) with j = 0,...,s-1 and i = 0,...,j.
Total nodes: s*(s+1)/2.

Three boundary edges:
  - Left:     i = 0,  j = 0,...,s-1
  - Diagonal: i = j,  j = 0,...,s-1
  - Bottom:   j = s-1, i = 0,...,s-1

Cells are triangles. Between rows j and j+1:
  - Downward triangles: (i,j), (i,j+1), (i+1,j+1)  for i = 0,...,j
  - Upward triangles:   (i+1,j), (i,j+1), (i+1,j+1) for i = 0,...,j-1
"""

import numpy as np
from typing import Tuple, List


class TriMesh:
    """Triangular mesh with nodes (i,j), j=0..s-1, i=0..j."""

    def __init__(self, s: int):
        self.s = s
        self.x = {}  # (i,j) -> x coordinate
        self.y = {}  # (i,j) -> y coordinate
        for j in range(s):
            for i in range(j + 1):
                self.x[i, j] = 0.0
                self.y[i, j] = 0.0

    def all_nodes(self):
        """Return list of all (i,j) pairs."""
        return [(i, j) for j in range(self.s) for i in range(j + 1)]

    def is_boundary(self, i, j):
        """Check if node (i,j) is on the boundary."""
        return i == 0 or i == j or j == self.s - 1

    def is_interior(self, i, j):
        return not self.is_boundary(i, j)

    def node_arrays(self):
        """Return flat arrays of all node coordinates (for phase recovery)."""
        nodes = self.all_nodes()
        xs = np.array([self.x[n] for n in nodes])
        ys = np.array([self.y[n] for n in nodes])
        return xs, ys

    # ---- Boundary traversal ----

    def boundary_nodes_ordered(self):
        """Return boundary nodes in counterclockwise order."""
        s = self.s
        nodes = []
        # Left edge: (0,j) for j=0 to s-1
        for j in range(s):
            nodes.append((0, j))
        # Bottom edge: (i, s-1) for i=1 to s-1
        for i in range(1, s):
            nodes.append((i, s - 1))
        # Diagonal edge: (j,j) for j=s-2 down to 1
        for j in range(s - 2, 0, -1):
            nodes.append((j, j))
        return nodes

    # ---- Initialization: uniform triangle ----

    def init_uniform_triangle(self, v0, v1, v2):
        """
        Place nodes uniformly on an equilateral (or any) triangle.

        v0 = top vertex, v1 = bottom-left, v2 = bottom-right.
        Node (i,j): row j has j+1 nodes linearly interpolated.
        """
        s = self.s
        v0, v1, v2 = np.array(v0), np.array(v1), np.array(v2)

        for j in range(s):
            u = j / (s - 1) if s > 1 else 0  # 0=apex, 1=bottom
            left = (1 - u) * v0 + u * v1
            right = (1 - u) * v0 + u * v2
            for i in range(j + 1):
                t = i / j if j > 0 else 0  # 0=left, 1=right
                pt = (1 - t) * left + t * right
                self.x[i, j] = pt[0]
                self.y[i, j] = pt[1]

    # ---- Initialization: boundary on circle, 6-point interpolation ----

    def init_circle(self, radius: float = 0.25, cx: float = 0.5, cy: float = 0.5):
        """
        Place boundary nodes uniformly on a circle,
        then interpolate interior using 6-point formula.
        """
        s = self.s

        # First init as uniform triangle (to set all positions)
        h = radius * np.sqrt(3)
        self.init_uniform_triangle(
            v0=(cx, cy + 2 * h / 3),
            v1=(cx - radius, cy - h / 3),
            v2=(cx + radius, cy - h / 3)
        )

        # Place boundary nodes uniformly on the circle
        boundary = self.boundary_nodes_ordered()
        n_b = len(boundary)
        for idx, (i, j) in enumerate(boundary):
            angle = np.pi / 2 - 2 * np.pi * idx / n_b  # start at top
            self.x[i, j] = cx + radius * np.cos(angle)
            self.y[i, j] = cy + radius * np.sin(angle)

        # 6-point interpolation for interior nodes
        self._six_point_interpolation()

    def _six_point_interpolation(self):
        """
        Interpolate interior nodes from their 6 associated boundary nodes
        along the 3 directional axes of the triangular grid.
        """
        s = self.s
        for j in range(1, s - 1):
            for i in range(1, j):
                if not self.is_interior(i, j):
                    continue

                # Find the 6 boundary nodes along 3 axes
                bp = self._find_6_boundary_points(i, j)
                if bp is None or len(bp) != 6:
                    continue

                # Distances in (i,j) index space
                d = [np.sqrt((i - bi) ** 2 + (j - bj) ** 2) for bi, bj in bp]

                # Weights from the 6-point formula (Studienarbeit Eq. 9)
                d1, d2, d3, d4, d5, d6 = d
                pairs = [(d1, d2), (d3, d4), (d5, d6)]

                # Check for zero distances
                if any(dd < 1e-14 for dd in d):
                    continue

                g = np.zeros(6)
                for k in range(6):
                    # Numerator: product of all d except d_k
                    num = 1.0
                    for m in range(6):
                        if m != k:
                            num *= d[m]
                    # Denominator: product of three terms
                    pair_idx = k // 2
                    other_pairs = [p for p_i, p in enumerate(pairs) if p_i != pair_idx]
                    dp_a, dp_b = pairs[pair_idx]
                    dp_prod = dp_a * dp_b
                    denom = (dp_a + dp_b)
                    for op in other_pairs:
                        denom *= (dp_prod + op[0] * op[1])
                    if abs(denom) < 1e-14:
                        continue
                    g[k] = num / denom

                # Normalize weights
                g_sum = np.sum(g)
                if g_sum < 1e-14:
                    continue
                g /= g_sum

                self.x[i, j] = sum(g[k] * self.x[bp[k]] for k in range(6))
                self.y[i, j] = sum(g[k] * self.y[bp[k]] for k in range(6))

    def _find_6_boundary_points(self, i0, j0):
        """
        Find 6 boundary nodes along the 3 axes for interior node (i0, j0).

        Axis 1 (constant j): left boundary (i=0) and diagonal (i=j)
        Axis 2 (constant i): diagonal (i=j, i.e. j=i) and bottom (j=s-1)
        Axis 3 (constant j-i): left boundary (i=0) and bottom (j=s-1)
        """
        s = self.s

        # Axis 1: constant j = j0
        p1 = (0, j0)       # left boundary
        p2 = (j0, j0)      # diagonal boundary

        # Axis 2: constant i = i0
        p3 = (i0, i0)      # diagonal boundary
        p4 = (i0, s - 1)   # bottom boundary

        # Axis 3: constant j - i = j0 - i0 = k
        k = j0 - i0
        p5 = (0, k)                    # left boundary (i=0, j=k)
        p6 = (s - 1 - k, s - 1)       # bottom boundary (j=s-1, i=s-1-k)

        # Validate all are valid nodes
        for p in [p1, p2, p3, p4, p5, p6]:
            if p not in self.x:
                return None

        return [p1, p2, p3, p4, p5, p6]

    # ---- Laplace smoothing ----

    def laplace_smooth(self, n_iter: int = 100, alpha: float = 0.5):
        """
        Smooth interior nodes using 6-neighbor Laplace operator.

        The 6 neighbors of (i,j) in triangular topology are:
          (i-1,j), (i+1,j), (i,j-1), (i,j+1), (i-1,j-1), (i+1,j+1)
        """
        s = self.s
        for _ in range(n_iter):
            x_new = dict(self.x)
            y_new = dict(self.y)

            for j in range(1, s - 1):
                for i in range(1, j):
                    if not self.is_interior(i, j):
                        continue

                    neighbors = [
                        (i - 1, j), (i + 1, j),
                        (i, j - 1), (i, j + 1),
                        (i - 1, j - 1), (i + 1, j + 1),
                    ]
                    valid = [(ni, nj) for ni, nj in neighbors
                             if (ni, nj) in self.x]
                    if not valid:
                        continue

                    cx = np.mean([self.x[n] for n in valid])
                    cy = np.mean([self.y[n] for n in valid])
                    x_new[i, j] = self.x[i, j] + alpha * (cx - self.x[i, j])
                    y_new[i, j] = self.y[i, j] + alpha * (cy - self.y[i, j])

            self.x = x_new
            self.y = y_new

    # ---- Visualization ----

    def get_edges(self):
        """Return list of ((x1,y1), (x2,y2)) for all mesh edges."""
        s = self.s
        edges = set()
        for j in range(s - 1):
            for i in range(j + 1):
                # Downward triangle: (i,j)-(i,j+1)-(i+1,j+1)
                a, b, c = (i, j), (i, j + 1), (i + 1, j + 1)
                for e in [(a, b), (b, c), (a, c)]:
                    edges.add(tuple(sorted(e)))
                # Upward triangle: (i+1,j)-(i,j+1)-(i+1,j+1)  if i < j
                if i < j:
                    a, b, c = (i + 1, j), (i, j + 1), (i + 1, j + 1)
                    for e in [(a, b), (b, c), (a, c)]:
                        edges.add(tuple(sorted(e)))
        return edges

    def plot(self, ax=None, title="Triangular Mesh", color='steelblue', linewidth=0.5):
        """Plot the triangular mesh."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        edges = self.get_edges()
        for (n1, n2) in edges:
            ax.plot([self.x[n1], self.x[n2]],
                    [self.y[n1], self.y[n2]],
                    '-', color=color, linewidth=linewidth)

        ax.set_aspect('equal')
        ax.set_title(title)
        return ax
