"""Grid A* base planner over fixture footprints for the scripted search oracle.

Privileged development tool: it reads the private fixture map, not policy
observations. Cells within `inflate` of any base blocker polygon, or within the
same margin of the room walls, are impassable. Paths are simplified by greedy
line-of-sight so the base servo receives a few straight legs.
(Copied from the room-search worktree, `room_search_planner.py`.)
"""
import heapq
import math

import numpy as np

from roboquest.search.layout import base_blockers, polygon_distance  # noqa: F401  (re-exported for tests)


class GridPlanner:
    def __init__(self, floor_bounds, blockers, resolution=.05, inflate=.42, wall_margin=.42):
        self.floor = [float(v) for v in floor_bounds]
        self.res = float(resolution)
        self.inflate = float(inflate)
        x0, x1, y0, y1 = self.floor
        self.nx = int(math.ceil((x1 - x0) / self.res)) + 1
        self.ny = int(math.ceil((y1 - y0) / self.res)) + 1
        self.blockers = dict(blockers)
        free = np.ones((self.nx, self.ny), dtype=bool)
        for ix in range(self.nx):
            for iy in range(self.ny):
                p = self.cell_centre((ix, iy))
                if (p[0] - x0 < wall_margin or x1 - p[0] < wall_margin
                        or p[1] - y0 < wall_margin or y1 - p[1] < wall_margin):
                    free[ix, iy] = False
                    continue
                if any(polygon_distance(p, poly) < self.inflate for poly in self.blockers.values()):
                    free[ix, iy] = False
        self.free = free

    def cell(self, xy):
        return (int(round((xy[0] - self.floor[0]) / self.res)), int(round((xy[1] - self.floor[2]) / self.res)))

    def cell_centre(self, cell):
        return (self.floor[0] + cell[0] * self.res, self.floor[2] + cell[1] * self.res)

    def inside(self, cell):
        return 0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny

    def nearest_free(self, cell, radius_m=.45):
        if self.inside(cell) and self.free[cell]:
            return cell
        r = int(math.ceil(radius_m / self.res))
        best, best_d = None, None
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                c = (cell[0] + dx, cell[1] + dy)
                if self.inside(c) and self.free[c]:
                    d = dx * dx + dy * dy
                    if best is None or d < best_d:
                        best, best_d = c, d
        if best is None:
            raise ValueError(f'no free cell within {radius_m} m of {self.cell_centre(cell)}')
        return best

    def line_free(self, a, b):
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for i in range(n + 1):
            t = i / max(n, 1)
            c = (int(round(a[0] + t * (b[0] - a[0]))), int(round(a[1] + t * (b[1] - a[1]))))
            if not (self.inside(c) and self.free[c]):
                return False
        return True

    def plan(self, start_xy, goal_xy):
        start = self.nearest_free(self.cell(start_xy))
        goal = self.nearest_free(self.cell(goal_xy))
        moves = [(1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]

        def h(c):
            dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        frontier = [(h(start), 0., start)]
        came, cost = {start: None}, {start: 0.}
        while frontier:
            _, g, current = heapq.heappop(frontier)
            if current == goal:
                break
            if g > cost.get(current, float('inf')):
                continue
            for dx, dy, w in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not (self.inside(nxt) and self.free[nxt]):
                    continue
                if dx and dy and not (self.free[(current[0] + dx, current[1])] and self.free[(current[0], current[1] + dy)]):
                    continue   # no corner cutting
                ng = g + w
                if ng < cost.get(nxt, float('inf')):
                    cost[nxt], came[nxt] = ng, current
                    heapq.heappush(frontier, (ng + h(nxt), ng, nxt))
        if goal not in came:
            raise ValueError(f'no base path from {start_xy} to {goal_xy}')
        cells = []
        node = goal
        while node is not None:
            cells.append(node)
            node = came[node]
        cells.reverse()
        # greedy line-of-sight simplification
        waypoints, anchor = [], 0
        while anchor < len(cells) - 1:
            far = anchor + 1
            for j in range(len(cells) - 1, anchor, -1):
                if self.line_free(cells[anchor], cells[j]):
                    far = j
                    break
            waypoints.append(self.cell_centre(cells[far]))
            anchor = far
        if waypoints:
            waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        else:
            waypoints = [(float(goal_xy[0]), float(goal_xy[1]))]
        return [list(map(float, w)) for w in waypoints], {'cells': len(cells), 'path_length_m': float(cost[goal] * self.res)}
