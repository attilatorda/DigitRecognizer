"""Build a structural graph from a skeletonized binary image.

Nodes:
  endpoint   — skeleton pixel with exactly 1 neighbour (stroke tip)
  junction   — skeleton pixel with 3+ neighbours (fork or crossing)
  loop_node  — pixel in a pure cycle (no endpoint reachable from it)

Edges:
  path of 2-neighbour pixels connecting two nodes, classified as:
    straight  — cumulative angular deviation < 15 deg
    curved    — 15 – 120 deg
    bent      — > 120 deg  (sharp hook or U-turn)
"""

import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Neighbour utilities
# ---------------------------------------------------------------------------

_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _neighbours(r, c, skel):
    h, w = skel.shape
    return [(r+dr, c+dc) for dr,dc in _OFFSETS
            if 0 <= r+dr < h and 0 <= c+dc < w and skel[r+dr, c+dc]]


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """For every skeleton pixel, count 8-connected skeleton neighbours."""
    from scipy.ndimage import convolve
    k = np.ones((3,3), dtype=np.uint8)
    k[1,1] = 0
    cnt = convolve(skel.astype(np.uint8), k, mode='constant', cval=0)
    cnt[~skel.astype(bool)] = 0
    return cnt


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_graph(skeleton_u8: np.ndarray) -> dict:
    """
    Parameters
    ----------
    skeleton_u8 : (H, W) uint8, values 0 or 255

    Returns
    -------
    dict with keys:
      'nodes'   : list of {'pos': (r,c), 'type': str}
      'edges'   : list of {'a': int, 'b': int, 'pixels': list[(r,c)], 'type': str}
      'isolated': list of (r,c)  — skeleton pixels not reachable by tracing
    """
    skel = skeleton_u8 > 0
    if not skel.any():
        return {'nodes': [], 'edges': [], 'isolated': []}

    ncnt = _neighbour_count(skel)

    # Classify structural pixels
    endpoints  = set(zip(*np.where((ncnt == 1) & skel)))
    junctions  = set(zip(*np.where((ncnt >= 3) & skel)))
    structural = endpoints | junctions
    path_pixels = set(zip(*np.where((ncnt == 2) & skel)))

    # Build node list
    nodes = []
    node_pos_to_idx = {}
    for pos in sorted(structural):
        idx = len(nodes)
        node_type = 'junction' if pos in junctions else 'endpoint'
        nodes.append({'pos': pos, 'type': node_type})
        node_pos_to_idx[pos] = idx

    edges = []
    visited_path = set()

    def _trace(start_pos, first_step):
        """Walk path pixels from start_pos in direction of first_step until
        a structural pixel is reached. Returns (end_pos, pixel_list)."""
        path = [start_pos, first_step]
        visited_path.add(first_step)
        cur = first_step
        while True:
            nbrs = [p for p in _neighbours(*cur, skel) if p != path[-2]]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt in structural:
                path.append(nxt)
                break
            if nxt in visited_path:
                break
            visited_path.add(nxt)
            path.append(nxt)
            cur = nxt
        return path

    # Trace edges from each structural pixel
    for node_idx, node in enumerate(nodes):
        r, c = node['pos']
        for nbr in _neighbours(r, c, skel):
            if nbr in structural:
                # Direct edge between two adjacent structural pixels
                nbr_idx = node_pos_to_idx[nbr]
                if nbr_idx > node_idx:
                    edge_type = _classify_edge([node['pos'], nbr])
                    edges.append({'a': node_idx, 'b': nbr_idx,
                                  'pixels': [node['pos'], nbr],
                                  'type': edge_type})
            elif nbr in path_pixels and nbr not in visited_path:
                visited_path.add(nbr)
                path = _trace(node['pos'], nbr)
                end = path[-1]
                if end in node_pos_to_idx:
                    end_idx = node_pos_to_idx[end]
                    edge_type = _classify_edge(path)
                    edges.append({'a': node_idx, 'b': end_idx,
                                  'pixels': path, 'type': edge_type})

    # Detect isolated loop pixels (no endpoints reachable — pure cycles)
    all_visited = visited_path | {n['pos'] for n in nodes}
    isolated = [p for p in path_pixels if p not in all_visited]
    # Mark isolated cycle pixels as loop_node nodes
    for pos in isolated:
        idx = len(nodes)
        nodes.append({'pos': pos, 'type': 'loop_node'})
        node_pos_to_idx[pos] = idx

    return {'nodes': nodes, 'edges': edges, 'isolated': isolated}


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------

def _classify_edge(pixels: list) -> str:
    """Classify an edge as straight / curved / bent by cumulative curvature."""
    if len(pixels) < 3:
        return 'straight'
    total_turn = 0.0
    for i in range(1, len(pixels) - 1):
        r0,c0 = pixels[i-1]
        r1,c1 = pixels[i]
        r2,c2 = pixels[i+1]
        v1 = np.array([r1-r0, c1-c0], dtype=float)
        v2 = np.array([r2-r1, c2-c1], dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        total_turn += np.degrees(np.arccos(cos_a))
    if total_turn < 15:
        return 'straight'
    if total_turn < 120:
        return 'curved'
    return 'bent'


# ---------------------------------------------------------------------------
# Graph-level queries
# ---------------------------------------------------------------------------

def bounding_box_diagonal(skeleton_u8: np.ndarray) -> float:
    """Diagonal of the tight bounding box around skeleton pixels."""
    rows, cols = np.where(skeleton_u8 > 0)
    if len(rows) == 0:
        return 1.0
    h = float(rows.max() - rows.min() + 1)
    w = float(cols.max() - cols.min() + 1)
    return max(np.sqrt(h*h + w*w), 1.0)


def edge_length(pixels: list) -> float:
    """Euclidean arc length of an edge (sum of step distances)."""
    if len(pixels) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pixels)):
        dr = pixels[i][0] - pixels[i-1][0]
        dc = pixels[i][1] - pixels[i-1][1]
        total += np.sqrt(dr*dr + dc*dc)
    return total
