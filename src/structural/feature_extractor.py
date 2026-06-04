"""Extract hierarchical structural features from a skeletonized image.

Returns a list of (level, feature_type, size_class) tuples — one entry per
detected feature. Feed to bag_of_features.vectorize() for classification.

Level 1 — skeleton graph nodes (endpoints, junctions, loop_nodes)
Level 2 — skeleton graph edges (straight, curved, bent) + angle relations
Level 3 — composed structures (open_polygon, closed_loop, arc_chord, crossbar)
"""

import numpy as np
from .skeleton_graph import build_graph, bounding_box_diagonal, edge_length


# ---------------------------------------------------------------------------
# Size quantization
# ---------------------------------------------------------------------------

SIZE_BINS = [0.0, 0.125, 0.25, 0.75, 0.875, 1e9]
SIZE_LABELS = ['XS', 'S', 'M', 'L', 'XL']

def _size_class(length: float, diag: float) -> str:
    frac = length / diag
    for i, hi in enumerate(SIZE_BINS[1:]):
        if frac < hi:
            return SIZE_LABELS[i]
    return SIZE_LABELS[-1]


# ---------------------------------------------------------------------------
# Level 2 helper: angle between two edges at a shared junction
# ---------------------------------------------------------------------------

def _edge_direction(pixels: list, from_node_pos: tuple) -> np.ndarray:
    """Unit vector pointing away from from_node_pos along the edge."""
    if len(pixels) < 2:
        return np.array([0.0, 0.0])
    # Determine which end is the anchor
    if pixels[0] == from_node_pos:
        p0, p1 = pixels[0], pixels[min(2, len(pixels)-1)]
    else:
        p0, p1 = pixels[-1], pixels[max(0, len(pixels)-3)]
    v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two direction vectors."""
    cos_a = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features(skeleton_u8: np.ndarray) -> list:
    """
    Parameters
    ----------
    skeleton_u8 : (H, W) uint8 skeleton

    Returns
    -------
    List of (level: int, feature_type: str, size_class: str) tuples.
    """
    graph = build_graph(skeleton_u8)
    diag = bounding_box_diagonal(skeleton_u8)
    nodes = graph['nodes']
    edges = graph['edges']
    features = []

    # ------------------------------------------------------------------
    # Level 1 — nodes
    # ------------------------------------------------------------------
    # Nodes are structural pixels; their "size" is always 1px → XS
    for node in nodes:
        features.append((1, node['type'], 'XS'))

    # ------------------------------------------------------------------
    # Level 2 — edges
    # ------------------------------------------------------------------
    # Build adjacency for junction-angle analysis
    adj = {i: [] for i in range(len(nodes))}  # node_idx -> list of edge indices
    for ei, edge in enumerate(edges):
        adj[edge['a']].append(ei)
        adj[edge['b']].append(ei)

    for ei, edge in enumerate(edges):
        length = edge_length(edge['pixels'])
        sc = _size_class(length, diag)
        features.append((2, edge['type'], sc))

    # Right-angle pairs: two straight edges at a junction with ~80-100°
    # Parallel pairs:   two straight edges at a junction within 15° of parallel
    for node_idx, node in enumerate(nodes):
        if node['type'] != 'junction':
            continue
        incident = adj[node_idx]
        straight_edges = [ei for ei in incident if edges[ei]['type'] == 'straight']
        pos = node['pos']
        for i in range(len(straight_edges)):
            for j in range(i+1, len(straight_edges)):
                ei, ej = straight_edges[i], straight_edges[j]
                vi = _edge_direction(edges[ei]['pixels'], pos)
                vj = _edge_direction(edges[ej]['pixels'], pos)
                ang = _angle_between(vi, vj)
                if 80 <= ang <= 100:
                    # size = avg length of the two segments
                    avg_len = (edge_length(edges[ei]['pixels']) +
                               edge_length(edges[ej]['pixels'])) / 2
                    features.append((2, 'right_angle', _size_class(avg_len, diag)))
                elif ang < 15 or ang > 165:
                    avg_len = (edge_length(edges[ei]['pixels']) +
                               edge_length(edges[ej]['pixels'])) / 2
                    features.append((2, 'parallel_pair', _size_class(avg_len, diag)))

    # ------------------------------------------------------------------
    # Level 3 — composed structures
    # ------------------------------------------------------------------

    # Crossbar: a junction with 4+ incident edges all of which are roughly straight
    for node_idx, node in enumerate(nodes):
        if node['type'] != 'junction':
            continue
        incident = adj[node_idx]
        if len(incident) >= 4:
            straight = [ei for ei in incident if edges[ei]['type'] == 'straight']
            if len(straight) >= 4:
                total_len = sum(edge_length(edges[ei]['pixels']) for ei in straight)
                features.append((3, 'crossbar', _size_class(total_len / len(straight), diag)))

    # Open polygon: any junction that has 2+ right-angle pairs (≥3 straight segments
    # meeting with roughly 90° angles) — indicates a partial rectangle or L-shape
    for node_idx, node in enumerate(nodes):
        if node['type'] != 'junction':
            continue
        incident = adj[node_idx]
        straight = [ei for ei in incident if edges[ei]['type'] == 'straight']
        pos = node['pos']
        ra_count = 0
        for i in range(len(straight)):
            for j in range(i+1, len(straight)):
                vi = _edge_direction(edges[straight[i]]['pixels'], pos)
                vj = _edge_direction(edges[straight[j]]['pixels'], pos)
                if 80 <= _angle_between(vi, vj) <= 100:
                    ra_count += 1
        if ra_count >= 2:
            total = sum(edge_length(edges[ei]['pixels']) for ei in straight)
            features.append((3, 'open_polygon', _size_class(total / max(len(straight),1), diag)))

    # Closed loop: isolated loop pixels or cycles formed by two edges sharing endpoints
    if graph['isolated']:
        # Estimate loop diameter from isolated pixel cluster bounding box
        rows = [p[0] for p in graph['isolated']]
        cols = [p[1] for p in graph['isolated']]
        h = max(rows) - min(rows) + 1 if rows else 1
        w = max(cols) - min(cols) + 1 if cols else 1
        loop_size = np.sqrt(h*h + w*w)
        features.append((3, 'closed_loop', _size_class(loop_size, diag)))

    # Detect 2-edge cycles: two edges with same pair of endpoints
    seen_pairs = {}
    for ei, edge in enumerate(edges):
        pair = (min(edge['a'], edge['b']), max(edge['a'], edge['b']))
        if pair in seen_pairs:
            # Two paths between same nodes = cycle
            total = (edge_length(edge['pixels']) +
                     edge_length(edges[seen_pairs[pair]]['pixels']))
            features.append((3, 'closed_loop', _size_class(total / 2, diag)))
        else:
            seen_pairs[pair] = ei

    # Arc + chord: a curved edge whose two endpoint nodes are close together
    for edge in edges:
        if edge['type'] == 'curved':
            na = nodes[edge['a']]['pos']
            nb = nodes[edge['b']]['pos']
            chord_len = np.sqrt((na[0]-nb[0])**2 + (na[1]-nb[1])**2)
            arc_len   = edge_length(edge['pixels'])
            # If chord << arc, it forms a significant arc
            if arc_len > 0 and chord_len / arc_len < 0.7:
                features.append((3, 'arc_chord', _size_class(arc_len, diag)))

    return features
