"""Rich structural feature vector — Track 6 v2.

Combines the original 60-dim bag-of-features (level x type x size counts) with
continuous descriptors that break the digit-collapse problem (1/2/3/5/7 all
reducing to {2 endpoints, 1 bent edge}):

  - topology counts          (7)  endpoints, junctions, loops, edges by type
  - orientation histogram    (8)  length-weighted edge orientation, 0-180 deg
  - curvature statistics     (3)  mean turn, max turn, total inflections
  - geometry                 (5)  aspect ratio, density, centroid, pixel count
  - endpoint vertical profile(3)  fraction of endpoints in top/mid/bottom third
  - loop position            (2)  loop present in top half / bottom half

Total continuous block = 28 dims.

v3 directional block (5 dims) — the *signed* curvature profile of the longest stroke,
which separates open single-stroke digits that share the same unsigned descriptors
(a 2 curves right-then-left, a 3 right-then-right, a 7 straight-then-sharp):

  - signed turn in stroke thirds (3)  net left/right bend in each third of the main stroke
  - net signed curvature        (1)  overall handedness across all edges
  - peak-curvature height       (1)  vertical position of the sharpest bend

Concatenated as [bag(60), cont(28), dir(5)] -> 93-dim vector.
"""

import numpy as np
from .skeleton_graph import build_graph, bounding_box_diagonal, edge_length
from .feature_extractor import extract_features
from .bag_of_features import vectorize, VOCAB_DIM

RICH_DIM = VOCAB_DIM + 28 + 5  # 60 + 28 + 5 = 93


def _edge_orientation_deg(pixels) -> float:
    """Undirected orientation of an edge in [0, 180), from first to last pixel."""
    if len(pixels) < 2:
        return 0.0
    dr = pixels[-1][0] - pixels[0][0]
    dc = pixels[-1][1] - pixels[0][1]
    ang = np.degrees(np.arctan2(dr, dc))  # -180..180
    if ang < 0:
        ang += 180.0
    if ang >= 180.0:
        ang -= 180.0
    return ang


def _inflection_count(pixels) -> int:
    """Number of curvature sign changes along an edge (S-curve detector)."""
    if len(pixels) < 3:
        return 0
    signs = []
    for i in range(1, len(pixels) - 1):
        r0, c0 = pixels[i - 1]
        r1, c1 = pixels[i]
        r2, c2 = pixels[i + 1]
        cross = (c1 - c0) * (r2 - r1) - (r1 - r0) * (c2 - c1)
        if cross > 0:
            signs.append(1)
        elif cross < 0:
            signs.append(-1)
    changes = 0
    prev = 0
    for s in signs:
        if s != 0 and prev != 0 and s != prev:
            changes += 1
        if s != 0:
            prev = s
    return changes


def _total_turn(pixels) -> float:
    if len(pixels) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(pixels) - 1):
        v1 = np.array([pixels[i][0] - pixels[i-1][0], pixels[i][1] - pixels[i-1][1]], float)
        v2 = np.array([pixels[i+1][0] - pixels[i][0], pixels[i+1][1] - pixels[i][1]], float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        total += np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
    return total


def _signed_turns(pixels):
    """Signed turn angle (deg, +left/-right) at each interior vertex of a path."""
    out = []
    for i in range(1, len(pixels) - 1):
        d1r, d1c = pixels[i][0] - pixels[i-1][0], pixels[i][1] - pixels[i-1][1]
        d2r, d2c = pixels[i+1][0] - pixels[i][0], pixels[i+1][1] - pixels[i][1]
        cross = d1r * d2c - d1c * d2r
        dot = d1r * d2r + d1c * d2c
        out.append(np.degrees(np.arctan2(cross, dot)))
    return out


def _directional_block(edges, rmin, h):
    """5-dim signed-curvature descriptor of the longest stroke (see module docstring)."""
    dir5 = np.zeros(5, dtype=np.float32)
    if not edges:
        return dir5
    longest = max(edges, key=lambda e: edge_length(e['pixels']))
    px = longest['pixels']
    turns = _signed_turns(px)
    if turns:
        # [0:3] net signed bend in each third of the main stroke (normalised by 180)
        k = len(turns)
        for j in range(3):
            seg = turns[j * k // 3:(j + 1) * k // 3]
            if seg:
                dir5[j] = float(np.clip(np.sum(seg) / 180.0, -1.0, 1.0))
        # [4] vertical position of the sharpest bend along the stroke
        peak = int(np.argmax(np.abs(turns)))
        dir5[4] = (px[peak + 1][0] - rmin) / max(h, 1.0)
    # [3] net handedness across all edges
    net = sum(np.sum(_signed_turns(e['pixels'])) for e in edges)
    dir5[3] = float(np.clip(net / 360.0, -1.0, 1.0))
    return dir5


def extract_rich_vector(skeleton_u8: np.ndarray) -> np.ndarray:
    """Return a 93-dim float32 feature vector for one skeleton image."""
    bag = vectorize(extract_features(skeleton_u8))  # 60-dim

    graph = build_graph(skeleton_u8)
    nodes, edges = graph['nodes'], graph['edges']
    diag = bounding_box_diagonal(skeleton_u8)

    rows, cols = np.where(skeleton_u8 > 0)
    cont = np.zeros(28, dtype=np.float32)
    if len(rows) == 0:
        return np.concatenate([bag, cont, np.zeros(5, dtype=np.float32)]).astype(np.float32)

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    h = float(rmax - rmin + 1)
    w = float(cmax - cmin + 1)

    n_end = sum(1 for n in nodes if n['type'] == 'endpoint')
    n_junc = sum(1 for n in nodes if n['type'] == 'junction')
    n_loop = sum(1 for n in nodes if n['type'] == 'loop_node')
    n_straight = sum(1 for e in edges if e['type'] == 'straight')
    n_curved = sum(1 for e in edges if e['type'] == 'curved')
    n_bent = sum(1 for e in edges if e['type'] == 'bent')

    # [0:7] topology counts (small ints, left raw)
    cont[0] = n_end
    cont[1] = n_junc
    cont[2] = n_loop
    cont[3] = len(edges)
    cont[4] = n_straight
    cont[5] = n_curved
    cont[6] = n_bent

    # [7:15] orientation histogram, length-weighted, normalized
    ori = np.zeros(8, dtype=np.float32)
    for e in edges:
        ang = _edge_orientation_deg(e['pixels'])
        length = edge_length(e['pixels'])
        b = min(int(ang / 22.5), 7)
        ori[b] += length
    if ori.sum() > 0:
        ori /= ori.sum()
    cont[7:15] = ori

    # [15:18] curvature stats
    turns = [_total_turn(e['pixels']) for e in edges]
    inflections = sum(_inflection_count(e['pixels']) for e in edges)
    cont[15] = (np.mean(turns) / 180.0) if turns else 0.0
    cont[16] = (np.max(turns) / 360.0) if turns else 0.0
    cont[17] = min(inflections, 10) / 10.0

    # [18:23] geometry
    cont[18] = h / max(w, 1.0)                       # aspect ratio
    cont[19] = len(rows) / max(h * w, 1.0)           # stroke density
    cont[20] = (rows.mean() - rmin) / max(h, 1.0)    # centroid y within bbox
    cont[21] = (cols.mean() - cmin) / max(w, 1.0)    # centroid x within bbox
    cont[22] = min(len(rows), 200) / 200.0           # pixel count (capped)

    # [23:26] endpoint vertical profile (top/mid/bottom third)
    if n_end > 0:
        ep_rows = [n['pos'][0] for n in nodes if n['type'] == 'endpoint']
        thirds = np.zeros(3)
        for r in ep_rows:
            t = min(int((r - rmin) / max(h, 1.0) * 3), 2)
            thirds[t] += 1
        thirds /= thirds.sum()
        cont[23:26] = thirds

    # [26:28] loop position (6 vs 9 vs 8 discriminator)
    loop_rows = [n['pos'][0] for n in nodes if n['type'] == 'loop_node']
    mid = (rmin + rmax) / 2.0
    cont[26] = 1.0 if any(r < mid for r in loop_rows) else 0.0
    cont[27] = 1.0 if any(r >= mid for r in loop_rows) else 0.0

    # v3 directional block (signed curvature of the longest stroke)
    dir5 = _directional_block(edges, rmin, h)

    return np.concatenate([bag, cont, dir5]).astype(np.float32)
