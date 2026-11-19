"""E0 — Oracle Object-Centric Behavior Transfer: shared geometric core.

Pure-numpy implementation shared by the ContextFlow-LIBERO and
BPP-LIBERO-Gen experiment drivers. A donor demonstration is reduced to an
object-centric EEF trajectory:

* gripper close/open transitions split the demo into pick windows;
* within each window the manipulated body is identified kinematically
  (largest displacement among movable bodies while the gripper is closed);
* the segment *before* the manipulated body starts moving is expressed in
  that body's frame (approach/grasp follows the object);
* the segment *after* is expressed in the placement anchor's frame
  (transport/place/release follows the goal);
* windows with no moving object (fixture stages — drawers, stove knobs)
  keep the world frame, which degrades the transform to raw replay.

At execution time each segment is re-anchored with the query scene's
*current* oracle pose of its reference body, then tracked closed-loop by a
proportional controller identical in gains/limits to the upstream
``move_to_pose`` servo. No learned components anywhere.

Poses are (pos[3], quat[4]) with xyzw quaternion convention.
"""

import numpy as np

# ---------------------------------------------------------------- quat math


def q_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def q_inv(q):
    out = np.asarray(q).copy()
    out[..., :3] *= -1.0
    return out


def q_normalize(q):
    return q / np.linalg.norm(q)


def q_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_quat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return q_normalize(np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]))
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q_normalize(q)


def q_rotvec(q):
    """Rotation vector (axis*angle) of quaternion, angle in (-pi, pi]."""
    q = q_normalize(np.asarray(q))
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-9:
        return np.zeros(3)
    return q[:3] / s * 2.0 * np.arctan2(s, q[3])


def rotvec_quat(rv):
    rv = np.asarray(rv, dtype=float)
    a = np.linalg.norm(rv)
    if a < 1e-12:
        return np.array([0., 0., 0., 1.])
    ax = rv / a
    return np.array([ax[0] * np.sin(a / 2), ax[1] * np.sin(a / 2),
                     ax[2] * np.sin(a / 2), np.cos(a / 2)])


def q_apply(q, p):
    """Rotate point(s) p by quaternion q."""
    return p + 2.0 * np.cross(q[:3], np.cross(q[:3], p) + q[3] * p)


def quat_distance(q1, q2):
    """Geodesic rotation distance in radians."""
    d = abs(float(np.dot(np.asarray(q1), np.asarray(q2))))
    return 2.0 * np.arccos(min(1.0, d))


# ------------------------------------------------------------------- poses


def pose_inv(pose):
    pos, quat = pose[:3], pose[3:7]
    qi = q_inv(quat)
    return np.r_[q_apply(qi, -pos), qi]


def pose_mul(a, b):
    """SE(3) product: apply b then a (frame change a * b)."""
    return np.r_[q_apply(a[3:7], b[:3]) + a[:3], q_mul(a[3:7], b[3:7])]


def pose_relative(pose, frame):
    """Express world ``pose`` in ``frame`` (both world poses)."""
    return pose_mul(pose_inv(frame), pose)


def pose_apply_delta(pose, delta):
    """Apply world-frame SE(3) delta ``delta`` to ``pose``."""
    return pose_mul(delta, pose)


def anchor_delta(anchor_query, anchor_demo):
    """SE(3) that maps poses from the demo anchor frame into the query."""
    return pose_mul(anchor_query, pose_inv(anchor_demo))


def pose_error(cur, des):
    """(pos_err_m, rot_err_rad) between two world poses."""
    pos_err = float(np.linalg.norm(des[:3] - cur[:3]))
    rel = q_mul(des[3:7], q_inv(cur[3:7]))
    return pos_err, float(np.linalg.norm(q_rotvec(rel)))


def ensure_quat_continuity(quats):
    """Flip signs so consecutive quaternions stay on the same hemisphere."""
    out = np.asarray(quats).copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


# --------------------------------------------------------- window detection


def detect_windows(gripper, close_th=0.5):
    """Pick windows from the gripper command channel (+1 close, -1 open).

    Returns list of (close_step, open_step) — open_step is the first step the
    command drops back to open; a trailing close is closed at trajectory end.
    """
    g = np.asarray(gripper, dtype=float)
    closes = list(np.nonzero(np.diff(g) > close_th)[0] + 1)
    opens = list(np.nonzero(np.diff(g) < -close_th)[0] + 1)
    windows = []
    oi = 0
    for c in closes:
        while oi < len(opens) and opens[oi] <= c:
            oi += 1
        if oi < len(opens):
            windows.append((int(c), int(opens[oi])))
            oi += 1
        else:
            windows.append((int(c), len(g)))
    if not windows and len(g):
        # No transitions: gripper held one state the whole demo.
        pass
    return windows


def movable_displacements(obj_poses, window, movable_idx):
    """Per-candidate translation drift inside a gripper window.

    ``obj_poses``: (T, K, 7) recorded body poses; ``movable_idx`` selects the
    movable rows. Returns (K_movable,) max |dp| inside the window and the
    step the max is reached (used for diagnostics only).
    """
    t0, t1 = window
    seg = obj_poses[t0:t1 + 1, movable_idx, :]
    disp = np.linalg.norm(seg - seg[:1], axis=-1)      # (T_w, K_m)
    return disp.max(axis=0)


def first_motion_step(obj_poses, window, body_idx, thresh=0.008, start=None):
    """First step >= start where the body's pos moved >thresh vs window start.

    Returns None if the body never moved past ``thresh`` inside the window.
    """
    t0, t1 = window
    i0 = t0 if start is None else max(start, t0)
    seg = obj_poses[t0:t1 + 1, body_idx, :3]
    base = obj_poses[i0, body_idx, :3] if i0 - t0 <= len(seg) - 1 else seg[0]
    idx = np.nonzero(np.linalg.norm(seg[i0 - t0:] - base, axis=-1) > thresh)[0]
    return int(t0 + idx[0]) if len(idx) else None


def motion_runs(obj_poses, t0, t1, body_idx, step_eps=0.0008, min_len=8,
                min_disp=0.015):
    """Contiguous runs where a body's per-step translation exceeds step_eps.

    Returns list of (start, end) index pairs (end exclusive), filtered to
    runs of at least ``min_len`` steps and ``min_disp`` total displacement.
    """
    pos = obj_poses[t0:t1, body_idx, :3]
    if len(pos) < 2:
        return []
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    moving = vel > step_eps
    runs = []
    i = 0
    while i < len(moving):
        if moving[i]:
            j = i
            while j < len(moving) and moving[j]:
                j += 1
            disp = float(np.linalg.norm(pos[j] - pos[i]))
            if j - i >= min_len and disp >= min_disp:
                runs.append((t0 + i, t0 + j))
            i = j
        else:
            i += 1
    return runs


# ------------------------------------------------------------- segmentation


def segment_donor(eef_poses, obj_poses, body_names, movable_names,
                  gripper, anchor_hints=None, motion_thresh=0.015,
                  release_margin=6):
    """Cut a donor trajectory into anchored segments.

    ``eef_poses``: (T,7) donor EEF poses (post-action).
    ``obj_poses``: (T,K,7) per-step body poses aligned with eef_poses steps.
    ``body_names``: K names matching obj_poses rows.
    ``movable_names``: names eligible as manipulated objects (not fixtures).
    ``gripper``: (T,) gripper command channel.
    ``anchor_hints``: optional dict manipulated_name -> anchor_name
        (e.g. from BDDL goal predicates / task metadata Place steps).
    ``motion_thresh``: displacement (m) for a body to count as manipulated.
    ``release_margin``: steps after window open still counted as carry.

    Returns dict(segments=[(t0,t1,frame_name)], windows=[info]).
    frame_name is a body name or 'world' (identity transform).
    """
    T = len(eef_poses)
    names = list(body_names)
    name2row = {n: i for i, n in enumerate(names)}
    movable_idx = [name2row[n] for n in movable_names if n in name2row]
    windows = detect_windows(gripper)
    winfo = []
    for w in windows:
        info = {'window': w, 'manipulated': None, 'anchor': None,
                'split': None}
        if movable_idx:
            disp = movable_displacements(obj_poses, w, movable_idx)
            k = int(np.argmax(disp))
            if disp[k] > motion_thresh:
                info['manipulated'] = names[movable_idx[k]]
        winfo.append(info)

    for info in winfo:
        m = info['manipulated']
        w = info['window']
        if m is None:
            # Fixture interaction (drawer/knob/door): no movable body moved.
            # Anchor the transit+window to the body nearest the EEF mid-window
            # (usually the fixture being actuated) so it transfers with that
            # body's frame instead of staying world-fixed.
            mid = (w[0] + w[1]) // 2
            info['anchor'] = nearest_body(
                eef_poses[mid, :3], obj_poses[mid], names, exclude=())
            continue
        if anchor_hints and m in anchor_hints:
            info['anchor'] = anchor_hints[m]
        else:
            info['anchor'] = nearest_body(
                obj_poses[w[1] - 1, name2row[m], :3], obj_poses[w[1] - 1],
                names, exclude={m})
        s = first_motion_step(obj_poses, w, name2row[m], thresh=0.008,
                              start=w[0])
        if s is None:
            # Closed on nothing movable: treat like a fixture window.
            info['manipulated'] = None
            mid = (w[0] + w[1]) // 2
            info['anchor'] = nearest_body(
                eef_poses[mid, :3], obj_poses[mid], names, exclude=())
            continue
        info['split'] = s

    # Push detection: movable bodies moving inside open-gripper spans are
    # treated as pushed objects; the span up to motion onset approaches them
    # in their frame and the motion itself is carried in that body's frame.
    def push_runs(t0, t1, exclude=()):
        best = None
        for idx in movable_idx:
            if names[idx] in exclude:
                continue
            for r in motion_runs(obj_poses, t0, t1, idx):
                if best is None or r[1] - r[0] > best[1] - best[0]:
                    best = (*r, names[idx])
        return best

    # Emit ordered segments.
    segments = []
    cursor = 0
    for info in winfo:
        w = info['window']
        m = info['manipulated']
        if m is None:
            # Fixture span: transit+window in the nearest body's frame.
            if w[0] > cursor:
                push = push_runs(cursor, w[0], exclude=())
                if push:
                    p0, p1, pn = push
                    segments.append((cursor, p0, pn))
                    segments.append((p0, p1, pn))
                    cursor = p1
                if w[0] > cursor:
                    segments.append((cursor, w[0], info['anchor'] or 'world'))
            if w[1] > cursor:
                segments.append((max(cursor, w[0]), w[1],
                                 info['anchor'] or 'world'))
            cursor = max(cursor, w[1])
            continue
        s = info['split']
        if s > cursor:
            push = push_runs(cursor, s, exclude={m})
            if push:
                p0, p1, pn = push
                if p0 > cursor:
                    segments.append((cursor, p0, pn))
                segments.append((p0, p1, pn))
                cursor = p1
            if s > cursor:
                segments.append((cursor, s, m))      # approach in obj frame
        carry_end = min(w[1] + release_margin, T)
        segments.append((max(s, cursor), carry_end, info['anchor'] or 'world'))
        cursor = carry_end
    if cursor < T:
        last_anchor = 'world'
        for info in reversed(winfo):
            if info['manipulated'] or info['anchor']:
                last_anchor = info['anchor'] or 'world'
                break
        segments.append((cursor, T, last_anchor))
    return {'segments': segments, 'windows': winfo}


def nearest_body(pos, poses_at_t, names, exclude=()):
    """Name of the body whose position is nearest ``pos`` (XY-weighted)."""
    best, best_d = None, np.inf
    for i, n in enumerate(names):
        if n in exclude:
            continue
        d = np.linalg.norm(poses_at_t[i, :3] - pos)
        if d < best_d:
            best, best_d = n, d
    return best


# ------------------------------------------------------------- transforms


def transform_poses(poses, delta):
    """Apply world-frame SE(3) ``delta`` to a (N,7) pose array."""
    out = np.asarray(poses).copy()
    if delta is None:
        return out
    for i in range(len(out)):
        out[i] = pose_mul(delta, out[i])
    return out


def transform_pose(pose, delta):
    return pose_mul(delta, pose)


# -------------------------------------------------------------- P tracker


class PoseTracker:
    """Closed-loop proportional tracker for a desired pose trajectory.

    Progress-based (pure pursuit): ``self.j`` tracks how far along the
    desired path the EEF has travelled (argmin distance over a forward
    window); the commanded target sits ``lookahead`` waypoints ahead so the
    P controller always pulls toward a moving goal instead of stalling
    behind a waypoint it lags. At segment (frame) boundaries the lookahead
    is disabled until the EEF converges to the boundary pose (stitch).

    Gripper transitions fire when the EEF reaches the *spatial* vicinity of
    the donor's close/open pose (or progress overruns it), preserving the
    place where the demo actuated the gripper rather than the time.
    """

    def __init__(self, des_poses, des_gripper, segments=None,
                 kp_pos=6.0, kp_rot=2.0, max_pos_speed=0.5,
                 max_rot_speed=0.5, reach_pos=0.018, reach_rot=0.12,
                 lookahead=6, progress_window=3,
                 stitch_steps=30, stitch_pos=0.020,
                 grip_thresh=0.018, grip_slack=8):
        self.des = np.asarray(des_poses)
        self.des[:, 3:7] = ensure_quat_continuity(self.des[:, 3:7])
        self.grip = np.asarray(des_gripper)
        self.T = len(self.des)
        self.kp_pos, self.kp_rot = kp_pos, kp_rot
        self.max_pos, self.max_rot = max_pos_speed, max_rot_speed
        self.reach_pos, self.reach_rot = reach_pos, reach_rot
        self.lookahead = lookahead
        self.progress_window = progress_window
        self.stitch_steps = stitch_steps
        self.stitch_pos = stitch_pos
        self.grip_thresh = grip_thresh
        self.grip_slack = grip_slack
        self.stitches = set()
        if segments:
            for (t0, _t1, _f) in segments[1:]:
                if 0 < t0 < self.T:
                    self.stitches.add(t0)
        # Pending gripper transitions: [(index, new_value)].
        dg = np.diff(self.grip)
        self.trans = [(i + 1, float(self.grip[i + 1]))
                      for i in np.nonzero(np.abs(dg) > 0.5)[0]]
        # Grip-close transitions are contact-critical waypoints: gate
        # progress past them until the EEF converges (the grasp point must
        # be reached before the fingers close) and let the gate release
        # drive the close — a loose spatial early-fire closes the fingers
        # ~1 cm above the bowl on slow descents. Release transitions are
        # deliberately NOT gated — dwelling at a release waypoint while the
        # held object presses the target disturbs the placement.
        self.stitch_thresh = {}
        self.gated_grips = set()
        for gi, gv in self.trans:
            if gv > 0 and 0 < gi < self.T - 1:
                self.stitches.add(gi)
                self.stitch_thresh[gi] = 0.010
                self.gated_grips.add(gi)
        self._grip_val = float(np.clip(self.grip[0], -1.0, 1.0)) \
            if len(self.grip) else -1.0
        self.j = 0
        self.stitch_wait = 0
        self._stitch = None   # stitch index currently being held
        self._stuck = 0
        self._tgt_err = None
        self._tgt_static = 0
        self._gate = self.T - 1

    def _update_progress(self, cur):
        # Progress may reach the next stitch index but cannot pass it while
        # the stitch is still held (unconverged frame boundary). The forward
        # window is small so non-monotone paths (descend-then-lift dips) are
        # not skipped by the argmin.
        self._gate = min([s for s in self.stitches if s >= self.j],
                         default=self.T - 1)
        hi = min(self.j + self.progress_window, self._gate)
        d = np.linalg.norm(self.des[self.j:hi + 1, :3] - cur[:3], axis=-1)
        adv = int(np.argmin(d))
        self.j += adv
        self._stuck = self._stuck + 1 if adv == 0 else 0

    def _maybe_skip(self):
        """Advance past a physically unreachable waypoint.

        Fires only when the EEF makes no progress toward the pursuit target
        either (blocked by contact / kinematic limit), not merely when the
        argmin index stalls — otherwise dips the EEF must actually visit
        would be skipped.
        """
        if self._stuck >= 4 and self._tgt_static >= 10 \
                and self.j < self._gate:
            self.j += 1
            self._stuck = 0
            self._tgt_static = 0

    def _current_grip(self, cur):
        # Latched semantics: once a transition fires its value holds until
        # the next transition fires — never falls back to grip[j] (which
        # would undo an early spatial fire while progress still lags).
        while self.trans:
            gi, gv = self.trans[0]
            if gi in self.gated_grips:
                # Fire only once the grasp-point gate has released (EEF
                # converged to within stitch_thresh, or patience lapsed) —
                # never on the loose spatial threshold alone.
                fire = gi not in self.stitches
            else:
                pe, _ = pose_error(cur, self.des[gi])
                fire = pe <= self.grip_thresh or self.j >= gi + self.grip_slack
            if fire:
                self.trans.pop(0)
                self._grip_val = gv
            else:
                break
        return self._grip_val

    def step(self, cur_pose):
        """Compute one delta action toward the current pursuit point."""
        self._update_progress(cur_pose)
        # Target: lookahead past progress, clamped at a held stitch.
        t_idx = min(self.j + self.lookahead, self.T - 1)
        held = [s for s in self.stitches if self.j <= s <= t_idx]
        if held:
            s = min(held)
            if self._stitch != s:
                self._stitch, self.stitch_wait = s, 0
            pe, _ = pose_error(cur_pose, self.des[s])
            th = self.stitch_thresh.get(s, self.stitch_pos)
            if pe <= th or self.stitch_wait >= self.stitch_steps:
                self.stitches.discard(s)
                self._stitch = None
            else:
                self.stitch_wait += 1
                t_idx = s
        des = self.des[t_idx]
        pos_err = des[:3] - cur_pose[:3]
        tgt_err = float(np.linalg.norm(pos_err))
        if self._tgt_err is not None and abs(tgt_err - self._tgt_err) < 1e-3:
            self._tgt_static += 1
        else:
            self._tgt_static = 0
        self._tgt_err = tgt_err
        pos_action = self.kp_pos * pos_err
        n = np.linalg.norm(pos_action)
        if n > self.max_pos:
            pos_action *= self.max_pos / n
        rel = q_mul(des[3:7], q_inv(cur_pose[3:7]))
        rot_action = self.kp_rot * q_rotvec(rel)
        n = np.linalg.norm(rot_action)
        if n > self.max_rot:
            rot_action *= self.max_rot / n
        g = self._current_grip(cur_pose)
        self._maybe_skip()
        return np.r_[pos_action, rot_action, g], self.j

    @property
    def done(self):
        return self.j >= self.T - 1


def proportional_action(cur, des, gripper, kp_pos=6.0, kp_rot=2.0,
                        max_pos=0.5, max_rot=0.5):
    """Single proportional-control delta action (same law as upstream)."""
    pos_action = kp_pos * (des[:3] - cur[:3])
    n = np.linalg.norm(pos_action)
    if n > max_pos:
        pos_action *= max_pos / n
    rel = q_mul(des[3:7], q_inv(cur[3:7]))
    rot_action = kp_rot * q_rotvec(rel)
    n = np.linalg.norm(rot_action)
    if n > max_rot:
        rot_action *= max_rot / n
    return np.r_[pos_action, rot_action, np.clip(gripper, -1.0, 1.0)]
