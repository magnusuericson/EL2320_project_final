
import numpy as np

class ParticleFilter:
    """
    Simple PF tracking lane as [x_bottom, x_top] in image coordinates.
    """
    def __init__(self, N, init_measurement, init_std):
        self.N = N
        init_measurement = np.asarray(init_measurement, dtype=np.float32)

        # particles: shape (N, 2)
        self.particles = np.tile(init_measurement, (N, 1)) \
                         + np.random.randn(N, 2) * init_std

        self.weights = np.ones(N, dtype=np.float32) / N

        self.process_std = np.array([3.0, 3.0], dtype=np.float32)
        self.meas_std    = np.array([10.0, 10.0], dtype=np.float32)

        self.process_std_base = np.array([2.0, 2.0], dtype=np.float32)
        self.meas_std_base    = np.array([14.0, 14.0], dtype=np.float32)

        self.process_std_max  = np.array([10.0, 10.0], dtype=np.float32)
        self.meas_std_min     = np.array([5.0, 5.0], dtype=np.float32)

        self.process_std = self.process_std_base.copy()
        self.meas_std    = self.meas_std_base.copy()

        self.prev_theta = None
        self.turn_ema = 0.0
        self.turn_alpha = 0.15  

    def _wrap_pi(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi


    def adapt_noise(self, z, y_bottom, y_top):
        """
        Update process_std (Q) and meas_std (R) based on how fast
        the observed lane angle changes.
        """
        if z is None or y_bottom is None or y_top is None:
            self.process_std = np.minimum(self.process_std * 1.10, self.process_std_max)
            self.meas_std = self.meas_std_base.copy()
            return

        xb, xt = float(z[0]), float(z[1])
        dy = float(y_top - y_bottom)
        dx = float(xt - xb)
        theta = np.arctan2(dy, dx)

        if self.prev_theta is None:
            dtheta = 0.0
        else:
            dtheta = abs(self._wrap_pi(theta - self.prev_theta))

        self.prev_theta = theta

        self.turn_ema = (1.0 - self.turn_alpha) * self.turn_ema + self.turn_alpha * dtheta

        d_lo = np.deg2rad(0.5)   # basically straight
        d_hi = np.deg2rad(6.0)   # clearly turning
        t = (self.turn_ema - d_lo) / (d_hi - d_lo + 1e-9)
        t = float(np.clip(t, 0.0, 1.0))

        self.process_std = self.process_std_base + t * (self.process_std_max - self.process_std_base)
        self.meas_std    = self.meas_std_base    + t * (self.meas_std_min  - self.meas_std_base)

    def predict(self):
        noise = np.random.randn(self.N, 2) * self.process_std
        self.particles += noise

    def update(self, z):
        if z is None:
            return

        z = np.asarray(z, dtype=np.float32)
        diff = self.particles - z  # shape (N, 2)

        var = self.meas_std ** 2
        exponent = -0.5 * (
            (diff[:, 0] ** 2) / var[0] +
            (diff[:, 1] ** 2) / var[1]
        )
        likelihood = np.exp(exponent)

        self.weights *= likelihood
        self.weights += 1e-300  # avoid zeros
        self.weights /= np.sum(self.weights)

    def neff(self):
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self):
        # Systematic resampling.
        N = self.N
        positions = (np.arange(N) + np.random.rand()) / N
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # avoid rounding errors
        indexes = np.searchsorted(cumulative_sum, positions)

        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / N)

    def estimate(self):
        # Clean up any non-finite weights
        if not np.all(np.isfinite(self.weights)):
            self.weights = np.ones(self.N, dtype=np.float32) / self.N

        wsum = np.sum(self.weights)

        # If sum of weights is zero or non finite, fall back to simple mean
        if wsum <= 0 or not np.isfinite(wsum):
            est = np.mean(self.particles, axis=0)
        else:
            # ensure weights sum to 1 exactly
            self.weights /= wsum
            est = np.average(self.particles, weights=self.weights, axis=0)

        # Final sanity check
        if not np.all(np.isfinite(est)):
            est = np.mean(self.particles, axis=0)

        return est



def pick_lane_measurement(
    lines,
    frame_shape,
    want_right_lane=True,
    prior=None,                 # np.array([x_bottom, x_top]) from PF estimate/prediction
    gate_px=(80.0, 80.0),       # (gate_bottom_px, gate_top_px)
    side_band=(0.45, 0.55),     # kept for compatibility, but used differently now
):

    if lines is None:
        h, w = frame_shape[:2]
        y_bottom = int(0.95 * h)
        y_top    = int(0.60 * h)
        return None, (y_bottom, y_top)

    h, w = frame_shape[:2]
    y_bottom = int(0.95 * h)
    y_top    = int(0.60 * h)

    gate_b, gate_t = float(gate_px[0]), float(gate_px[1])

    # Heuristic bands for bottom x in pixels
    # These numbers matter much more than the side_band args now.
    center = 0.5 * w

    if prior is None:
        if want_right_lane:
            x_min_init = 0.52 * w
            x_max_init = 0.80 * w 
        else:
            x_min_init = 0.20 * w   
            x_max_init = 0.48 * w   
    else:
        # When tracking, allow a bit more freedom around the prior
        if want_right_lane:
            x_min_init = max(0.50 * w, prior[0] - 60.0)
            x_max_init = min(0.90 * w, prior[0] + 60.0)
        else:
            x_min_init = max(0.10 * w, prior[0] - 60.0)
            x_max_init = min(0.50 * w, prior[0] + 60.0)

    candidates = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        m = (y2 - y1) / (x2 - x1)
        if abs(m) < 1e-3:
            continue

        # slope sign: right lane ~ positive, left lane ~ negative
        if want_right_lane and m <= 0:
            continue
        if (not want_right_lane) and m >= 0:
            continue

        b = y1 - m * x1

        x_bottom = (y_bottom - b) / m
        x_top    = (y_top    - b) / m

        if x_bottom < -0.2*w or x_bottom > 1.2*w:
            continue
        if x_top < -0.2*w or x_top > 1.2*w:
            continue

        if not (x_min_init <= x_bottom <= x_max_init):
            continue

        # PF gating (only when we have a prior)
        if prior is not None:
            dx_b = abs(x_bottom - float(prior[0]))
            dx_t = abs(x_top    - float(prior[1]))
            if dx_b > gate_b or dx_t > gate_t:
                continue

        # scoring: prefer longer lines and ones closer to our expected x-band
        length = np.hypot(x2 - x1, y2 - y1)

        # instead of center bias, bias towards middle of the [x_min_init, x_max_init] band
        band_mid = 0.5 * (x_min_init + x_max_init)
        band_span = max(x_max_init - x_min_init, 1.0)
        band_bias = 1.0 - abs(x_bottom - band_mid) / band_span
        band_bias = max(band_bias, 0.1)

        score = length * band_bias
        candidates.append((score, x_bottom, x_top))

    if not candidates:
        return None, (y_bottom, y_top)

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x_bottom, x_top = candidates[0]

    z = np.array([x_bottom, x_top], dtype=np.float32)
    return z, (y_bottom, y_top)
