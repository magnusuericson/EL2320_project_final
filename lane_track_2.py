
#!/usr/bin/env python3
import argparse
import cv2
import numpy as np

from preprocess import process_frame
from particle_filter import ParticleFilter, pick_lane_measurement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video (e.g. mp4)")
    parser.add_argument("--N", type=int, default=500, help="Particles per lane PF")
    parser.add_argument("--no-left", action="store_true", help="Track only right lane")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    pf_right = None
    pf_left = None

    init_std = np.array([60.0, 60.0], dtype=np.float32)
    gate_px  = (100.0, 100.0)
    side_band = (0.45, 0.55)
    lane_switch_thresh = 100.0  

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_with_lines, edges_roi, lines = process_frame(frame)
        h, w = frame.shape[:2]

        # RIGHT LANE 
        prior_r = None if pf_right is None else pf_right.estimate()

        z_right, (y_bottom_r, y_top_r) = pick_lane_measurement(
            lines,
            frame.shape,
            want_right_lane=True,
            prior=prior_r,
            gate_px=gate_px,
            side_band=side_band,
        )

        if pf_right is None and z_right is not None:
            pf_right = ParticleFilter(N=args.N, init_measurement=z_right, init_std=init_std)
            prior_r = z_right

        if pf_right is not None:
            if z_right is not None and prior_r is not None:
                dx = abs(z_right[0] - prior_r[0])
                if dx > lane_switch_thresh:
                    pf_right = ParticleFilter(N=args.N, init_measurement=z_right, init_std=init_std)

            pf_right.adapt_noise(z_right, y_bottom_r, y_top_r)
            pf_right.predict()
            pf_right.update(z_right)

            if pf_right.neff() < pf_right.N / 2:
                pf_right.resample()

            est_r = pf_right.estimate()
            if est_r is not None and np.all(np.isfinite(est_r)):
                xb, xt = float(est_r[0]), float(est_r[1])
                cv2.line(
                    frame_with_lines,
                    (int(xb), int(y_bottom_r)),
                    (int(xt), int(y_top_r)),
                    (0, 255, 0),
                    5,
                )
                cv2.putText(
                    frame_with_lines,
                    "RIGHT (PF)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # LEFT LANE 
        if not args.no_left:
            prior_l = None if pf_left is None else pf_left.estimate()

            z_left, (y_bottom_l, y_top_l) = pick_lane_measurement(
                lines,
                frame.shape,
                want_right_lane=False,
                prior=prior_l,
                gate_px=gate_px,
                side_band=side_band,
            )

            if pf_left is None and z_left is not None:
                pf_left = ParticleFilter(N=args.N, init_measurement=z_left, init_std=init_std)
                prior_l = z_left

            if pf_left is not None:
                if z_left is not None and prior_l is not None:
                    dx = abs(z_left[0] - prior_l[0])
                    if dx > lane_switch_thresh:
                        pf_left = ParticleFilter(N=args.N, init_measurement=z_left, init_std=init_std)

                pf_left.adapt_noise(z_left, y_bottom_l, y_top_l)
                pf_left.predict()
                pf_left.update(z_left)

                if pf_left.neff() < pf_left.N / 2:
                    pf_left.resample()

                est_l = pf_left.estimate()
                if est_l is not None and np.all(np.isfinite(est_l)):
                    xb, xt = float(est_l[0]), float(est_l[1])
                    cv2.line(
                        frame_with_lines,
                        (int(xb), int(y_bottom_l)),
                        (int(xt), int(y_top_l)),
                        (255, 0, 0),
                        5,
                    )
                    cv2.putText(
                        frame_with_lines,
                        "LEFT (PF)",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

        cv2.imshow("Lane detection (Hough + PF)", frame_with_lines)
        cv2.imshow("Edges + ROI", edges_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
