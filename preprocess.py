
#!/usr/bin/env python3
import cv2
import numpy as np



def lane_color_mask(frame_bgr):

    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    # # White: bright + a bit of saturation to avoid pure glare
    # white = (L >= 200) & (S >= 40)
    #
    # # Yellow: hue band for yellow in OpenCV HLS (0..179)
    # # Typical yellow ~ 15..40, but depends on camera.
    # yellow = (H >= 15) & (H <= 40) & (S >= 80) & (L >= 120)


    white = (L >= 180) & (S >= 20)
    yellow = (H >= 10) & (H <= 45) & (S >= 60) & (L >= 100)

    mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
    mask[white | yellow] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def make_roi_mask(image_shape):
    """Return a trapezoid ROI mask for the bottom part of the image."""
    h, w = image_shape[:2]

    # Tune these four points if needed
    # bottom_left  = (int(0.1 * w), h)
    # bottom_right = (int(0.9 * w), h)
    # top_right    = (int(0.6 * w), int(0.6 * h))
    # top_left     = (int(0.4 * w), int(0.6 * h))

    # bottom_left  = (int(0.15 * w), h)
    # bottom_right = (int(0.65 * w), h)
    # top_right    = (int(0.55 * w), int(0.6 * h))
    # top_left     = (int(0.45 * w), int(0.6 * h))

    bottom_left  = (int(0.15 * w), h)
    bottom_right = (int(0.65 * w), h)
    top_right    = (int(0.55 * w), int(0.65 * h))
    top_left     = (int(0.45 * w), int(0.65 * h))

    # bottom_left  = (0.20 * w, h)
    # bottom_right = (0.70 * w, h)
    # top_right    = (0.60 * w, int(0.6 * h))
    # top_left     = (0.50 * w, int(0.6 * h))

    # bottom_left  = (0.30 * w, h)
    # bottom_right = (0.60 * w, h)
    # top_right    = (0.55 * w, int(0.6 * h))
    # top_left     = (0.45 * w, int(0.6 * h))

    polygon = np.array([[bottom_left, bottom_right, top_right, top_left]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 255)
    return mask

def make_rect_mask(image_shape, x1_frac=0.15, x2_frac=0.85, y1_frac=0.65, y2_frac=1.0):

    h, w = image_shape[:2]

    x1 = int(x1_frac * w)
    x2 = int(x2_frac * w)
    y1 = int(y1_frac * h)
    y2 = int(y2_frac * h)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask


def make_trapezoid_mask(image_shape,
                        bottom_left_frac=0.10,   # wider bottom
                        bottom_right_frac=0.90,
                        top_left_frac=0.40,      # narrower top
                        top_right_frac=0.60,
                        top_y_frac=0.60):        # how high the top edge is (0 = top of image)

    h, w = image_shape[:2]

    # Compute pixel coords
    bl = (int(bottom_left_frac * w), h)
    br = (int(bottom_right_frac * w), h)
    tl = (int(top_left_frac    * w), int(top_y_frac * h))
    tr = (int(top_right_frac   * w), int(top_y_frac * h))

    pts = np.array([tl, tr, br, bl], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)

    return mask



def process_frame(frame):


    #  Grayscale
    # gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    # 3) Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("blur", blur)

    #  Canny edges
    edges = cv2.Canny(blur, 50, 150)

    # cv2.imshow("mask", paint_mask)
    # cv2.imshow("masked_gray", gray)
    cv2.imshow("canny", edges)

    #  ROI mask
    # roi = make_roi_mask(edges.shape)
    # edges_roi = cv2.bitwise_and(edges, roi)

    # roi = make_rect_mask(edges.shape, x1_frac=0.15, x2_frac=0.85, y1_frac=0.65, y2_frac=1.0)
    # edges_roi = cv2.bitwise_and(edges, roi)

    roi = make_trapezoid_mask(
                                edges.shape, 
                                bottom_left_frac=0.10,
                                bottom_right_frac=0.90,
                                top_left_frac=0.45,
                                top_right_frac=0.55,
                                top_y_frac=0.65
                            )
    edges_roi = cv2.bitwise_and(edges, roi)

    #  Hough transform
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=120,
        maxLineGap=30,
    )

 
    frame_with_lines = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return frame_with_lines, edges_roi, lines


