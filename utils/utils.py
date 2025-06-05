import cv2
import numpy as np


def gravity_align(
    img,
    r,
    p,
    K=np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]]).astype(np.float32),
    mode=0,
):
    """
    Align the image with gravity direction
    Input:
        img: input image
        r: roll
        p: pitch
        K: camera intrisics
        mode: interpolation mode for warping, default: 0 - 'linear', else 1 - 'nearest'
    Output:
        aligned_img: gravity aligned image
    """
    # calculate R_gc from roll and pitch
    # From gravity to camera, yaw->pitch->roll
    # From camera to gravity, roll->pitch->yaw
    p = (
        -p
    )  # this is because the pitch axis of robot and camera is in the opposite direction
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)

    # compute R_cg first
    # pitch
    R_x = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])

    # roll
    R_z = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])

    R_cg = R_z @ R_x
    R_gc = R_cg.T

    # get shape
    h, w = list(img.shape[:2])

    # directly compute the homography
    persp_M = K @ R_gc @ np.linalg.inv(K)

    aligned_img = cv2.warpPerspective(
        img, persp_M, (w, h), flags=cv2.INTER_NEAREST if mode == 1 else cv2.INTER_LINEAR
    )

    return aligned_img