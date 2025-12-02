import numpy as np
import cv2


def get_bboxes_from_mask(mask, threshold=0.95, min_area=100):
    # Ensure mask is properly converted to uint8 for findContours
    mask_binary = (mask > threshold).astype(np.uint8) * 255
    # OpenCV findContours requires contiguous array
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append([x, y, w, h])
    return bboxes


def enhance_mask(mask, min_threshold=0.5):
    binary = (mask > min_threshold).astype(np.uint8) * 255
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    kernel = np.ones((3, 2), np.uint8)
    local_max = cv2.dilate(dist, kernel)
    markers = ((dist == local_max) & (dist > 0.7 * dist.max())).astype(np.uint8)
    _, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    markers[binary == 0] = 0
    cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)
    result = binary.copy()
    result[markers == -1] = 0
    return result
