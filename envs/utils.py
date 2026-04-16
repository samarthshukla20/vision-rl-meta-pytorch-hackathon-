def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    boxA, boxB: list or numpy arrays in the format [x_min, y_min, width, height]
    
    Returns:
    float: IoU score between 0.0 and 1.0
    """
    
    # 1. Convert [x, y, w, h] to [x_min, y_min, x_max, y_max] for easier intersection math
    xA_min, yA_min = boxA[0], boxA[1]
    xA_max, yA_max = boxA[0] + boxA[2], boxA[1] + boxA[3]
    
    xB_min, yB_min = boxB[0], boxB[1]
    xB_max, yB_max = boxB[0] + boxB[2], boxB[1] + boxB[3]

    # 2. Find the coordinates of the intersecting rectangle
    inter_x_min = max(xA_min, xB_min)
    inter_y_min = max(yA_min, yB_min)
    inter_x_max = min(xA_max, xB_max)
    inter_y_max = min(yA_max, yB_max)

    # 3. Calculate intersection area 
    # max(0, ...) ensures that if the boxes don't overlap, the area is 0 (not negative)
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # 4. Calculate the area of both individual bounding boxes
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    
    # 5. Calculate union area (Area A + Area B - Intersection)
    union_area = float(boxA_area + boxB_area - inter_area)

    # 6. Safety check to prevent division by zero in degenerate cases
    if union_area <= 0:
        return 0.0

    # 7. Return the final IoU score
    return inter_area / union_area