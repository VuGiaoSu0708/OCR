def sort_table(bboxes, row_height_threshold=20):
    # Group into rows based on y-coordinates
    rows = {}
    for box in bboxes:
        y_center = box[1] + box[3]/2
        assigned = False
        for row_y in rows.keys():
            if abs(row_y - y_center) < row_height_threshold:
                rows[row_y].append(box)
                assigned = True
                break
        if not assigned:
            rows[y_center] = [box]
    
    # Sort rows by y and entries within rows by x
    result = []
    for row_y in sorted(rows.keys()):
        result.extend(sorted(rows[row_y], key=lambda box: box[0]))
    return result
