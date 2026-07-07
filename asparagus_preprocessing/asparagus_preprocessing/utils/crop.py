def crop_to_box(array, bbox):
    """
    Crops an array to the Bounding Box indices
    Should be a list of [xmin, xmax, ymin, ymax (, zmin, zmax)]

    We add +1 because slicing excludes the high val index. Which it should not do here.
    """
    if len(bbox) > 5:
        bbox_slices = (
            slice(bbox[0], bbox[1] + 1),
            slice(bbox[2], bbox[3] + 1),
            slice(bbox[4], bbox[5] + 1),
        )
    else:
        bbox_slices = (slice(bbox[0], bbox[1] + 1), slice(bbox[2], bbox[3] + 1))
    return array[bbox_slices]
