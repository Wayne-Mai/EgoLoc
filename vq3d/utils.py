import cv2

def format_frames(frames):
    # frame_number -> fno
    # add x2, y2
    for frame in frames:
        frame['fno'] = frame.pop('frame_number')
        frame['x1'] = int(frame['x'])
        frame['y1'] = int(frame['y'])
        frame['x2'] = int(frame['x'] + frame.pop('width'))
        frame['y2'] = int(frame['y'] + frame.pop('height'))
    return frames

def scale_im_height(image, H):
    im_H, im_W = image.shape[:2]
    W = int(1.0 * H * im_W / im_H)
    return cv2.resize(image, (W, H))


def _get_box(annot_box):
    x, y, w, h = annot_box["x"], annot_box["y"], annot_box["width"], annot_box[
        "height"]
    return (int(x), int(y), int(x + w), int(y + h))

