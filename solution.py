from copy import copy
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter

FOCAL_LEN = 910

def load_video(number, color_code=cv2.COLOR_BGR2RGB):
    path = f"D:/projects/calib_challenge/labeled/{number}.hevc"
    cap = cv2.VideoCapture(path)

    frames = []
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, color_code)
            frames.append(img)
    video = np.stack(frames, axis=0)

    return video

def load_labels(number):
    path = f"D:/projects/calib_challenge/labeled/{number}.txt"
    labels = np.loadtxt(path)
    return labels

def abs_diff(a, b):
    return np.absolute(np.subtract(a, b, dtype=np.int16))

def slice_around(x, r):
    return slice(x - r, x + r + 1)

# x, y = top left corner of slice
def find_slice_in_frame_new(frame, slice, x, y, search_radius):
    pad_x, pad_y = slice.shape[1] + search_radius, slice.shape[0] + search_radius
    slice_w, slice_h = slice.shape[1], slice.shape[0]

    padded_frame = np.pad(frame, ((pad_y, pad_y), (pad_x, pad_x)), "symmetric")

    res = np.empty((search_radius * 2 + 1, search_radius * 2 + 1))

    for i in range(-search_radius, search_radius + 1):
        for j in range(-search_radius, search_radius + 1):
            Y = y + i + pad_y
            X = x + j + pad_x
            search_slice = padded_frame[Y:Y + slice_h, X:X + slice_w]
            res[i + search_radius, j + search_radius] = np.mean(abs_diff(slice, search_slice))
    loc = np.unravel_index(res.argmin(), res.shape)
    dy, dx = loc[0] - search_radius, loc[1] - search_radius
    return dx, dy, res

def build_line(slope, intercept, width, height):
    points = []

    if slope != math.inf:
        line_x = lambda x: slope * x + intercept
        for x in range(width):
            y = line_x(x)
            if 0 <= y < height:
                points.append([x, y])

    if slope != 0:
        line_y = lambda y: (y - intercept) / slope
        for y in range(height):
            x = line_y(y)
            if 0 <= x < width:
                points.append([x, y])

    return np.array(points)

def clamp(val, lower, upper):
    if val < lower: return lower
    elif val > upper: return upper
    else: return val
    
def find_movement_of_slice_new(frames, slice, x, y, search_radius):
    shape = frames[0].shape
    slice_shape = slice.shape
    height, width = shape[0], shape[1]
    slice_height, slice_width = slice_shape[0], slice_shape[1]
    adjust_x, adjust_y = (slice_width - 1) / 2, (slice_height - 1) / 2
    points = [[x + adjust_x, y + adjust_y]]
    for source_frame, search_frame in zip(frames, frames[1:]):
        slice = source_frame[y:y + slice_height, x:x + slice_width]
        dx, dy, _ = find_slice_in_frame_new(search_frame, slice, x, y, search_radius)
        x, y = clamp(x + dx, 0, width - 1 - slice_width), clamp(y + dy, 0, height - 1 - slice_height)
        points.append([x + adjust_x, y + adjust_y])
    points = np.array(points)

    try:
        line = linregress(points)
    except ValueError:
        line = None

    return line, points

def diff_to_angle(diff, focal_length=FOCAL_LEN):
    return 2 * np.arctan(diff / (2 * focal_length))

def find_best_slices(frame, partitions, slices_per_partition, pick_best_per_partition):
    part_x, part_y = partitions
    slices_x, slices_y = slices_per_partition
    result_slices = []
    y = 0
    for row in np.array_split(frame, part_y, axis=0):
        x = 0
        for partition in np.array_split(row, part_x, axis=1):
            slices = []
            sy = y
            for part_row in np.array_split(partition, slices_y, axis=0):
                sx = x
                for slice in np.array_split(part_row, slices_x, axis=1):
                    slices.append({ "slice": slice, "x": sx, "y": sy })

                    sx = sx + slice.shape[1]
                sy = sy + part_row.shape[0]

            result_slices.extend(sorted(slices, key=lambda s: s["slice"].std(), reverse=True)[0:pick_best_per_partition])

            x = x + partition.shape[1]
        y = y + row.shape[0]
    return result_slices

def evaluate_path(points):
    diffs = []
    for p0, p1 in zip(points, points[1:]):
        diffs.append(p1 - p0)

    diffs = np.array(diffs)

    return np.mean(np.std(diffs, axis=0))

def find_pitch_yaw_new(video, index, frame_count, search_radius, partitions, slices_per_partition, pick_best_per_partition):
    width, height = video.shape[2], video.shape[1]

    video = video[:,:int(height - 300),:]

    frame = video[index]
    frames = video[max(index - frame_count + 1, 0):index + 1]

    slices = find_best_slices(frame, partitions, slices_per_partition, pick_best_per_partition)

    canvas = np.zeros((height, width))

    results = []
    for slice in slices:
        line, points = find_movement_of_slice_new(np.flip(frames, axis=0), slice["slice"], slice["x"], slice["y"], search_radius)
        results.append((line, points))

    results = list(filter(lambda res: res[0] is not None, results))
    results = sorted(results, key=lambda res: evaluate_path(res[1]))#[0: int(len(results) / 2)]

    for line, points in results:
        for x, y in build_line(line.slope, line.intercept, width, height):
            canvas[int(y), int(x)] += 1

    res = gaussian_filter(canvas, sigma=5)
    y, x = np.unravel_index(res.argmax(), res.shape)
    pitch, yaw = diff_to_angle(height / 2 - y), diff_to_angle(x - width / 2)

    dy = y - height / 2
    dx = x - width / 2
    return pitch, yaw, dy, dx, y, x, canvas, res

object = {
    "box": (0, 0, 4, 4),
    "slice": np.array([[1, 1]]),
    "points": np.array([[0, 0], [1, 1]])
}

def std_of_slices_in_frame(frame, w, h):
    width, height = frame.shape[1], frame.shape[0]
    result = np.empty((height - h + 1, width - w + 1))

    for y in range(height - h + 1):
        for x in range(width - w + 1):
            result[y, x] = np.std(frame[y:y+h, x:x+w])
    return result
        
def inside_box(box, x, y):
    bx, by, w, h = box
    return x >= bx and y >= by and x < bx + w and y < by + h

def split_num_into_similar_ints(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range (div)]

def divide_box_into_boxes(width, height, x, y):
    boxes = []
    cum_w = 0
    for w in split_num_into_similar_ints(width, x):
        cum_h = 0
        for h in split_num_into_similar_ints(height, y):
            boxes.append([cum_w, cum_h, w, h])
            cum_h += h
        cum_w += w
    return boxes

def find_good_slice(frame, objects, w, h):
    boxes = divide_box_into_boxes(frame.shape[1] - w + 1, frame.shape[0] - h + 1, 8, 4)
    boxes_number = list(map(lambda box: { "box": box, "number": 0 }, boxes))

    for box_number in boxes_number:
        for object in objects:
            if inside_box(box_number["box"], object["box"][0], object["box"][1]):
                box_number["number"] += 1

    emptiest_box = sorted(boxes_number, key=lambda box_number: box_number["number"])[0]["box"]
    bx, by, bw, bh = emptiest_box

    inner_boxes = divide_box_into_boxes(bw, bh, bw // w, bh // h)

    def box_w_std(box):
        return { 
            "box": box, 
            "std": np.std(frame[by+box[1]:by+box[1]+h, bx+box[0]:bx+box[0]+w]) 
        }

    boxes_with_std = map(box_w_std , inner_boxes)
    for box_with_std in sorted(boxes_with_std, key=lambda o: o["std"], reverse=True):
        x, y = box_with_std["box"][0], box_with_std["box"][1]
        good = True
        for object in objects:
            if abs(bx + x - object["box"][0]) < w and abs(by + y - object["box"][1]) < h:
                good = False
                break

        if not good:
            continue

        return {
            "box": (bx + x, by + y, w, h),
            "slice": frame[by + y:by + y + h, bx + x:bx + x + w],
            "points": np.array([[bx + x, by + y]])
        }

    return None

def find_best_objects(frame, num, old_objects, w, h):
    frame = frame[:-250]
    objects = old_objects
    # for _ in range(num):
    #     object = find_good_slice(frame, objects, w, h)

    #     if object is None:
    #         return objects
    
    #     objects.append(object)

    corners = cv2.goodFeaturesToTrack(frame, num, 0.005, 2 * w)

    for corner in corners:
        x, y = corner.ravel()

        if 500 <= x <= 700 and 300 <= y <= 500:
            continue

        rx = max(x - w // 2, 0)
        adjx = max(rx + w - frame.shape[1], 0)
        rx = int(rx - adjx)

        ry = max(y - h // 2, 0)
        adjy = max(ry + h - frame.shape[0], 0)
        ry = int(ry - adjy)

        objects.append({
            "box": (rx, ry, w, h),
            "slice": frame[ry:ry+h, rx:rx+w],
            "points": np.array([[rx, ry]])
        })

    return objects

def in_bounds(object, width, height):
    x, y, w, h = object["box"]
    out_of_bounds = x < 0 or y < 0 or x + w > width or y + h > height
    return not out_of_bounds

def analyze_video(video, search_radius, number_objects, object_w, object_h):
    # slices = find_best_slices(video[0], partitions, slices_per_partition, pick_best_per_partition)

    # def slice_to_obj(slice):
    #     return { 
    #         "box": (slice["x"], slice["y"], slice["slice"].shape[1], slice["slice"].shape[0]),
    #         "slice": slice["slice"],
    #         "points": np.array([slice["x"], slice["y"]])
    #     } 

    objects = find_best_objects(video[0], number_objects, [], object_w, object_h)

    width, height = video.shape[2], video.shape[1]

    extra_info = []
    pitches_yaws = []
    for frame in video[1:]:
        add = number_objects - len(objects)
        print("Add", add)
        # if add > 0:
        #     objects = find_best_objects(frame, add, objects, object_w, object_h)

        for object in objects:
            dx, dy, _ = find_slice_in_frame_new(frame, object["slice"], object["box"][0], object["box"][1], search_radius)

            x, y, w, h = object["box"]
            x = x + dx
            y = y + dy
            object["box"] = (x, y, w, h)
            object["slice"] = frame[y:y+h, x:x+w]
            object["points"] = np.concatenate([np.array([[x, y]]), object["points"]], axis=0)

        lines = []
        for object in filter(lambda o: o["points"].shape[0] >= 5, objects):
            try:
                line = linregress(object["points"][:20])
            except ValueError:
                line = None

            if line is None:
                continue

            lines.append((line, object["points"][:20]))


        canvas = np.zeros((frame.shape[0], frame.shape[1]))
        for line, _ in sorted(lines, key=lambda res: evaluate_path(res[1]))[0: int(len(lines) / 2)]:#[0: int(len(results) / 2)]
            for x, y in build_line(line.slope, line.intercept, width, height):
                canvas[int(y), int(x)] += 1

        res = gaussian_filter(canvas, sigma=5)
        y, x = np.unravel_index(res.argmax(), res.shape)
        pitch, yaw = diff_to_angle(height / 2 - y), diff_to_angle(x - width / 2)

        dy = y - height / 2
        dx = x - width / 2

        extra_info.append((yaw, dy, dx, y, x, canvas, res, objects))
        pitches_yaws.append([pitch, yaw])

        objects = list(filter(lambda o: in_bounds(o, width, height), objects))
    
    return np.array(pitches_yaws), extra_info









if __name__ == "__main__":
    np.seterr(invalid='raise')

    video = load_video(2, cv2.COLOR_BGR2GRAY)
    labels = load_labels(2)

    objs = find_best_objects(video[100], 10, [], 40, 40)
    plt.imshow(video[100])
    objs