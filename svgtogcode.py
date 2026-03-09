from dataclasses import dataclass
from itertools import chain
from typing import Any

import lxml.etree as ET
import math
from datetime import datetime
import sys
import argparse
import shapely

import numpy as np
from shapely import MultiLineString, LineString, Point

from src.util.misc import linestring_to_coordinate_pairs

DEFAULT_MAX_LENGTH_SEGMENT = 4 # 20  # in m

OFFSET = [0, 0]

TRAVEL_SPEED = 6000
WRITE_SPEED = 1000
PEN_LIFT_SPEED = 5000
PEN_DIP_SPEED = 500

PEN_UP_DISTANCE = 3.0
PEN_DIP_UP_DISTANCE = 7.0
PEN_DIP_DOWN_DISTANCE = 2.0

WAIT_INIT = 5000

DIP_LOCATION = [0, -20]  # Dip mode 1: reservoir placed at dip location
DIP_LOCATION = [-20, None]  # Dip mode 2: reservoir mounted on X gantry
DIP_DISTANCE = 180

CMD_MOVE = "G1 X{0:.3f} Y{1:.3f}\n"
CMD_MOVE_X = "G1 X{0:.3f}\n"
CMD_MOVE_Y = "G1 Y{1:.3f}\n"
CMD_PEN_UP = "G1 Z{} F{}\n".format(PEN_UP_DISTANCE, PEN_LIFT_SPEED)
CMD_PEN_DIP_UP = "G1 Z{} F{}\n".format(PEN_DIP_UP_DISTANCE, PEN_LIFT_SPEED)
CMD_PEN_DIP_DOWN = "G1 Z{} F{}\n".format(PEN_DIP_DOWN_DISTANCE, PEN_DIP_SPEED)

OPTIMIZE_ORDER = True

# np.set_printoptions(precision=4,
#                        threshold=10000,
#                        linewidth=150)

np.set_printoptions(suppress=True)


@dataclass
class SvgToGcodeConfig:
    comp_tolerance: float = 0.20
    min_line_length: float = 0.10  # in mm


def process_count(e: Any, default_namespace: str) -> int:
    if e.tag == default_namespace + "rect":
        return 4

    if e.tag == default_namespace + "line":
        return 1

    if e.tag == default_namespace + "path":
        d = e.attrib["d"]
        d = d[1:]  # cut off the M
        return len(d.split("L")) - 1

    return 0


def process(e: Any, default_namespace: str) -> list[tuple[float, float, float, float]]:
    lines = []

    if e.tag == default_namespace + "rect":
        x1 = float(e.attrib["x"])
        y1 = float(e.attrib["y"])
        x2 = x1 + float(e.attrib["width"])
        y2 = y1 + float(e.attrib["height"])

        lines.append([x1, y1, x1, y2])  # left
        lines.append([x1, y1, x2, y1])  # top
        lines.append([x1, y2, x2, y2])  # bottom
        lines.append([x2, y1, x2, y2])  # right

        return lines

    if e.tag == default_namespace + "line":
        lines.append(
            [
                float(e.attrib["x1"]),
                float(e.attrib["y1"]),
                float(e.attrib["x2"]),
                float(e.attrib["y2"]),
            ]
        )
        return lines

    if e.tag == default_namespace + "path":
        d = e.attrib["d"]

        # cut off the M
        d = d[1:]

        # cut off the Z
        closed = True if d.endswith("Z") else False
        if closed:
            d = d[:-1]

        segments = d.split("L")
        l = []

        for s in segments:
            pairs = s.strip().split(" ")
            l.append([float(pairs[0]), float(pairs[1])])

        if closed:
            l.append(l[0])

        for i in range(1, len(l)):
            lines.append([l[i - 1][0], l[i - 1][1], l[i][0], l[i][1]])

        return lines

    if e.tag == default_namespace + "circle":
        print("invalid element: {}".format(e.tag))
        return lines

    if e.tag == default_namespace + "image":
        print("invalid element: {}".format(e.tag))
        return lines

    print("unknown element: {}".format(e.tag))
    return lines


def compare_equal(e0, e1, config: SvgToGcodeConfig):
    if math.isclose(e0[0], e1[0], abs_tol=config.comp_tolerance):
        if math.isclose(e0[1], e1[1], abs_tol=config.comp_tolerance):
            return True

    return False


# from: https://stackoverflow.com/a/34325723
def printProgressBar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def filter_linestrings(g: shapely.Geometry) -> list[LineString]:
    match g:
        case Point():
            return []

        case LineString():
            return [g]

        case MultiLineString():
            result = []
            for sub_geometry in g.geoms:
                result += filter_linestrings(sub_geometry)
            return result

        case _:
            print(f"cropping: unexpected shapely geometry: {type(g)}")
            return []


def _dip_pen(out: Any) -> None:

    out.write(CMD_PEN_DIP_UP)
    out.write(f"G1 F{TRAVEL_SPEED}\n")

    match DIP_LOCATION:
        case None, None:
            raise Exception(f"invalid DIP_LOCATION {DIP_LOCATION}")
        case None, dip_y:
            out.write(CMD_MOVE_Y.format(dip_y))
        case dip_x, None:
            out.write(CMD_MOVE_X.format(dip_x))
        case dip_x, dip_y:
            out.write(CMD_MOVE.format(dip_x, dip_y))

    out.write(CMD_PEN_DIP_DOWN)
    out.write(CMD_PEN_DIP_UP)
    out.write(CMD_PEN_DIP_DOWN)
    out.write(CMD_PEN_DIP_UP)

    out.write(f"G1 F{TRAVEL_SPEED}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename")
    parser.add_argument("--crop", nargs="*", type=int, help="crop: center-X center-Y width height")
    parser.add_argument(
        "--max-length-segment",
        type=int,
        default=DEFAULT_MAX_LENGTH_SEGMENT,
        help="maximum length of segment [m]",
    )
    parser.add_argument("--filter-layer", type=str, default=None, help="filter layers by name")
    parser.add_argument("--high-precision", action="store_true", help="high precision mode")
    parser.add_argument("--limit", type=int, default=0, help="process only the first n lines")
    parser.add_argument("--dip-mode", default=False, action="store_true", help="enable dip pen mode (reservoir placed at DIP_LOCATION)")
    parser.add_argument("--rotate90", default=False, action="store_true", help="rotate by 90 degrees")
    args = parser.parse_args()

    config = SvgToGcodeConfig()

    crop_region = None
    if args.crop is not None:
        if len(args.crop) != 4:
            print(f"invalid crop arguments: {args.crop}. Must be exactly four int values: [center-X, center-Y, widht, height]")
            sys.exit(2)

        crop_region = shapely.box(
            args.crop[0] - args.crop[2] // 2,
            args.crop[1] - args.crop[3] // 2,
            args.crop[0] + args.crop[2] // 2,
            args.crop[1] + args.crop[3] // 2,
        )

    args.max_length_segment *= 100 * 10  # m to mm

    tree = ET.parse(args.input_filename)
    root = tree.getroot()
    svg_default_namespace = "{" + root.nsmap[None] + "}"
    svg_inkscape_namespace = "{" + root.nsmap["inkscape"] + "}"

    output_filename = f"map_layer_{args.filter_layer}"

    if args.dip_mode:
        output_filename += "_dip"

    size = [root.get("width"), root.get("height")]
    for i, dim in enumerate(size):
        if dim.endswith("px") or dim.endswith("mm"):
            dim = dim[:-2]

        dim = int(dim)

        if dim is None or dim <= 0:
            print(f"SVG size attributes not correct ({size})")
            exit(-1)

        size[i] = dim

    if args.high_precision:
        print("set to high precision mode")
        config.comp_tolerance = 0.001
        config.min_line_length = 0.1

    all_lines: list[list[tuple[float, float, float, float]]] = []

    for layer in root.findall("g", root.nsmap):
        if args.filter_layer is not None:
            if layer.attrib["id"] != args.filter_layer:
                print(f"skip layer {layer.attrib['id']}")
                continue

        line_count = 0
        for child in layer:
            line_count += process_count(child, svg_default_namespace)

        for i, child in enumerate(layer):
            if i % 100 == 0:
                printProgressBar(
                    i,
                    len(layer),
                    prefix=f"process layer: {layer.attrib.get('id', 'UNNAMED_LAYER'):<25}",
                )

            all_lines.append(process(child, svg_default_namespace))

        print("")

    if args.limit > 0:
        limit = min(len(all_lines), args.limit)
        print(f"processing limited to {limit} lines")
        all_lines = all_lines[0:limit]

    if crop_region is not None:
        cropped_lines = []
        crop_translation = crop_region.bounds[0:2]

        for i, line in enumerate(all_lines):
            if i % 100 == 0:
                printProgressBar(i, len(all_lines), prefix="cropping")

            coords = [l[0:2] for l in line] + [line[-1][2:4]]
            ls = LineString(coords)
            result = crop_region.intersection(ls)

            if result.is_empty:
                continue

            filtered_linestrings = filter_linestrings(result)
            for ls in filtered_linestrings:
                cropped_lines.append(shapely.affinity.translate(ls, xoff=-crop_translation[0], yoff=-crop_translation[1]))

        print("")

        all_lines = []
        for line in cropped_lines:
            all_lines.append([(pair[0][0], pair[0][1], pair[1][0], pair[1][1]) for pair in linestring_to_coordinate_pairs(line)])

    print(" ")
    print("--------------------------------------------------")
    print(" ")

    number_of_lines = len(all_lines)
    print("number of lines: {}".format(number_of_lines))

    if number_of_lines == 0:
        exit(0)

    # ------------------------------------------------------------------------------------
    # filter duplicates

    # nplines = np.array(all_lines, dtype=float)
    # unique = np.unique(nplines, axis=0)
    #
    # number_duplicates = len(all_lines) - unique.shape[0]
    # print("cleaned duplicates: {0} | duplicate ratio: {1:.2f}%".format(number_duplicates, (number_duplicates / len(all_lines)) * 100))
    #
    # nplines = unique

    # disabled ... need to keep the LineStrings intact (do not treat every single line of a line string individually)

    # ------------------------------------------------------------------------------------
    # filter tiny lines

    # distances = np.sqrt(np.add(np.power(np.subtract(nplines[:, 0], nplines[:, 2]), 2), np.power(np.subtract(nplines[:, 1], nplines[:, 3]), 2)))
    # indices_shortlines = np.where(distances < config.min_line_length)[0]
    # nplines = np.delete(nplines, indices_shortlines, axis=0)
    # print("cleaned short lines: {0} | short line ratio: {1:.2f}%".format(indices_shortlines.shape[0], (indices_shortlines.shape[0]/len(all_lines))*100))

    # evil ... breaks paths without pen up/down events in two and creates gaps

    # ------------------------------------------------------------------------------------
    # optimize drawing order. greedy (and inefficient)

    ordered_lines = None

    if OPTIMIZE_ORDER:
        timer = datetime.now()

        indices_done = []

        # treat each line string as a single line, compute distance only on the start and end point of this line
        # and then order the linestrings based on the indices of these distances
        nplines = np.array([(line[0][0], line[0][1], line[-1][0], line[-1][1]) for line in all_lines])

        indices_done_mask = np.zeros(nplines.shape[0], dtype=bool)
        indices_done_mask[indices_done] = True

        ordered_lines = []
        last = [0, 0, 0, 0]

        for i in range(0, nplines.shape[0]):
            if i % 100 == 0:
                printProgressBar(len(ordered_lines), nplines.shape[0], prefix="optimize order")

            indices_done_mask[indices_done] = True

            # pythagorean distance

            distance_forw = np.sqrt(np.add(np.power(np.subtract(nplines[:, 0], last[2]), 2), np.power(np.subtract(nplines[:, 1], last[3]), 2)))
            distance_forw_masked = np.ma.masked_array(distance_forw, mask=indices_done_mask)
            distance_forw_min = np.argmin(distance_forw_masked)

            distance_back = np.sqrt(np.add(np.power(np.subtract(nplines[:, 2], last[2]), 2), np.power(np.subtract(nplines[:, 3], last[3]), 2)))
            distance_back_masked = np.ma.masked_array(distance_back, mask=indices_done_mask)
            distance_back_min = np.argmin(distance_back_masked)

            if distance_forw[distance_forw_min] < distance_back[distance_back_min]:
                indices_done.append(distance_forw_min)
                ordered_lines.append(all_lines[distance_forw_min])
            else:
                indices_done.append(distance_back_min)
                flipped = [(p[2], p[3], p[0], p[1]) for p in reversed(all_lines[distance_back_min])]
                ordered_lines.append(flipped)

            last = ordered_lines[-1][-1]

        print("")
        print("optimization done. time: {0:.2f}s".format((datetime.now() - timer).total_seconds()))

    else:
        ordered_lines = all_lines

    ordered_lines = list(chain.from_iterable(ordered_lines))
    nplines = np.array(ordered_lines, dtype=float)

    # ------------------------------------------------------------------------------------
    # translate across X-axis to transfer SVG coordinate system (0 top left) to gcode (0 bottom left)

    maxy = size[1]  # np.max([np.max(nplines[:, 1]), np.max(nplines[:, 3])])
    if crop_region is not None:
        maxy = crop_region.bounds[3] - crop_region.bounds[1]
        print(maxy)

    nplines[:, 1] = np.multiply(nplines[:, 1], -1)
    nplines[:, 3] = np.multiply(nplines[:, 3], -1)

    nplines[:, 1] = np.add(nplines[:, 1], maxy)
    nplines[:, 3] = np.add(nplines[:, 3], maxy)

    # ------------------------------------------------------------------------------------
    # filter tiny edges/leaves/whatever (small lines which are not connected)

    distances = np.sqrt(
        np.add(
            np.power(np.subtract(nplines[:, 0], nplines[:, 2]), 2),
            np.power(np.subtract(nplines[:, 1], nplines[:, 3]), 2),
        )
    )
    indices_shortlines = np.where(distances < config.min_line_length)[0]

    unconnected_indices = []
    for i in range(1, nplines.shape[0] - 1):
        prv = nplines[i - 1, :]
        cur = nplines[i, :]
        nxt = nplines[i + 1, :]

        if not prv[2] == cur[0] or not prv[3] == cur[1] or not cur[2] == nxt[0] or not cur[3] == nxt[1]:
            if i in indices_shortlines:
                unconnected_indices.append(i)

    nplines = np.delete(nplines, unconnected_indices, axis=0)
    ordered_lines = nplines

    print(
        "cleaned unconnected short lines: {0} | short line ratio: {1:.2f}%".format(
            len(unconnected_indices), (len(unconnected_indices) / len(all_lines)) * 100
        )
    )

    # ------------------------------------------------------------------------------------

    # ordered_lines = []
    # ordered_lines.append(all_lines[0])
    # all_lines.remove(ordered_lines[0])

    # while(len(all_lines) > 0):

    #     print("{0:.2f}".format((len(ordered_lines)/number_of_lines)*100.0))

    #     src = ordered_lines[-1][1]
    #     dst = all_lines[0]

    #     candidate = dst
    #     candidate_distance = distance(src, candidate[0])
    #     candidateFlip = False

    #     candidate_i = 0

    #     for i in range(0, len(all_lines)):
    #         dst = all_lines[i]

    #         distance0 = distance(src, dst[0])
    #         distance1 = distance(src, dst[1])

    #         if distance0 < 0.001:
    #             candidate = dst
    #             candidateFlip = False
    #             candidate_i = i
    #             break

    #         if distance1 < 0.001:
    #             candidate = dst
    #             candidateFlip = True
    #             candidate_i = i
    #             break

    #         if distance0 < candidate_distance:
    #             candidate = dst
    #             candidate_distance = distance0
    #             candidateFlip = False

    #             candidate_i = i

    #         if distance1 < candidate_distance:
    #             candidate = dst
    #             candidate_distance = distance1
    #             candidateFlip = True

    #             candidate_i = i

    #     # print("{}|{} {}".format(candidate_i, len(all_lines), candidateFlip))

    #     if candidateFlip:
    #         ordered_lines.append([candidate[1], candidate[0]])
    #     else:
    #         ordered_lines.append(candidate)

    #     all_lines.pop(candidate_i)

    # print("number of ordered_lines: {}".format(len(ordered_lines)))

    # print(order_index)

    segments = [[]]
    number_lines = len(ordered_lines)
    total_length_segment = 0
    for i in range(0, number_lines):
        dist = math.sqrt((ordered_lines[i][2] - ordered_lines[i][0]) ** 2 + (ordered_lines[i][3] - ordered_lines[i][1]) ** 2)

        if (dist + total_length_segment) > args.max_length_segment:
            segments.append([])
            print("new segment          [{:5.2f}m]".format(total_length_segment / 1000))
            total_length_segment = 0
        else:
            total_length_segment += dist

        segments[-1].append(ordered_lines[i])

    print("last segment         [{:5.2f}m]".format(total_length_segment / 1000))

    count_pen_up = 0
    count_pen_down = 0
    count_draw_moves = 0
    count_travel_moves = 0
    count_dip_moves = 0

    state_pen_up = True

    for s in range(0, len(segments)):
        segment = segments[s]
        filename = output_filename + f"_{s + 1}of{len(segments)}.nc"

        distance_travelled = 0

        with open(filename, "w") as out:
            out.write("G90\n")  # absolute positioning
            out.write("G21\n")  # Set Units to Millimeters
            out.write(CMD_PEN_UP)  # move pen up
            # out.write(f"G4 P{WAIT_INIT}\n")              # wait before making the first move
            out.write(f"G1 F{TRAVEL_SPEED}\n")  # Set feedrate to TRAVEL_SPEED mm/min
            state_pen_up = True
            out.write("\n")

            count_pen_up += 1
            number_lines = len(segment)

            if args.dip_mode: # initial dip
                _dip_pen(out)
                state_pen_up = True
                count_pen_up += 1
                count_dip_moves += 1

            for i in range(0, number_lines):
                line = segment[i]
                line_next = None
                if (i + 1) < number_lines:
                    line_next = segment[i + 1]

                line_start = [line[0] + OFFSET[0], line[1] + OFFSET[1]]
                if args.rotate90:
                    line_start = [
                        line[1] + OFFSET[1],
                        (line[0] + OFFSET[0]) * -1 + size[0],
                    ]

                out.write(CMD_MOVE.format(*line_start))

                # pen down
                if state_pen_up:
                    count_travel_moves += 1

                    out.write(f"G1 Z0 F{PEN_LIFT_SPEED}\n")
                    out.write(f"G1 F{WRITE_SPEED}\n")
                    state_pen_up = False
                    count_pen_down += 1

                line_end = line[2] + OFFSET[0], line[3] + OFFSET[1]
                if args.rotate90:
                    line_end = line[3] + OFFSET[1], (line[2] + OFFSET[0]) * -1 + size[0]

                out.write(CMD_MOVE.format(*line_end))

                count_draw_moves += 1

                move_pen_up = True
                if line_next is not None:
                    if math.isclose(line[2], line_next[0], abs_tol=config.comp_tolerance):
                        if math.isclose(line[3], line_next[1], abs_tol=config.comp_tolerance):
                            move_pen_up = False
                if move_pen_up:
                    out.write(CMD_PEN_UP)
                    out.write(f"G1 F{TRAVEL_SPEED}\n")
                    out.write("\n")
                    state_pen_up = True
                    count_pen_up += 1

                # dip pen calculations

                distance_travelled += math.sqrt((line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2)

                if args.dip_mode and distance_travelled > DIP_DISTANCE:
                    distance_travelled = 0

                    _dip_pen(out)
                    state_pen_up = True
                    count_pen_up += 1
                    count_dip_moves += 1

            out.write(CMD_PEN_UP)
            out.write(f"G1 F{TRAVEL_SPEED}\n")
            out.write("G1 X0 Y0\n")

            count_pen_up += 1

            # Lower pen (will fall down anyway when motor is turned off)
            # out.write("G1 Z0 F{}\n".format(PEN_LIFT_SPEED))
            # count_pen_down += 1

        print(f"write segment {s + 1}/{len(segments)}: {filename}")

    print(f"count_pen_up:        {count_pen_up:>5}")
    print(f"count_pen_down:      {count_pen_down:>5}")
    print(f"count_draw_moves:    {count_draw_moves:>5}")
    print(f"count_travel_moves:  {count_travel_moves:>5}")
    print(f"ratio draw/travel:   {float(count_draw_moves) / float(count_travel_moves):>5.3f}")

    if args.dip_mode:
        print(f"count dip:           {count_pen_up:>5}")


if __name__ == "__main__":
    main()
