from shapely.geometry import LineString


def linestring_to_coordinate_pairs(linestring: LineString) -> list[list[tuple[float, float]]]:
    pairs = []

    for i in range(len(linestring.coords) - 1):
        pairs.append([linestring.coords[i], linestring.coords[i + 1]])

    return pairs
