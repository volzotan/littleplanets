from pathlib import Path

from shapely.geometry import LineString
from shapely.affinity import translate
from svgwriter import SvgWriter
from hershey import HersheyFont, Align

DIMENSIONS = [200, 30]
OUTPUT_FILE = "test_pattern.svg"

BOX_WIDTH = 40
FONT_SIZE = 10

all_lines = []

# Vertical lines
lines = []
x = 0
while x < BOX_WIDTH:
    lines.append(LineString([[x, 0], [x, DIMENSIONS[1]]]))
    x += 0.5
    x *= 1.04

all_lines += [translate(l, xoff=0) for l in lines]

# Crossed lines
lines = []
x = 0
while x < BOX_WIDTH:
    lines.append(LineString([[x, 0], [x, DIMENSIONS[1]]]))
    x += 1.0
    x *= 1.04

y = 0
while y < DIMENSIONS[1]:
    lines.append(LineString([[0, y], [BOX_WIDTH, y]]))
    y += 1.0
    y *= 1.04

all_lines += [translate(l, xoff=BOX_WIDTH + 5) for l in lines]

# Text
font = HersheyFont(font_file=".." / Path(HersheyFont.DEFAULT_FONT))

text = font.lines_for_text("THE QUICK brown fox", 10)
all_lines += [translate(l, xoff=(BOX_WIDTH + 5) * 2, yoff=10) for l in text]

text = font.lines_for_text("THE QUICK brown fox", 8)
all_lines += [translate(l, xoff=(BOX_WIDTH + 5) * 2, yoff=10 + 8 + 1) for l in text]

text = font.lines_for_text("THE QUICK brown fox", 6)
all_lines += [translate(l, xoff=(BOX_WIDTH + 5) * 2, yoff=10 + 8 + 6 + 3) for l in text]


svg = SvgWriter(OUTPUT_FILE, DIMENSIONS)
svg.add_style(
    "main",
    {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.40",
        "fill-opacity": "1.0",
    },
)
svg.add("main", all_lines)
svg.write()
