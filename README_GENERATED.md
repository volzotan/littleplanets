## Usage

Make Targets

| Target | Description |
|--------|-------------|
| `make all` | Full pipeline: setup + run |
| `make setup` | Create directories and sync Python environment |
| `make run` | Generate SVG and PNG with grid, axis, contour, and POI overlays |
| `make run_clouds_coastlines` | Same as `run` plus coastline and cloud overlays (Earth) |
| `make run_no_overlays` | Hatchlines only, no overlays |
| `make gcode` | Convert final SVG to G-code for pen plotters |
| `make clean` | Remove the `build/` directory |

## Configuration

Each planet has a TOML config file in `config/`. The config has two levels:

1. **Global keys** (top-level) are shared across all stages: rotation angles, light mode, color palette, output dimensions.
2. **Section keys** (e.g. `[mesh]`, `[hatch]`, `[combine]`) configure individual pipeline stages.

The configurator splits the master config into per-stage TOML files in `build/`, each containing the global keys plus that stage's section.

### Example: Mars Config (abbreviated)

```toml
# Global parameters
rotX = -90                              # Pitch rotation (degrees)
rotY = 84.375                           # Roll rotation
rotZ = -22.5                            # Yaw rotation
light_mode = "axis"                     # Derive lighting from rotation axis
colors = [[240, 126, 50], [65, 102, 174]]  # Orange + blue ink palette
dimensions = [699, 699]                 # Output size in pixels

[downloader]
dem_url = "https://..."                 # USGS HRSC-MOLA blended DEM
surface_color_url = "https://..."       # Viking color mosaic

[mesh]
scale = 0.04                            # Elevation exaggeration factor
subdivision = 10                        # Mesh refinement iterations

[hatch]
flowlines_line_distance = [0.5, 7.0]   # Min/max spacing between lines
flowlines_line_max_length = [10, 20]    # Min/max line length
flowlines_max_angle_discontinuity = 0.78  # ~45 degrees

[combine]
hatchlines_smoothing_iterations = 0     # Chaikin smoothing passes
contours_smoothing_iterations = 5
```

### Supported Planets

| Planet | Config | Data Source (DEM) | Data Source (Color) | Notable Features |
|--------|--------|-------------------|---------------------|------------------|
| Mars | `mars.toml` | USGS HRSC-MOLA blend (200m) | Viking color mosaic | POIs for Olympus Mons, Valles Marineris, etc. |
| Moon | `moon.toml` | LRO LOLA (118m) | LROC WAC color mosaic | Apollo/Luna/Chang'e landing sites as POIs |
| Earth | `earth.toml` | GEBCO bathymetry | NASA Blue Marble | Cloud overlays from ERA5, coastlines |
| Venus | `venus.toml` | Magellan radar topography | Surface color mosaic | Radar-derived elevation |
| Jupiter | `jupiter.toml` | Flat (no topography) | Cassini color imagery | Atmosphere bands only |

### Points of Interest

POI files are JSON arrays. Each entry places a labeled marker on the globe:

```json
{
  "name": "Apollo 11",
  "lat": 0.674,
  "lon": 23.473,
  "label_angle": 0
}
```

Optional fields: `label_lat`/`label_lon` (separate label position), `circle_radius` (override marker size), `path` (KMZ file with line geometry, e.g. rover traverses), `invisible` (hide this POI).

Labels are rendered using Hershey vector fonts (`fonts/HersheySans1.svg`, `fonts/HersheySerifMed.svg`) — single-stroke fonts suitable for pen plotters.


## Pipeline Stages in Detail

### 1. Configuration (`src/configurator.py`)

Reads the master TOML and writes one config file per pipeline stage into `build/`. Each output file merges the global keys with the stage's section keys. Files are only rewritten when content changes, so Make's dependency tracking works correctly.

### 2. Data Download (`src/downloader.py`)

Downloads DEM and surface color rasters via `wget`. Supports GeoTIFF URLs directly; non-TIFF formats are converted via ImageMagick. If the DEM URL is empty, a zero-value TIFF is written (used for Jupiter, which has no topography).

For Earth, optionally downloads ERA5 reanalysis data from the Copernicus Climate Data Store: total cloud cover, 10m wind components, and land-sea mask as netCDF.

### 3. TIFF Processing (`src/modify_tiff.py`)

Applies a configurable sequence of operations to the raw rasters:
- Resize to target width (typically 10,000-15,000 px)
- Gaussian blur (kernel as percentage of image size)
- Contrast stretching and CLAHE enhancement
- Data type conversion (e.g. float32 to uint8)
- Floor/ceil clipping and binary thresholding

Spatial metadata (CRS, geotransform) is preserved through all operations via rasterio.

### 4. Mesh Generation (`src/mesh.py`)

Starts with a base polyhedron (cube), recursively subdivides it to the configured level (each step splits every triangle into four), then projects vertices onto a sphere. Elevation data from the DEM scales each vertex radially outward:

```
radius = 1.0 + elevation * scale
```

Surface colors are sampled from the color raster at each vertex's lat/lon position and stored as per-face vertex colors. The mesh is exported as a PLY file.

Key parameters:
- `subdivision`: Controls mesh density (10 = ~2M triangles)
- `scale`: Elevation exaggeration (0.04 for Mars, 0.10 for Earth)
- `blur`: Additional smoothing of the elevation raster

### 5. Blender Processing (scripts in `src/blender/`)

The pipeline runs six Blender operations in headless mode, each saving a `.blend` file that the next stage picks up:

1. **adjust_paths.py** — Set render output directories in the compositor
2. **adjust_scene.py** — Configure camera (position, focal length) and area light (position, size, power). Camera distance is computed from the sphere radius and focal length to frame the planet with a configurable margin.
3. **import_ply.py** — Import the mesh, apply rotation, enable smooth shading, create a vertex-color material (roughness=1, specular=0 for a fully matte look)
4. **raytracing.py** — For each pixel, cast a ray from the camera and record the 3D world-space hit position. Optionally cast a second ray through the first hit for back-face detection. Output: NPY arrays of shape `[height, width, 3]`, NaN for misses.
5. **Blender render** — Render normals to EXR and the color image to TIFF via the compositor
6. **export_projection_matrix.py** — Compute and export the 3x4 camera projection matrix (intrinsic x extrinsic) for later 3D-to-2D projection of overlay geometry
7. **export_ply.py** — Re-export the mesh from Blender (with any modifications applied)

Each script is invoked via `src/blender_wrapper.py`, which converts TOML config keys to CLI arguments.

### 6. Mapping Generation (`src/process_blender.py`)

Converts the Blender render outputs into four grayscale mapping images that drive the hatching stage:

- **`mapping_angle.png`** — Direction field. For each pixel, computes the tangent direction from the interplay of light direction and elevation gradient, projected onto the surface normal plane. Values encode angles as 0-255 = 0-2pi. Three light modes are supported: `implicit` (light position from config), `explicit` (direct light vector), `axis` (derived from planet rotation).

- **`mapping_distance.png`** — Line spacing. Derived from the dot product of the surface normal and view direction, with CLAHE contrast enhancement and percentile clipping. Darker areas get denser lines.

- **`mapping_line_length.png`** — Maximum line length per pixel. Multiple computation modes: surface flatness, local roughness (windowed variance), brightness, or altitude.

- **`mapping_background.png`** — Binary mask separating the planet from empty space.

Additionally generates `mapping_color.png` (the rendered RGB image) for the palette stage.

### 7. Overlay Generation

Seven overlay types, each following a common workflow:

**Generate** (create raw geometry in 3D) -> **Project** (snap onto mesh surface via KD-tree) -> **Visibility** (raycast in Blender to test occlusion) -> **Crop** (keep only visible segments)

| Overlay | Source | Description |
|---------|--------|-------------|
| **Coastlines** (`overlay_coastlines.py`) | DEM raster | Binary threshold on elevation, contour extraction with morphological cleanup, projected to sphere |
| **Grid** (`overlay_grid.py`) | Config params | Lat/lon circles at configurable intervals |
| **Contours** (`overlay_contours.py`) | Normals EXR + raytrace | Silhouette edges extracted from the rendered normal map |
| **POIs** (`overlay_pois.py`) | JSON files | Circles, labels (Hershey font), optional KMZ paths (e.g. rover traverses), rotated to lat/lon position |
| **Axis** (`overlay_axis.py`) | Config params | Dashed Z-axis reference line |
| **Clouds** (`overlay_clouds.py`) | ERA5 netCDF + raytrace | Cloud coverage mapped to a separate mesh layer, wind direction used for hatching angle |
| **Projection** (`overlay_project.py`) | Mesh PLY + linestrings | Snaps overlay lines to the mesh surface using KD-tree, scaled slightly outward (1.02x) |

Visibility testing (`src/blender/export_overlay_lines.py`) imports overlay linestrings as Blender curve objects with bevel, then raycasts from the camera through each point. If the ray hits the overlay curve before the mesh, the point is marked visible.

### 8. Hatching (`src/hatch.py`, `src/util/flowlines.py`)

Generates evenly-spaced streamlines using the Jobard-Lefer algorithm ("Creating Evenly-Spaced Streamlines of Arbitrary Density", 1997).

The algorithm:
1. Seed the field with a grid of starting points plus contour vertices as high-priority seeds
2. For each seed point, trace a line in both directions along the angle field
3. At each step, check for collisions with existing lines. Stop on boundary, collision, or excessive angle change.
4. Discard lines shorter than the minimum length
5. Extract new seed points perpendicular to the completed line
6. Repeat until seeds are exhausted

Collision detection uses a fast raster-based approach: a boolean grid at scaled resolution marks occupied cells. Line spacing varies per-pixel according to the distance mapping.

Key parameters:
- `flowlines_line_distance`: `[min, max]` spacing in pixels (mapped from the distance image)
- `flowlines_line_max_length`: `[min, max]` line length
- `flowlines_max_angle_discontinuity`: Maximum angle change between consecutive points before a line is terminated (in radians)
- `blur_angle_kernel_size`, `blur_distance_kernel_size`: Gaussian blur on input mappings (as fraction of image size)

### 9. SVG Composition (`src/combine.py`)

Merges all layers into a single SVG:

1. **Load** hatchlines, overlays, contours (all NPZ files containing linestrings)
2. **Project** 3D overlay geometry to 2D using the camera projection matrix (perspective divide)
3. **Cut** hatchlines around overlay geometry. Two buffer distances control this: `overlay_cutout_cut_distance` for POI/path cutouts, `overlay_layering_cut_distance` for stacking overlays on top of hatching. Uses Shapely's `difference` operation with an STRtree spatial index for performance.
4. **Smooth** all linestrings with Chaikin's corner-cutting algorithm (configurable iterations per layer)
5. **Color** hatchlines by sampling the color mapping at each line's centroid, then probabilistically assigning a palette color weighted by similarity
6. **Frame** generation (optional decorative border)
7. **Write** SVG via `src/svgwriter.py` with named, styled layers

The SVG writer supports Inkscape-compatible layers with independent stroke colors and display toggles.

### 10. Export

**PNG**: Inkscape converts the SVG to a rasterized PNG (black background, configurable width).

**G-code** (`svgtogcode.py`): Reads SVG paths and converts to G-code for CNC pen plotters. Features:
- Configurable travel/write/pen-lift speeds
- Automatic pen dipping at a reservoir location (two modes: fixed location or gantry-mounted)
- Pen up/down distances
- Line segmentation for smooth curves