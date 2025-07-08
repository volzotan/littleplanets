# littleplanets Makefile

.PHONY: all run clean

BLENDER_BIN := /Applications/Blender.app/Contents/MacOS/Blender
BLENDER_DIR := blender
BLENDER_FILE := $(BLENDER_DIR)/moon_Z.blend

PYPROJECT_FILE := pyproject.toml
SRC_DIR := src
DATA_DIR := data

DATA_LOWRES_DIR := data_lowres
BUILD_DIR := build

SCALING_FACTOR := 0.5

all: littleplanets.png

setup: $(PYPROJECT_FILE)
	@echo "Sync Environment"
	mkdir -p $(BUILD_DIR)
	mkdir -p $(DATA_LOWRES_DIR)
	uv sync

clean:
	rm -r $(BUILD_DIR)
	rm -r $(DATA_LOWRES_DIR)

# ----------

$(DATA_DIR)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif:
	@echo "Downloading digital elevation model data"
	wget -N https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif $@

$(DATA_LOWRES_DIR)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif: $(SRC_DIR)/resize_DEM.py $(DATA_DIR)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
	@echo "Resize $@"
	uv run $^ $@ 0.25

$(DATA_DIR)/lroc_color_poles.tif:
	@echo "Downloading surface color data"
	wget https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles.tif $@

$(BUILD_DIR)/mesh.ply: $(SRC_DIR)/mesh.py $(DATA_LOWRES_DIR)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif $(DATA_DIR)/lroc_color_poles.tif 
	@echo "Generating mesh"
	uv run $^ --output $@

$(BLENDER_FILE): $(BUILD_DIR)/mesh.ply
	@echo "Running blender mesh update"
	$(BLENDER_BIN) $(BLENDER_FILE) --python $(BLENDER_DIR)/import_ply.py -- --input $(BUILD_DIR)/mesh.ply --rotX -45 --rotY 90 --rotZ 0
	touch $@

$(BUILD_DIR)/raytrace.npy: $(BLENDER_FILE) $(BLENDER_DIR)/raytracing.py
	@echo "Running blender raytracer"
	$(BLENDER_BIN) $(BLENDER_FILE) --python $(BLENDER_DIR)/raytracing.py -- --output $@

$(BUILD_DIR)/normals.exr $(BUILD_DIR)/image.tif &: $(BLENDER_DIR)/moon_Z.blend
	@echo "Running blender renderer"
	$(BLENDER_BIN) -b $(BLENDER_FILE) -f 0 || true
	cp /tmp/Normals0000.exr $(BUILD_DIR)/normals.exr
	cp /tmp/Image0000.tif $(BUILD_DIR)/image.tif

$(BUILD_DIR)/projection_matrix.npy: $(BLENDER_FILE) $(BLENDER_DIR)/export_projection_matrix.py
	@echo "Running blender P matrix exporter"
	$(BLENDER_BIN) $(BLENDER_FILE) --python $(BLENDER_DIR)/export_projection_matrix.py -- --output $@

$(BUILD_DIR)/mesh_blender.ply: $(BLENDER_FILE) $(BLENDER_DIR)/export_ply.py
	@echo "Running blender mesh export"
	$(BLENDER_BIN) $(BLENDER_FILE) --python $(BLENDER_DIR)/export_ply.py -- --output $@

$(BUILD_DIR)/overlay.npz: $(SRC_DIR)/project_overlay.py $(BUILD_DIR)/mesh_blender.ply $(DATA_DIR)/Moon_apollo_landing_sites.json
	@echo "Projecting overlay POIs"
	uv run $^ --output $@

$(BUILD_DIR)/mapping_angle.png $(BUILD_DIR)/mapping_distance.png $(BUILD_DIR)/mapping_flat.png &: $(SRC_DIR)/process_blender.py $(BUILD_DIR)/normals.exr $(BUILD_DIR)/image.tif $(BUILD_DIR)/raytrace.npy 
	@echo "Processing blender mappings: $@"
	uv run $^ --output $(BUILD_DIR)

$(BUILD_DIR)/littleplanets.svg: $(SRC_DIR)/hatch.py $(BUILD_DIR)/mapping_angle.png $(BUILD_DIR)/mapping_distance.png $(BUILD_DIR)/mapping_flat.png $(BUILD_DIR)/overlay.npz $(BUILD_DIR)/projection_matrix.npy 
	@echo "Processing blender output"
	uv run $^ --output $(BUILD_DIR)/littleplanets.svg

littleplanets.png: $(BUILD_DIR)/littleplanets.svg
	inkscape $^ --export-filename=littleplanets.png --export-width=2000 --export-background=#000000
