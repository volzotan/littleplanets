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

all: setup run

setup: $(PYPROJECT_FILE)
	@echo "Sync Environment"
	mkdir -p $(BUILD_DIR)
	mkdir -p $(DATA_LOWRES_DIR)
	uv sync

clean:
	rm -r $(BUILD_DIR)
	rm -r $(DATA_LOWRES_DIR)

# ----------

# Moon

$(DATA_DIR)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif:
	@echo "Downloading digital elevation model data"
	wget -N https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif -O $@

$(DATA_LOWRES_DIR)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif: $(SRC_DIR)/resize_DEM.py $(DATA_DIR)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
	@echo "Resize $@"
	uv run $^ $@ 0.25

$(DATA_DIR)/lroc_color_poles.tif:
	@echo "Downloading surface color data"
	wget https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles.tif -O $@

# Mars

$(DATA_DIR)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif:
	# https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m
	@echo "Downloading digital elevation model data"
	wget -N https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif -O $@

$(DATA_DIR)/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif:
	# https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m
	@echo "Downloading digital elevation model data"
	wget -N https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif -O $@

$(DATA_DIR)/marsexpress_hrsc_globalmosaic.rgb.tif:
	# https://hrscteam.dlr.de/public/data/globalcolor.php
	@echo "Downloading surface color data"
	wget https://hrscteam.dlr.de/public/data/global_mosaic/extra/globalmosaic.rgb.tif -O $@

$(DATA_DIR)/Mars_Viking_ClrMosaic_global_925m.tif:
	# https://astrogeology.usgs.gov/search/map/mars_viking_global_color_mosaic_925m
	@echo "Downloading surface color data"
	wget https://planetarymaps.usgs.gov/mosaic/Mars_Viking_ClrMosaic_global_925m.tif -O $@


$(DATA_LOWRES_DIR)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif: $(SRC_DIR)/resize_DEM.py $(DATA_DIR)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
	@echo "Resize $@"
	uv run $^ $@ 0.25

# ----------

DEM_FILE := Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
COLOR_FILE := lroc_color_poles.tif 

ROT_X := -90
ROT_Y := 90
ROT_Z := 0

# DEM_FILE := Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
# COLOR_FILE := Mars_Viking_ClrMosaic_global_925m.tif

# ROT_X := -90
# ROT_Y := 150
# ROT_Z := 0

# ----------

$(BUILD_DIR)/mesh.ply: $(SRC_DIR)/mesh.py $(DATA_LOWRES_DIR)/$(DEM_FILE) $(DATA_DIR)/$(COLOR_FILE)
	@echo "Generating mesh"
	uv run $^ --output $@ --scale 0.08

$(BLENDER_FILE): $(BUILD_DIR)/mesh.ply
	@echo "Running blender mesh update"
	$(BLENDER_BIN) $(BLENDER_FILE) --background --python $(BLENDER_DIR)/import_ply.py -- --input $(BUILD_DIR)/mesh.ply --rotX $(ROT_X) --rotY $(ROT_Y) --rotZ $(ROT_Z)
	touch $@

$(BUILD_DIR)/raytrace.npy: $(BLENDER_FILE) $(BLENDER_DIR)/raytracing.py
	@echo "Running blender raytracer"
	$(BLENDER_BIN) $(BLENDER_FILE) --background --python $(BLENDER_DIR)/raytracing.py -- --output $@

$(BUILD_DIR)/normals.exr $(BUILD_DIR)/image.tif &: $(BLENDER_DIR)/moon_Z.blend
	@echo "Running blender renderer"
	$(BLENDER_BIN) --background $(BLENDER_FILE) -f 0 || true
	cp /tmp/Normals0000.exr $(BUILD_DIR)/normals.exr
	cp /tmp/Image0000.tif $(BUILD_DIR)/image.tif

$(BUILD_DIR)/projection_matrix.npy: $(BLENDER_FILE) $(BLENDER_DIR)/export_projection_matrix.py
	@echo "Running blender P matrix exporter"
	$(BLENDER_BIN) $(BLENDER_FILE) --background --python $(BLENDER_DIR)/export_projection_matrix.py -- --output $@

$(BUILD_DIR)/mesh_blender.ply: $(BLENDER_FILE) $(BLENDER_DIR)/export_ply.py
	@echo "Running blender mesh export"
	$(BLENDER_BIN) $(BLENDER_FILE) --background --python $(BLENDER_DIR)/export_ply.py -- --output $@

$(BUILD_DIR)/overlay.npz: $(SRC_DIR)/project_overlay.py $(BUILD_DIR)/mesh_blender.ply $(DATA_DIR)/Moon_apollo_landing_sites.json
	@echo "Projecting overlay POIs"
	uv run $^ --output $@ --rotZ -90

$(BUILD_DIR)/contours.npz: $(SRC_DIR)/contours.py $(BUILD_DIR)/normals.exr $(BUILD_DIR)/raytrace.npy
	@echo "Computing contours"
	uv run $^ --output $@

$(BUILD_DIR)/mapping_angle_0.png $(BUILD_DIR)/mapping_distance.png $(BUILD_DIR)/mapping_line_length.png $(BUILD_DIR)/mapping_flat.png &: $(SRC_DIR)/process_blender.py $(BUILD_DIR)/normals.exr $(BUILD_DIR)/image.tif $(BUILD_DIR)/raytrace.npy 
	@echo "Processing blender mappings: $@"
	uv run $^ --output $(BUILD_DIR)

run: $(SRC_DIR)/hatch.py $(BUILD_DIR)/mapping_angle_0.png $(BUILD_DIR)/mapping_distance.png $(BUILD_DIR)/mapping_line_length.png $(BUILD_DIR)/mapping_flat.png $(BUILD_DIR)/overlay.npz $(BUILD_DIR)/projection_matrix.npy $(BUILD_DIR)/contours.npz
	@echo "Processing blender output"
	uv run $(SRC_DIR)/hatch.py									\
		$(BUILD_DIR)/mapping_angle_0.png 						\
		$(BUILD_DIR)/mapping_distance.png 						\
		$(BUILD_DIR)/mapping_line_length.png 					\
		$(BUILD_DIR)/mapping_flat.png 							\
		--overlay $(BUILD_DIR)/overlay.npz 						\
		--projection-matrix $(BUILD_DIR)/projection_matrix.npy  \
		--contours $(BUILD_DIR)/contours.npz					\
		--output $(BUILD_DIR)/littleplanets.svg
	inkscape $(BUILD_DIR)/littleplanets.svg --export-filename=littleplanets.png --export-width=2000 --export-background=#000000

run_no_overlay: $(SRC_DIR)/hatch.py $(BUILD_DIR)/mapping_angle_0.png $(BUILD_DIR)/mapping_distance.png $(BUILD_DIR)/mapping_line_length.png $(BUILD_DIR)/mapping_flat.png
	@echo "Processing blender output"
	uv run $^ --output $(BUILD_DIR)/littleplanets.svg
	inkscape $(BUILD_DIR)/littleplanets.svg --export-filename=littleplanets.png --export-width=2000 --export-background=#000000

special: $(SRC_DIR)/hatch.py $(BUILD_DIR)/mapping_angle_0.png $(BUILD_DIR)/mapping_distance.png $(BUILD_DIR)/mapping_flat.png
	@echo "Processing blender output"
	# for i in mapping_angle_0.png mapping_angle_1.png mapping_angle_2.png mapping_angle_3.png mapping_angle_4.png mapping_angle_5.png mapping_angle_6.png mapping_angle_7.png ; do \
	for i in mapping_angle_4.png mapping_angle_5.png mapping_angle_6.png mapping_angle_7.png ; do \
		echo "$$i"	; \
		uv run $(SRC_DIR)/hatch.py 					\
			$(BUILD_DIR)/$$i 						\
			$(BUILD_DIR)/mapping_distance.png 		\
			$(BUILD_DIR)/mapping_line_length.png 	\
			$(BUILD_DIR)/mapping_flat.png 			\
			--output $(BUILD_DIR)/littleplanets.svg \
			--blur-angle 0.50 ; \
		inkscape $(BUILD_DIR)/littleplanets.svg --export-filename=littleplanets_$$i --export-width=2000 --export-background=#000000 ; \
	done

gcode: $(BUILD_DIR)/littleplanets.svg
	uv run svgtogcode.py $^
