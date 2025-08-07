# littleplanets Makefile

.PHONY: all run clean

DIR_BLENDER := blender
BLENDER_BIN := /Applications/Blender.app/Contents/MacOS/Blender
BLENDER_FILE := moon_Z.blend

INKSCAPE_BIN := /Applications/Inkscape.app/Contents/MacOS/inkscape

PYPROJECT_FILE := pyproject.toml

DIR_SRC := src
DIR_DATA := data
DIR_DATA_LOWRES := data_lowres
DIR_BUILD := build
DIR_DEBUG := debug

SCALING_FACTOR := 0.5

all: setup run

setup: $(PYPROJECT_FILE)
	@echo "Sync Environment"
	mkdir -p $(DIR_BUILD)
	mkdir -p $(DIR_DATA_LOWRES)
	mkdir -p $(DIR_DEBUG)
	uv sync

clean:
	rm -r $(DIR_BUILD)
	rm -r $(DIR_DATA_LOWRES)

# ----------

# Moon

$(DIR_DATA)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif:
	@echo "Downloading digital elevation model data"
	wget -N https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif -O $@

$(DIR_DATA_LOWRES)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif: $(DIR_SRC)/resize_DEM.py $(DIR_DATA)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
	@echo "Resize $@"
	uv run $^ $@ 0.25

$(DIR_DATA)/lroc_color_poles.tif:
	@echo "Downloading surface color data"
	wget https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles.tif -O $@

$(DIR_BUILD)/lroc_color_poles_imagemagick_contrast.tif: $(DIR_DATA)/lroc_color_poles.tif
	@echo "Increasing contrast for surface color data"
	magick $^ -level 20%,95% -brightness-contrast 0x10 $@

# Mars

$(DIR_DATA)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif:
	# https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m
	@echo "Downloading digital elevation model data"
	wget -N https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif -O $@

$(DIR_DATA)/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif:
	# https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m
	@echo "Downloading digital elevation model data"
	wget -N https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif -O $@

$(DIR_DATA)/marsexpress_hrsc_globalmosaic.rgb.tif:
	# https://hrscteam.dlr.de/public/data/globalcolor.php
	@echo "Downloading surface color data"
	wget https://hrscteam.dlr.de/public/data/global_mosaic/extra/globalmosaic.rgb.tif -O $@

$(DIR_DATA)/Mars_Viking_ClrMosaic_global_925m.tif:
	# https://astrogeology.usgs.gov/search/map/mars_viking_global_color_mosaic_925m
	@echo "Downloading surface color data"
	wget https://planetarymaps.usgs.gov/mosaic/Mars_Viking_ClrMosaic_global_925m.tif -O $@

$(DIR_DATA_LOWRES)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif: $(DIR_SRC)/resize_DEM.py $(DIR_DATA)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
	@echo "Resize $@"
	uv run $^ $@ 0.25

# ----------

DEM_FILE := $(DIR_DATA_LOWRES)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
COLOR_FILE := $(DIR_BUILD)/lroc_color_poles_imagemagick_contrast.tif 

ROT_X := -90
ROT_Y := 90
ROT_Z := 0

DEM_FILE := $(DIR_DATA_LOWRES)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
COLOR_FILE := $(DIR_DATA)/Mars_Viking_ClrMosaic_global_925m.tif

ROT_X := -90
ROT_Y := 150
ROT_Z := 0 

# ----------

$(DIR_BUILD)/mesh.ply: $(DIR_SRC)/mesh.py $(DEM_FILE) $(COLOR_FILE)
	@echo "Generating mesh"
	uv run $^ --output $@ --scale 0.04 --subdivision 10

$(DIR_BUILD)/$(BLENDER_FILE): $(DIR_BLENDER)/$(BLENDER_FILE) $(DIR_BUILD)/mesh.ply
	@echo "Running blender mesh update"
	cp $(DIR_BLENDER)/$(BLENDER_FILE) $(DIR_BUILD)/$(BLENDER_FILE)
	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/import_ply.py -- --input $(DIR_BUILD)/mesh.ply --rotX $(ROT_X) --rotY $(ROT_Y) --rotZ $(ROT_Z)
	touch $@

$(DIR_BUILD)/raytrace.npy: $(DIR_BUILD)/$(BLENDER_FILE) $(DIR_BLENDER)/raytracing.py
	@echo "Running blender raytracer"
	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/raytracing.py -- --output $@

$(DIR_BUILD)/normals.exr $(DIR_BUILD)/image.tif &: $(DIR_BUILD)/$(BLENDER_FILE)
	@echo "Running blender renderer"
	$(BLENDER_BIN) --background $(DIR_BUILD)/$(BLENDER_FILE) -f 0 || true
	cp /tmp/Normals0000.exr $(DIR_BUILD)/normals.exr
	cp /tmp/Image0000.tif $(DIR_BUILD)/image.tif

$(DIR_BUILD)/projection_matrix.npy: $(DIR_BUILD)/$(BLENDER_FILE) $(DIR_BLENDER)/export_projection_matrix.py
	@echo "Running blender P matrix exporter"
	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/export_projection_matrix.py -- --output $@

$(DIR_BUILD)/mesh_blender.ply: $(DIR_BUILD)/$(BLENDER_FILE) $(DIR_BLENDER)/export_ply.py
	@echo "Running blender mesh export"
	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/export_ply.py -- --output $@

$(DIR_BUILD)/overlay.npz: $(DIR_SRC)/project_overlay.py $(DIR_BUILD)/mesh_blender.ply $(DIR_DATA)/Moon_apollo_landing_sites.json
	@echo "Projecting overlay POIs"
	uv run $^ --output $@ --rotZ $(ROT_X) --circle-radius 0.015 --font-size 0.025

$(DIR_BUILD)/contours.npz: $(DIR_SRC)/contours.py $(DIR_BUILD)/normals.exr $(DIR_BUILD)/raytrace.npy
	@echo "Computing contours"
	uv run $^ --output $@

$(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_flat.png &: $(DIR_SRC)/process_blender.py $(DIR_BUILD)/normals.exr $(DIR_BUILD)/image.tif $(DIR_BUILD)/raytrace.npy 
	@echo "Processing blender mappings: $@"
	uv run $^ --output $(DIR_BUILD)

run: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/overlay.npz $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/contours.npz
	@echo "Processing blender output"
	uv run $(DIR_SRC)/hatch.py									\
		$(DIR_BUILD)/mapping_angle_5.png 						\
		$(DIR_BUILD)/mapping_distance.png 						\
		$(DIR_BUILD)/mapping_line_length.png 					\
		$(DIR_BUILD)/mapping_flat.png 							\
		--overlay $(DIR_BUILD)/overlay.npz 						\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  \
		--contours $(DIR_BUILD)/contours.npz					\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets.png --export-width=2000 --export-background=#000000

run_no_overlay: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/contours.npz
	@echo "Processing blender output"
	uv run $(DIR_SRC)/hatch.py 						\
		$(DIR_BUILD)/mapping_angle_5.png 			\
		$(DIR_BUILD)/mapping_distance.png 			\
		$(DIR_BUILD)/mapping_line_length.png 		\
		$(DIR_BUILD)/mapping_flat.png 				\
		--contours $(DIR_BUILD)/contours.npz		\
		--output $(DIR_BUILD)/littleplanets.svg 	\
		--blur-angle 0.20 							\
		--blur-distance 0.40						\
		--config-line-distance-end-factor 0.5 ;		\
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets.png --export-width=2000 --export-background=#000000

gcode: $(DIR_BUILD)/littleplanets.svg
	uv run svgtogcode.py $^

gcode_crop: $(DIR_BUILD)/littleplanets.svg
	uv run svgtogcode.py $^ --crop 375 375 100 400

# ----------

test_angle: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_angle_0.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/contours.npz
	for i in mapping_angle_0.png mapping_angle_1.png mapping_angle_2.png mapping_angle_3.png mapping_angle_4.png mapping_angle_5.png mapping_angle_6.png mapping_angle_7.png ; do \
		echo "$$i"	; \
		uv run $(DIR_SRC)/hatch.py 					\
			$(DIR_BUILD)/$$i 						\
			$(DIR_BUILD)/mapping_distance.png 		\
			$(DIR_BUILD)/mapping_line_length.png 	\
			$(DIR_BUILD)/mapping_flat.png 			\
			--contours $(DIR_BUILD)/contours.npz	\
			--output $(DIR_BUILD)/littleplanets.svg \
			--blur-angle 0.20 ; \
		$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets_$$i --export-width=2000 --export-background=#000000 ; \
	done

test_blur: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/contours.npz
	for i in 0 0.1 0.2 0.3 0.4 0.5 0.6 ; do \
		echo "$$i"	; \
		uv run $(DIR_SRC)/hatch.py 					\
			$(DIR_BUILD)/mapping_angle_5.png 		\
			$(DIR_BUILD)/mapping_distance.png 		\
			$(DIR_BUILD)/mapping_line_length.png 	\
			$(DIR_BUILD)/mapping_flat.png 			\
			--contours $(DIR_BUILD)/contours.npz	\
			--output $(DIR_BUILD)/littleplanets.svg \
			--blur-distance $$i 					\
			--config-line-distance-end-factor 0.5 	\
			--debug									\
			--suffix _$$i ; \
		$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets_$$i.png --export-width=2000 --export-background=#000000 ; \
	done

test_end_factor: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/contours.npz
	for i in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ; do \
		echo "$$i"	; \
		uv run $(DIR_SRC)/hatch.py 					\
			$(DIR_BUILD)/mapping_angle_5.png 		\
			$(DIR_BUILD)/mapping_distance.png 		\
			$(DIR_BUILD)/mapping_line_length.png 	\
			$(DIR_BUILD)/mapping_flat.png 			\
			--contours $(DIR_BUILD)/contours.npz	\
			--output $(DIR_BUILD)/littleplanets.svg \
			--blur-angle 0.10 						\
			--config-line-distance-end-factor $$i ; \
		$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets_$$i.png --export-width=2000 --export-background=#000000 ; \
	done