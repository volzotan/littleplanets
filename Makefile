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

# ----------

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

# Moon

#DEM_FILE := $(DIR_DATA_LOWRES)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
#COLOR_FILE := $(DIR_BUILD)/lroc_color_poles_imagemagick_contrast.tif
#POI_FILE := $(DIR_DATA)/Moon_apollo_landing_sites.json
#
#ROT_X := -90
#ROT_Y := 90
#ROT_Z := -6.68
#
#LIGHT_ANGLE_XY := 83.32
#LIGHT_ANGLE_Z := 60
#
#COLOR_1 := 255 255 255
#COLOR_2 := 111 115 122

# Mars

DEM_FILE := $(DIR_DATA_LOWRES)/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
COLOR_FILE := $(DIR_DATA)/Mars_Viking_ClrMosaic_global_925m.tif
POI_FILE := $(DIR_DATA)/Mars_poi.json

ROT_X := -90
ROT_Y := 80
ROT_Z := -22.5

LIGHT_ANGLE_XY := 67.5
LIGHT_ANGLE_Z := 50

COLOR_1 := 240 126 50
COLOR_2 := 65 102 174

OVERLAYS := grid pois

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

# Overlays

$(DIR_BUILD)/overlay_pois.npz: $(DIR_SRC)/overlay_pois.py $(POI_FILE)
	@echo "Create POI overlay"
	uv run $^ --output $@ --rotX $(ROT_X) --rotY $(ROT_Y) --rotZ $(ROT_Z) --circle-radius 0.015 --font-size 0.025

$(DIR_BUILD)/overlay_grid.npz: $(DIR_SRC)/overlay_grid.py
	@echo "Create grid overlay"
	uv run $^ --output $@ --rotX $(ROT_X) --rotY $(ROT_Y) --rotZ $(ROT_Z) --grid-num-lat 8 --grid-num-lon 16


#$(DIR_BUILD)/overlay_visible_%.npz: $(DIR_BUILD)/$(BLENDER_FILE) $(DIR_BLENDER)/export_overlay_lines.py $(DIR_BUILD)/overlay_%.npz
#	@echo "Running blender overlay visibility detection"
#	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay.npz --output $@
#
#$(DIR_BUILD)/overlay_%_cropped.npz: $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_%.npz $(DIR_BUILD)/overlay_visible_%.npz
#	@echo "Cropping visible overlay lines"
#	uv run $^ --output $@


$(DIR_BUILD)/overlay_pois_cropped.npz: $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_pois.npz
	@echo "Cropping visible overlay lines: POIs"
	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_pois.npz --output $(DIR_BUILD)/overlay_pois_projected.npz
	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_pois_projected.npz --output $(DIR_BUILD)/overlay_pois_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_pois_projected.npz $(DIR_BUILD)/overlay_pois_visible.npz --output $@

$(DIR_BUILD)/overlay_grid_cropped.npz: $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_grid.npz
	@echo "Cropping visible overlay lines: GRID"
	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_grid.npz --output $(DIR_BUILD)/overlay_grid_projected.npz
	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_grid_projected.npz --output $(DIR_BUILD)/overlay_grid_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_grid_projected.npz $(DIR_BUILD)/overlay_grid_visible.npz --output $@




$(DIR_BUILD)/contours.npz: $(DIR_SRC)/contours.py $(DIR_BUILD)/normals.exr $(DIR_BUILD)/raytrace.npy
	@echo "Computing contours"
	uv run $^ --output $@

$(DIR_BUILD)/mapping_color.png $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_flat.png &: $(DIR_SRC)/process_blender.py $(DIR_BUILD)/normals.exr $(DIR_BUILD)/image.tif $(DIR_BUILD)/raytrace.npy
	@echo "Processing blender mappings: $@"
	uv run $^ --light-angle $(LIGHT_ANGLE_XY) $(LIGHT_ANGLE_Z) --output $(DIR_BUILD)

$(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_brightness_difference.png &: $(DIR_SRC)/palette.py $(DIR_BUILD)/image.tif
	@echo "Processing palette colors"
	uv run $^ 													\
		--palette-mixture $(DIR_BUILD)/mapping_color.npy 		\
		--palette-brightness-difference $(DIR_BUILD)/mapping_brightness_difference.png \
		--palette-color $(COLOR_1)								\
		--palette-color $(COLOR_2)								\

run: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/overlay_pois_cropped.npz $(DIR_BUILD)/overlay_grid_cropped.npz  $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/contours.npz
	@echo "Processing blender output"
	uv run $(DIR_SRC)/hatch.py									\
		$(DIR_BUILD)/mapping_color.npy 							\
		$(DIR_BUILD)/mapping_angle_5.png 						\
		$(DIR_BUILD)/mapping_distance.png 						\
		$(DIR_BUILD)/mapping_line_length.png 					\
		$(DIR_BUILD)/mapping_flat.png 							\
		--overlay $(DIR_BUILD)/overlay_pois_cropped.npz $(DIR_BUILD)/overlay_grid_cropped.npz 		\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  \
		--contours $(DIR_BUILD)/contours.npz					\
		--config config_hatch.toml 								\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets.png --export-width=2000 --export-background=#000000

run_palette: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_brightness_difference.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_flat.png config_hatch.toml $(DIR_BUILD)/contours.npz
	@echo "Processing blender output"
	uv run $(DIR_SRC)/hatch.py									\
		$(DIR_BUILD)/mapping_color.npy 							\
		$(DIR_BUILD)/mapping_angle_5.png 						\
		$(DIR_BUILD)/mapping_brightness_difference.png			\
		$(DIR_BUILD)/mapping_line_length.png 					\
		$(DIR_BUILD)/mapping_flat.png 							\
		--config config_hatch.toml 								\
		--palette-color $(COLOR_1)								\
		--palette-color $(COLOR_2)								\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets.png --export-width=2000 --export-background=#000000


#run_no_overlay: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_flat.png config_hatch.toml $(DIR_BUILD)/contours.npz
#	@echo "Processing blender output"
#	uv run $(DIR_SRC)/hatch.py 						\
#		$(DIR_BUILD)/mapping_color.npy 				\
#		$(DIR_BUILD)/mapping_angle_5.png 			\
#		$(DIR_BUILD)/mapping_distance.png 			\
#		$(DIR_BUILD)/mapping_line_length.png 		\
#		$(DIR_BUILD)/mapping_flat.png 				\
#		--config config_hatch.toml 					\
#		--contours $(DIR_BUILD)/contours.npz		\
#		--output $(DIR_BUILD)/littleplanets.svg 	\
#		--blur-angle 0.20 							\
#		--blur-distance 0.40						\
#		--config-line-distance-end-factor 0.5 ;		\
#	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets.png --export-width=2000 --export-background=#000000

gcode: $(DIR_BUILD)/littleplanets.svg
	uv run svgtogcode.py $^

gcode_crop: $(DIR_BUILD)/littleplanets.svg
	uv run svgtogcode.py $^ --crop 375 375 100 400

# ----------

test_angle: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_color.png $(DIR_BUILD)/mapping_angle_0.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/contours.npz
	for i in mapping_angle_0.png mapping_angle_1.png mapping_angle_2.png mapping_angle_3.png mapping_angle_4.png mapping_angle_5.png mapping_angle_6.png mapping_angle_7.png ; do \
		echo "$$i"	; \
		uv run $(DIR_SRC)/hatch.py 					\
			$(DIR_BUILD)/mapping_color.png 			\
			$(DIR_BUILD)/$$i 						\
			$(DIR_BUILD)/mapping_distance.png 		\
			$(DIR_BUILD)/mapping_line_length.png 	\
			$(DIR_BUILD)/mapping_flat.png 			\
			--config config_hatch.toml				\
			--contours $(DIR_BUILD)/contours.npz	\
			--output $(DIR_BUILD)/littleplanets.svg ; \
		$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets_$$i --export-width=2000 --export-background=#000000 ; \
	done

test_blur: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_color.png $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/contours.npz
	for i in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0 ; do \
		echo "$$i"	; \
		cp config_hatch.toml config_hatch_override.toml ; \
		echo "blur_angle_kernel_size_perc=$$i" > config_hatch_override.toml ;\
		uv run $(DIR_SRC)/hatch.py 					\
			$(DIR_BUILD)/mapping_color.png 			\
			$(DIR_BUILD)/mapping_angle_5.png 		\
			$(DIR_BUILD)/mapping_distance.png 		\
			$(DIR_BUILD)/mapping_line_length.png 	\
			$(DIR_BUILD)/mapping_flat.png 			\
			--config config_hatch_override.toml	\
			--contours $(DIR_BUILD)/contours.npz	\
			--output $(DIR_BUILD)/littleplanets.svg \
			--debug									\
			--suffix _$$i ; \
		echo "blur_angle_kernel_size_perc=$$i" > littleplanets_$$i.toml
		$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets_$$i.png --export-width=2000 --export-background=#000000 ; \
	done

test_end_factor: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_color.png $(DIR_BUILD)/mapping_angle_5.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_flat.png $(DIR_BUILD)/contours.npz
	for i in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ; do \
		echo "$$i"	; \
		cp config_hatch.toml config_hatch_overwrite.toml ; \
		echo "flowlines_line_distance_end_factor=$$i" > config_hatch_override.toml ;\
		uv run $(DIR_SRC)/hatch.py 					\
			$(DIR_BUILD)/mapping_color.png 			\
			$(DIR_BUILD)/mapping_angle_5.png 		\
			$(DIR_BUILD)/mapping_distance.png 		\
			$(DIR_BUILD)/mapping_line_length.png 	\
			$(DIR_BUILD)/mapping_flat.png 			\
			--config config_hatch_override.toml	\
			--contours $(DIR_BUILD)/contours.npz	\
			--output $(DIR_BUILD)/littleplanets.svg ; \
		$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets_$$i.png --export-width=2000 --export-background=#000000 ; \
	done