# littleplanets Makefile

.PHONY: all run clean

DIR_BLENDER := blender
BLENDER_BIN := /Applications/Blender.app/Contents/MacOS/Blender
BLENDER_FILE := template.blend

INKSCAPE_BIN := /Applications/Inkscape.app/Contents/MacOS/inkscape

PYPROJECT_FILE := pyproject.toml

DIR_SRC := src
DIR_DATA := data
DIR_DATA_LOWRES := $(DIR_DATA)_lowres
DIR_BUILD := build
DIR_CONFIG := config
DIR_DEBUG := debug

CONFIG := mars.toml
POI_FILE := $(DIR_CONFIG)/poi.json
OUTPUT_PNG := littleplanets.png

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

#$(DIR_DATA)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif:
#	@echo "Downloading digital elevation model data"
#	wget -N https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif -O $@
#
#$(DIR_DATA_LOWRES)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif: $(DIR_SRC)/resize_DEM.py $(DIR_DATA)/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
#	@echo "Resize $@"
#	uv run $^ $@ 0.25
#
#$(DIR_DATA)/lroc_color_poles.tif:
#	@echo "Downloading surface color data"
#	wget https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles.tif -O $@
#
#$(DIR_BUILD)/lroc_color_poles_imagemagick_contrast.tif: $(DIR_DATA)/lroc_color_poles.tif
#	@echo "Increasing contrast for surface color data"
#	magick $^ -level 20%,95% -brightness-contrast 0x10 $@

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


# ----------

$(DIR_BUILD)/%.toml: $(DIR_SRC)/configurator.py $(CONFIG)
	@echo "Configurator for file $(CONFIG)"
	uv run $^ --output $(DIR_BUILD)

$(DIR_DATA)/dem.tif $(DIR_DATA)/surface_color.tif: $(DIR_SRC)/downloader.py $(DIR_BUILD)/downloader.toml
	@echo "Downloader"
	uv run $(DIR_SRC)/downloader.py --output-dir $(DIR_DATA) --config $(DIR_BUILD)/downloader.toml

$(DIR_DATA_LOWRES)/dem.tif: $(DIR_SRC)/resize_DEM.py $(DIR_DATA)/dem.tif
	@echo "Resize $@"
	uv run $^ $@ 0.25

$(DIR_BUILD)/mesh.ply: $(DIR_SRC)/mesh.py $(DIR_DATA_LOWRES)/dem.tif $(DIR_DATA)/surface_color.tif
	@echo "Generating mesh"
	uv run $^ --output $@ --scale 0.04 --subdivision 10

$(DIR_BUILD)/blender_mesh.blend: $(DIR_SRC)/blender_wrapper.py $(DIR_BLENDER)/import_ply.py $(DIR_BUILD)/mesh.ply $(DIR_BUILD)/import_ply.toml
	@echo "Running blender mesh update"
	cp $(DIR_BLENDER)/$(BLENDER_FILE) $@
	uv run $(DIR_SRC)/blender_wrapper.py $(BLENDER_BIN) $@ $(DIR_BLENDER)/import_ply.py --config $(DIR_BUILD)/import_ply.toml --params "--input $(DIR_BUILD)/mesh.ply"

$(DIR_BUILD)/raytrace.npy: $(DIR_BUILD)/blender_mesh.blend $(DIR_BLENDER)/raytracing.py
	@echo "Running blender raytracer"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background --python $(DIR_BLENDER)/raytracing.py -- --output $@

$(DIR_BUILD)/normals.exr $(DIR_BUILD)/image.tif &: $(DIR_BUILD)/blender_mesh.blend
	@echo "Running blender renderer"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background -f 0 || true
	cp /tmp/Normals0000.exr $(DIR_BUILD)/normals.exr
	cp /tmp/Image0000.tif $(DIR_BUILD)/image.tif

$(DIR_BUILD)/projection_matrix.npy: $(DIR_BUILD)/blender_mesh.blend $(DIR_BLENDER)/export_projection_matrix.py
	@echo "Running blender P matrix exporter"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background --python $(DIR_BLENDER)/export_projection_matrix.py -- --output $@

$(DIR_BUILD)/mesh_blender.ply: $(DIR_BUILD)/blender_mesh.blend $(DIR_BLENDER)/export_ply.py
	@echo "Running blender mesh export"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background --python $(DIR_BLENDER)/export_ply.py -- --output $@

# Overlays

$(DIR_BUILD)/overlay_pois.npz: $(DIR_SRC)/overlay_pois.py $(POI_FILE) $(DIR_BUILD)/overlay_pois.toml
	@echo "Create POI overlay"
	uv run $(DIR_SRC)/overlay_pois.py $(POI_FILE) --output $@ --config $(DIR_BUILD)/overlay_pois.toml

$(DIR_BUILD)/overlay_grid.npz: $(DIR_SRC)/overlay_grid.py $(DIR_BUILD)/overlay_grid.toml
	@echo "Create grid overlay"
	uv run $(DIR_SRC)/overlay_grid.py --output $@ --config $(DIR_BUILD)/overlay_grid.toml

$(DIR_BUILD)/overlay_axis.npz: $(DIR_SRC)/overlay_axis.py $(DIR_BUILD)/overlay_axis.toml
	@echo "Create axis overlay"
	uv run $(DIR_SRC)/overlay_axis.py --output $@ --config $(DIR_BUILD)/overlay_axis.toml


#$(DIR_BUILD)/overlay_visible_%.npz: $(DIR_BUILD)/$(BLENDER_FILE) $(DIR_BLENDER)/export_overlay_lines.py $(DIR_BUILD)/overlay_%.npz
#	@echo "Running blender overlay visibility detection"
#	$(BLENDER_BIN) $(DIR_BUILD)/$(BLENDER_FILE) --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay.npz --output $@
#
#$(DIR_BUILD)/overlay_%_cropped.npz: $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_%.npz $(DIR_BUILD)/overlay_visible_%.npz
#	@echo "Cropping visible overlay lines"
#	uv run $^ --output $@


$(DIR_BUILD)/overlay_pois_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_pois.npz $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
	@echo "Cropping visible overlay lines: POIs"
	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_pois.npz --output $(DIR_BUILD)/overlay_pois_projected.npz
	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_pois_projected.npz --output $(DIR_BUILD)/overlay_pois_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_pois_projected.npz $(DIR_BUILD)/overlay_pois_visible.npz --output $(DIR_BUILD)/overlay_pois_cropped.npz

$(DIR_BUILD)/overlay_grid_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_grid.npz $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
	@echo "Cropping visible overlay lines: GRID"
	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_grid.npz --output $(DIR_BUILD)/overlay_grid_projected.npz
	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_grid_projected.npz --output $(DIR_BUILD)/overlay_grid_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_grid_projected.npz $(DIR_BUILD)/overlay_grid_visible.npz --output $(DIR_BUILD)/overlay_grid_cropped.npz

$(DIR_BUILD)/overlay_axis_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_axis.npz $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
	@echo "Cropping visible overlay lines: AXIS"
	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_axis.npz --output $(DIR_BUILD)/overlay_axis_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_axis.npz $(DIR_BUILD)/overlay_axis_visible.npz --output $(DIR_BUILD)/overlay_axis_cropped.npz



$(DIR_BUILD)/contours.npz: $(DIR_SRC)/contours.py $(DIR_BUILD)/normals.exr $(DIR_BUILD)/raytrace.npy
	@echo "Computing contours"
	uv run $^ --output $@

$(DIR_BUILD)/mapping_color.png $(DIR_BUILD)/mapping_angle.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_background.png &: $(DIR_SRC)/process_blender.py $(DIR_BUILD)/normals.exr $(DIR_BUILD)/image.tif $(DIR_BUILD)/raytrace.npy $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/process_blender.toml
	@echo "Processing blender mappings: $@"
	uv run $(DIR_SRC)/process_blender.py 						\
		$(DIR_BUILD)/normals.exr 								\
		$(DIR_BUILD)/image.tif 									\
		$(DIR_BUILD)/raytrace.npy 								\
		--config $(DIR_BUILD)/process_blender.toml 				\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy 	\
		--output $(DIR_BUILD)									\
		--debug

$(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_brightness_difference.png &: $(DIR_SRC)/palette.py $(DIR_BUILD)/image.tif $(DIR_BUILD)/palette.toml
	@echo "Processing palette colors"
	uv run $(DIR_SRC)/palette.py $(DIR_BUILD)/image.tif			\
		--palette-mixture $(DIR_BUILD)/mapping_color.npy 		\
		--palette-brightness-difference $(DIR_BUILD)/mapping_brightness_difference.png \
		--config $(DIR_BUILD)/palette.toml


run: $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_angle.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_background.png
run: $(DIR_BUILD)/overlay_pois_cropped.npz $(DIR_BUILD)/overlay_grid_cropped.npz $(DIR_BUILD)/overlay_axis_cropped.npz $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/contours.npz
run: $(DIR_SRC)/hatch.py
	@echo "Hatch"
	uv run $(DIR_SRC)/hatch.py									\
		$(DIR_BUILD)/mapping_color.npy 							\
		$(DIR_BUILD)/mapping_angle.png 							\
		$(DIR_BUILD)/mapping_distance.png 						\
		$(DIR_BUILD)/mapping_line_length.png 					\
		$(DIR_BUILD)/mapping_background.png 					\
		--cutouts $(DIR_BUILD)/overlay_grid_cropped.npz 		\
		--overlays $(DIR_BUILD)/overlay_pois_cropped.npz $(DIR_BUILD)/overlay_axis_cropped.npz 		\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  \
		--contours $(DIR_BUILD)/contours.npz					\
		--config $(DIR_BUILD)/hatch.toml 						\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000


run_palette: $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_angle.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_background.png
run_palette: $(DIR_BUILD)/overlay_pois_cropped.npz $(DIR_BUILD)/overlay_grid_cropped.npz $(DIR_BUILD)/overlay_axis_cropped.npz $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/contours.npz $(DIR_BUILD)/hatch.toml
run_palette: $(DIR_SRC)/hatch.py
	@echo "Hatch Palette"
	uv run $(DIR_SRC)/hatch.py									\
		$(DIR_BUILD)/mapping_color.npy 							\
		$(DIR_BUILD)/mapping_angle.png 							\
		$(DIR_BUILD)/mapping_distance.png 						\
		$(DIR_BUILD)/mapping_line_length.png 					\
		$(DIR_BUILD)/mapping_background.png 					\
		--cutouts $(DIR_BUILD)/overlay_grid_cropped.npz 		\
		--overlays $(DIR_BUILD)/overlay_pois_cropped.npz $(DIR_BUILD)/overlay_axis_cropped.npz 		\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  \
		--contours $(DIR_BUILD)/contours.npz					\
		--config $(DIR_BUILD)/hatch.toml 						\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000

gcode: $(DIR_BUILD)/littleplanets.svg
	uv run svgtogcode.py $^

gcode_crop: $(DIR_BUILD)/littleplanets.svg
	uv run svgtogcode.py $^ --crop 375 375 100 400

# ----------


test: $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_angle.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_background.png
test: $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/hatch.toml
test: $(DIR_SRC)/hatch.py
	@echo "Hatch Palette"
	uv run $(DIR_SRC)/hatch.py									\
		$(DIR_BUILD)/mapping_color.npy 							\
		$(DIR_BUILD)/mapping_angle.png 							\
		$(DIR_BUILD)/mapping_distance.png 						\
		$(DIR_BUILD)/mapping_line_length.png 					\
		$(DIR_BUILD)/mapping_background.png 					\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  \
		--config $(DIR_BUILD)/hatch.toml 						\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000


test_angle: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_angle_0.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_background.png
	for i in mapping_angle_9.png mapping_angle_10.png ; do \
		echo "$$i"	; \
		uv run $(DIR_SRC)/hatch.py 					\
			$(DIR_BUILD)/mapping_color.npy 			\
			$(DIR_BUILD)/$$i 						\
			$(DIR_BUILD)/mapping_distance.png 		\
			$(DIR_BUILD)/mapping_line_length.png 	\
			$(DIR_BUILD)/mapping_background.png 	\
			--config $(DIR_BUILD)/hatch.toml 		\
			--projection-matrix $(DIR_BUILD)/projection_matrix.npy  \
			--output $(DIR_BUILD)/littleplanets.svg ; \
		$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=littleplanets_$$i --export-width=2000 --export-background=#000000 ; \
	done
