# littleplanets Makefile

.PHONY: all run run_clouds_coastlines clean

DIR_BLENDER := blender
BLENDER_BIN := /Applications/Blender.app/Contents/MacOS/Blender
BLENDER_FILE := template.blend

INKSCAPE_BIN := /Applications/Inkscape.app/Contents/MacOS/inkscape

PYPROJECT_FILE := pyproject.toml

DIR_SRC := src
DIR_DATA := data
DIR_BUILD := build
DIR_CONFIG := config
DIR_DEBUG := debug

CONFIG_FILE := $(DIR_CONFIG)/mars.toml
POI_FILES := $(DIR_CONFIG)/pois*.json
OUTPUT_PNG := littleplanets.png

# ----------

POI_DIR := $(dir $(POI_FILES))
GLOB_FILES := $(wildcard $(POI_FILES))
BASENAMES := $(notdir $(GLOB_FILES))
POI_OUTPUTS := $(addprefix $(DIR_BUILD)/overlay_poi_, $(addsuffix .npz, $(BASENAMES)))

# ----------

all: setup run

setup: $(PYPROJECT_FILE)
	@echo "Sync Environment"
	mkdir -p $(DIR_DATA)
	mkdir -p $(DIR_BUILD)
	mkdir -p $(DIR_DEBUG)
	uv sync

clean:
	rm -rf $(DIR_BUILD)

# ----------

$(DIR_BUILD)/%.toml: $(DIR_SRC)/configurator.py $(CONFIG_FILE)
	@echo "Configurator for file $@"
	uv run $^ --output $(DIR_BUILD)

$(DIR_DATA)/dem.tif $(DIR_DATA)/surface_color.tif $(DIR_DATA)/cds_clouds.nc: $(DIR_SRC)/downloader.py $(DIR_BUILD)/downloader.toml
	@echo "Downloader"
	uv run $(DIR_SRC)/downloader.py --output-dir $(DIR_DATA) --config $(DIR_BUILD)/downloader.toml

$(DIR_BUILD)/dem_scaled.tif: $(DIR_SRC)/modify_tiff.py $(DIR_DATA)/dem.tif $(DIR_BUILD)/modify_dem_stage1.toml
	@echo "Modify TIFF $@"
	uv run $(DIR_SRC)/modify_tiff.py $(DIR_DATA)/dem.tif $@ --config $(DIR_BUILD)/modify_dem_stage1.toml --pause-below-minimum-available-memory 4096

$(DIR_BUILD)/dem.tif: $(DIR_SRC)/modify_tiff.py $(DIR_BUILD)/dem_scaled.tif $(DIR_BUILD)/modify_dem_stage2.toml
	@echo "Modify TIFF $@"
	uv run $(DIR_SRC)/modify_tiff.py $(DIR_BUILD)/dem_scaled.tif $@ --config $(DIR_BUILD)/modify_dem_stage2.toml

$(DIR_BUILD)/surface_color.tif: $(DIR_SRC)/modify_tiff.py $(DIR_DATA)/surface_color.tif $(DIR_BUILD)/modify_surfacecolor.toml
	@echo "Modify TIFF $@"
	uv run $(DIR_SRC)/modify_tiff.py $(DIR_DATA)/surface_color.tif $@ --config $(DIR_BUILD)/modify_surfacecolor.toml --pause-below-minimum-available-memory 4096

$(DIR_BUILD)/mesh.ply: $(DIR_SRC)/mesh.py $(DIR_BUILD)/dem.tif $(DIR_BUILD)/surface_color.tif $(DIR_BUILD)/mesh.toml
	@echo "Generating mesh"
	uv run $(DIR_SRC)/mesh.py --elevation $(DIR_BUILD)/dem.tif --color $(DIR_BUILD)/surface_color.tif --output $@ --config $(DIR_BUILD)/mesh.toml

$(DIR_BUILD)/blender_paths.blend: $(DIR_BLENDER)/$(BLENDER_FILE) $(DIR_BLENDER)/adjust_paths.py
	@echo "Adjusting blender paths"
	cp $(DIR_BLENDER)/$(BLENDER_FILE) $@
	$(BLENDER_BIN) $(DIR_BUILD)/blender_paths.blend --background --python $(DIR_BLENDER)/adjust_paths.py -- --render-output-dir $(DIR_BUILD)

$(DIR_BUILD)/blender_scene.blend: $(DIR_BUILD)/blender_paths.blend $(DIR_SRC)/blender_wrapper.py $(DIR_BLENDER)/adjust_scene.py $(DIR_BUILD)/adjust_scene.toml
	@echo "Running blender scene update"
	cp $(DIR_BUILD)/blender_paths.blend $@
	uv run $(DIR_SRC)/blender_wrapper.py $(BLENDER_BIN) $@ $(DIR_BLENDER)/adjust_scene.py --config $(DIR_BUILD)/adjust_scene.toml

$(DIR_BUILD)/blender_mesh.blend: $(DIR_BUILD)/blender_scene.blend $(DIR_SRC)/blender_wrapper.py $(DIR_BLENDER)/import_ply.py $(DIR_BUILD)/mesh.ply $(DIR_BUILD)/import_ply.toml
	@echo "Running blender mesh update"
	cp $(DIR_BUILD)/blender_scene.blend $@
	uv run $(DIR_SRC)/blender_wrapper.py $(BLENDER_BIN) $@ $(DIR_BLENDER)/import_ply.py --config $(DIR_BUILD)/import_ply.toml --params "--input $(DIR_BUILD)/mesh.ply"

$(DIR_BUILD)/raytrace.npy: $(DIR_BUILD)/blender_mesh.blend $(DIR_BLENDER)/raytracing.py
	@echo "Running blender raytracer"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background --python $(DIR_BLENDER)/raytracing.py -- --output $@

$(DIR_BUILD)/normals.exr $(DIR_BUILD)/image.tif &: $(DIR_BUILD)/blender_mesh.blend
	@echo "Running blender renderer"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background -f 0
	mv $(DIR_BUILD)/Normals0000.exr $(DIR_BUILD)/normals.exr
	mv $(DIR_BUILD)/Image0000.tif $(DIR_BUILD)/image.tif
	# mv $(DIR_BUILD)/Image0000.png $(DIR_BUILD)/freestyle.png

$(DIR_BUILD)/projection_matrix.npy: $(DIR_BUILD)/blender_mesh.blend $(DIR_BLENDER)/export_projection_matrix.py
	@echo "Running blender P matrix exporter"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background --python $(DIR_BLENDER)/export_projection_matrix.py -- --output $@

$(DIR_BUILD)/mesh_blender.ply: $(DIR_BUILD)/blender_mesh.blend $(DIR_BLENDER)/export_ply.py
	@echo "Running blender mesh export"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh.blend --background --python $(DIR_BLENDER)/export_ply.py -- --output $@

# Overlay Coastlines

$(DIR_BUILD)/overlay_coastlines.npz: $(DIR_SRC)/overlay_coastlines.py $(DIR_BUILD)/dem.tif $(DIR_BUILD)/overlay_coastlines.toml
	@echo "Create overlay Coastlines"
	uv run $(DIR_SRC)/overlay_coastlines.py $(DIR_BUILD)/dem.tif --output $@ --config $(DIR_BUILD)/overlay_coastlines.toml

$(DIR_BUILD)/overlay_coastlines_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_coastlines.npz $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
	@echo "Cropping visible overlay lines: Coastlines"
	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_coastlines.npz --output $(DIR_BUILD)/overlay_coastlines_projected.npz
	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_coastlines_projected.npz --output $(DIR_BUILD)/overlay_coastlines_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_coastlines_projected.npz $(DIR_BUILD)/overlay_coastlines_visible.npz --output $(DIR_BUILD)/overlay_coastlines_cropped.npz

# Overlay POIs

# $(DIR_BUILD)/overlay_pois.npz: $(DIR_SRC)/overlay_pois.py $(POI_FILE) $(DIR_BUILD)/overlay_pois.toml
# 	@echo "Create overlay POIs"
# 	uv run $(DIR_SRC)/overlay_pois.py $(POI_FILE) --output $@ --config $(DIR_BUILD)/overlay_pois.toml
#
# $(DIR_BUILD)/overlay_pois_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_pois.npz $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
# 	@echo "Cropping visible overlay lines: POIs"
# 	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
# 	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_pois.npz --output $(DIR_BUILD)/overlay_pois_projected.npz
# 	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_pois_projected.npz --output $(DIR_BUILD)/overlay_pois_visible.npz
# 	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_pois_projected.npz $(DIR_BUILD)/overlay_pois_visible.npz --output $(DIR_BUILD)/overlay_pois_cropped.npz

$(DIR_BUILD)/overlay_poi_%.npz: $(POI_DIR)/% $(DIR_SRC)/overlay_pois.py $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_pois.toml
	@echo "Create overlay POI: $< -> $@"

	uv run $(DIR_SRC)/overlay_pois.py $< --output $@_raw --config $(DIR_BUILD)/overlay_pois.toml

	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $@_raw --output $@_projected

	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $@_projected --output $@_visible
	uv run $(DIR_SRC)/overlay_crop.py $@_projected $@_visible --output $@


# Overlay Grid

$(DIR_BUILD)/overlay_grid.npz: $(DIR_SRC)/overlay_grid.py $(DIR_BUILD)/overlay_grid.toml
	@echo "Create overlay Grid"
	uv run $(DIR_SRC)/overlay_grid.py --output $@ --config $(DIR_BUILD)/overlay_grid.toml

$(DIR_BUILD)/overlay_grid_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_grid.npz $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
	@echo "Cropping visible overlay lines: GRID"
	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	uv run $(DIR_SRC)/overlay_project.py $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_grid.npz --output $(DIR_BUILD)/overlay_grid_projected.npz
	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_grid_projected.npz --output $(DIR_BUILD)/overlay_grid_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_grid_projected.npz $(DIR_BUILD)/overlay_grid_visible.npz --output $(DIR_BUILD)/overlay_grid_cropped.npz

# Overlay Contours

$(DIR_BUILD)/overlay_contours.npz: $(DIR_SRC)/overlay_contours.py $(DIR_BUILD)/raytrace.npy $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/overlay_contours.toml
	@echo "Create overlay Contours"
	uv run $(DIR_SRC)/overlay_contours.py $(DIR_BUILD)/raytrace.npy  --output $@ --projection-matrix $(DIR_BUILD)/projection_matrix.npy --config $(DIR_BUILD)/overlay_contours.toml

$(DIR_BUILD)/overlay_contours_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/overlay_contours.npz $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
	@echo "Cropping visible overlay lines: CONTOURS"
	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_contours.npz --output $(DIR_BUILD)/overlay_contours_visible.npz --raycast-from-light
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_contours.npz $(DIR_BUILD)/overlay_contours_visible.npz --output $(DIR_BUILD)/overlay_contours_cropped.npz

# Overlay Axis

$(DIR_BUILD)/overlay_axis.npz: $(DIR_SRC)/overlay_axis.py $(DIR_BUILD)/overlay_axis.toml
	@echo "Create overlay Axis"
	uv run $(DIR_SRC)/overlay_axis.py --output $@ --config $(DIR_BUILD)/overlay_axis.toml

$(DIR_BUILD)/overlay_axis_cropped.npz: $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/mesh_blender.ply $(DIR_BUILD)/overlay_axis.npz $(DIR_SRC)/overlay_project.py $(DIR_BLENDER)/export_overlay_lines.py $(DIR_SRC)/overlay_crop.py
	@echo "Cropping visible overlay lines: AXIS"
	rsync -au $(DIR_BUILD)/blender_mesh.blend $(DIR_BUILD)/blender_overlays.blend
	$(BLENDER_BIN) $(DIR_BUILD)/blender_overlays.blend --background --python $(DIR_BLENDER)/export_overlay_lines.py -- --input $(DIR_BUILD)/overlay_axis.npz --output $(DIR_BUILD)/overlay_axis_visible.npz
	uv run $(DIR_SRC)/overlay_crop.py $(DIR_BUILD)/overlay_axis.npz $(DIR_BUILD)/overlay_axis_visible.npz --output $(DIR_BUILD)/overlay_axis_cropped.npz

# Overlay Clouds

$(DIR_BUILD)/mesh_clouds.ply: $(DIR_SRC)/mesh.py $(DIR_BUILD)/mesh_clouds.toml
	@echo "Generating mesh $@"
	uv run $(DIR_SRC)/mesh.py --output $@ --config $(DIR_BUILD)/mesh_clouds.toml

$(DIR_BUILD)/blender_mesh_clouds.blend: $(DIR_BUILD)/blender_mesh.blend $(DIR_SRC)/blender_wrapper.py $(DIR_BLENDER)/import_ply.py $(DIR_BUILD)/mesh_clouds.ply $(DIR_BUILD)/import_ply_clouds.toml
	@echo "Running blender mesh update"
	cp $(DIR_BUILD)/blender_mesh.blend $@
	uv run $(DIR_SRC)/blender_wrapper.py $(BLENDER_BIN) $@ $(DIR_BLENDER)/import_ply.py --config $(DIR_BUILD)/import_ply_clouds.toml --params "--input $(DIR_BUILD)/mesh_clouds.ply"

$(DIR_BUILD)/raytrace_clouds.npy $(DIR_BUILD)/raytrace_backface_clouds.npy: $(DIR_BUILD)/blender_mesh_clouds.blend $(DIR_BLENDER)/raytracing.py
	@echo "Running blender raytracer"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh_clouds.blend 		\
		--background --python $(DIR_BLENDER)/raytracing.py 		\
		-- 														\
		--output $(DIR_BUILD)/raytrace_clouds.npy 				\
		--output-backface $(DIR_BUILD)/raytrace_backface_clouds.npy \
		--filter-object-name mesh_clouds

$(DIR_BUILD)/normals_clouds.exr $(DIR_BUILD)/image_clouds.tif &: $(DIR_BUILD)/blender_mesh_clouds.blend
	@echo "Running blender renderer"
	$(BLENDER_BIN) $(DIR_BUILD)/blender_mesh_clouds.blend --background -f 0 || true
	mv $(DIR_BUILD)/Normals0000.exr $(DIR_BUILD)/normals_clouds.exr
	mv $(DIR_BUILD)/Image0000.tif $(DIR_BUILD)/image_clouds.tif

$(DIR_BUILD)/clouds_mapping_front_angle.png $(DIR_BUILD)/clouds_mapping_front_distance.png $(DIR_BUILD)/clouds_mapping_front_background.png $(DIR_BUILD)/clouds_mapping_back_angle.png $(DIR_BUILD)/clouds_mapping_back_distance.png $(DIR_BUILD)/clouds_mapping_back_background.png: $(DIR_SRC)/overlay_clouds.py $(DIR_BUILD)/raytrace_clouds.npy $(DIR_BUILD)/raytrace_backface_clouds.npy $(DIR_DATA)/cds_clouds.nc $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/overlay_clouds.toml
	@echo "Create clouds overlay"
	uv run $(DIR_SRC)/overlay_clouds.py 						\
		$(DIR_BUILD)/raytrace_clouds.npy 						\
		$(DIR_BUILD)/raytrace_backface_clouds.npy 				\
		$(DIR_DATA)/cds_clouds.nc								\
		$(DIR_BUILD)/projection_matrix.npy 						\
		--output $(DIR_BUILD)									\
		--config $(DIR_BUILD)/overlay_clouds.toml

$(DIR_BUILD)/overlay_clouds_front.npz: $(DIR_SRC)/hatch.py $(DIR_BUILD)/clouds_mapping_front_angle.png $(DIR_BUILD)/clouds_mapping_front_distance.png $(DIR_BUILD)/clouds_mapping_front_background.png $(DIR_BUILD)/hatch.toml
	@echo "Hatch clouds overlay front"
	uv run $(DIR_SRC)/hatch.py													\
		--mapping-angle $(DIR_BUILD)/clouds_mapping_front_angle.png 			\
		--mapping-distance $(DIR_BUILD)/clouds_mapping_front_distance.png 		\
		--mapping-background $(DIR_BUILD)/clouds_mapping_front_background.png 	\
		--config $(DIR_BUILD)/overlay_clouds_hatch.toml 							\
		--output $@

$(DIR_BUILD)/overlay_clouds_back.npz: $(DIR_SRC)/hatch.py $(DIR_BUILD)/clouds_mapping_back_angle.png $(DIR_BUILD)/clouds_mapping_back_distance.png $(DIR_BUILD)/clouds_mapping_back_background.png $(DIR_BUILD)/hatch.toml
	@echo "Hatch clouds overlay back"
	uv run $(DIR_SRC)/hatch.py													\
		--mapping-angle $(DIR_BUILD)/clouds_mapping_back_angle.png 				\
		--mapping-distance $(DIR_BUILD)/clouds_mapping_back_distance.png 		\
		--mapping-background $(DIR_BUILD)/clouds_mapping_back_background.png 	\
		--config $(DIR_BUILD)/overlay_clouds_hatch.toml 							\
		--output $@

# Targets

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

$(DIR_BUILD)/hatchlines.npz: $(DIR_SRC)/hatch.py $(DIR_BUILD)/mapping_angle.png $(DIR_BUILD)/mapping_distance.png $(DIR_BUILD)/mapping_line_length.png $(DIR_BUILD)/mapping_background.png $(DIR_BUILD)/hatch.toml
	@echo "Hatch"
	uv run $(DIR_SRC)/hatch.py											\
		--mapping-angle $(DIR_BUILD)/mapping_angle.png 					\
		--mapping-distance $(DIR_BUILD)/mapping_distance.png 			\
		--mapping-line-length $(DIR_BUILD)/mapping_line_length.png 		\
		--mapping-background $(DIR_BUILD)/mapping_background.png 		\
		--config $(DIR_BUILD)/hatch.toml 								\
		--output $@

# ----------

run: $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_background.png $(DIR_BUILD)/hatchlines.npz
run: $(POI_OUTPUTS) $(DIR_BUILD)/overlay_grid_cropped.npz $(DIR_BUILD)/overlay_axis_cropped.npz $(DIR_BUILD)/overlay_contours_cropped.npz $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/combine.toml
run: $(DIR_SRC)/combine.py
	@echo "Combine"
	uv run $(DIR_SRC)/combine.py									\
		--mapping-color $(DIR_BUILD)/mapping_color.npy 				\
		--mapping-background $(DIR_BUILD)/mapping_background.png 	\
		--hatchlines $(DIR_BUILD)/hatchlines.npz					\
		--cutouts $(DIR_BUILD)/overlay_grid_cropped.npz 			\
		--overlays 													\
			$(POI_OUTPUTS) 					                        \
			$(DIR_BUILD)/overlay_axis_cropped.npz  					\
		--contours $(DIR_BUILD)/overlay_contours_cropped.npz		\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  	\
		--config $(DIR_BUILD)/combine.toml 							\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000

run_clouds_coastlines: $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_background.png $(DIR_BUILD)/hatchlines.npz
run_clouds_coastlines: $(POI_OUTPUTS) $(DIR_BUILD)/overlay_grid_cropped.npz $(DIR_BUILD)/overlay_axis_cropped.npz $(DIR_BUILD)/overlay_coastlines_cropped.npz $(DIR_BUILD)/overlay_clouds_front.npz $(DIR_BUILD)/overlay_clouds_back.npz $(DIR_BUILD)/overlay_contours_cropped.npz $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/combine.toml
run_clouds_coastlines: $(DIR_SRC)/combine.py
	@echo "Combine"
	uv run $(DIR_SRC)/combine.py									\
		--mapping-color $(DIR_BUILD)/mapping_color.npy 				\
		--mapping-background $(DIR_BUILD)/mapping_background.png 	\
		--hatchlines $(DIR_BUILD)/hatchlines.npz					\
		--cutouts $(DIR_BUILD)/overlay_grid_cropped.npz 			\
		--overlays 													\
			$(POI_OUTPUTS) 					                        \
			$(DIR_BUILD)/overlay_axis_cropped.npz 					\
			$(DIR_BUILD)/overlay_coastlines_cropped.npz 			\
			$(DIR_BUILD)/overlay_clouds_back.npz 					\
			$(DIR_BUILD)/overlay_clouds_front.npz 					\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  	\
		--contours $(DIR_BUILD)/overlay_contours_cropped.npz		\
		--config $(DIR_BUILD)/combine.toml 							\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000

run_no_overlays: $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_background.png $(DIR_BUILD)/hatchlines.npz
run_no_overlays: $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/combine.toml
run_no_overlays: $(DIR_SRC)/combine.py
	@echo "Combine"
	uv run $(DIR_SRC)/combine.py									\
		--mapping-color $(DIR_BUILD)/mapping_color.npy 				\
		--mapping-background $(DIR_BUILD)/mapping_background.png 	\
		--hatchlines $(DIR_BUILD)/hatchlines.npz					\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  	\
		--config $(DIR_BUILD)/combine.toml 							\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000


run_no_axis: $(DIR_BUILD)/mapping_color.npy $(DIR_BUILD)/mapping_background.png $(DIR_BUILD)/hatchlines.npz
run_no_axis: $(POI_OUTPUTS) $(DIR_BUILD)/overlay_grid_cropped.npz $(DIR_BUILD)/overlay_axis_cropped.npz $(DIR_BUILD)/overlay_contours_cropped.npz $(DIR_BUILD)/projection_matrix.npy $(DIR_BUILD)/combine.toml
run_no_axis: $(DIR_SRC)/combine.py
	@echo "Combine"
	uv run $(DIR_SRC)/combine.py									\
		--mapping-color $(DIR_BUILD)/mapping_color.npy 				\
		--mapping-background $(DIR_BUILD)/mapping_background.png 	\
		--hatchlines $(DIR_BUILD)/hatchlines.npz					\
		--cutouts $(DIR_BUILD)/overlay_grid_cropped.npz 			\
		--overlays 													\
			$(POI_OUTPUTS) 					                        \
		--contours $(DIR_BUILD)/overlay_contours_cropped.npz		\
		--projection-matrix $(DIR_BUILD)/projection_matrix.npy  	\
		--config $(DIR_BUILD)/combine.toml 							\
		--output $(DIR_BUILD)/littleplanets.svg
	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000


# Postprocessing

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

test_hatch:

	uv run $(DIR_SRC)/hatch.py											\
		--mapping-angle debug/clouds_mapping_angle.png 					\
		--mapping-distance debug/clouds_mapping_clouds.png 				\
		--config $(DIR_BUILD)/hatch.toml 								\
		--output $(DIR_BUILD)/hatchlines.npz

	uv run $(DIR_SRC)/combine.py								\
		--hatchlines $(DIR_BUILD)/hatchlines.npz				\
		--config $(DIR_BUILD)/combine.toml 						\
		--output $(DIR_BUILD)/littleplanets.svg

	$(INKSCAPE_BIN) $(DIR_BUILD)/littleplanets.svg --export-filename=$(OUTPUT_PNG) --export-width=2000 --export-background=#000000