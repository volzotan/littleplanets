![Header](media/header.jpg)

Littleplanets: a pipeline to generate paths for pen plotters from elevation and color data of celestial bodies.

# Dependencies:

* [uv](https://docs.astral.sh/uv/), the python package manager
* [Inkscape](https://inkscape.org/), for SVG-to-PNG conversion
* Blender
* wget

On Linux you will need to modify BLENDER_BIN path in `makefile` or export it as an environment variable.

# Generate the SVGs

Adjust Blender path and inkscape path in Makefile, then:

```sh
make setup run CONFIG_FILE=config/moon.toml DIR_BUILD=build_moon DIR_DATA=data_moon POI_FILES="config/moon_poi*.json" OUTPUT_PNG=moon.png
```

or

```sh
sh make_all.sh
```

# Config

Find the config files for each planet in [config](/config), example [config/moon.toml](config/moon.toml).

Note: for Earth cloud overlays, a [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) API key is required (`~/.cdsapirc`).

---

Refer to a more extensive, but LLM-generated ReadMe: [README_GENERATED.md](README_GENERATED.md). 

---

# License Notes:

The stroke-based Hershey Fonts used in this project have been curated by [Windell Oskay](http://windell.oskay.net/) and made available in his [SVG Fonts](https://gitlab.com/oskay/svg-fonts) repository. They are available under the [SIL Open Font License](https://openfontlicense.org/) and are included without alterations in this repository for convenience. The licensing of the code in this project does not extend to the fonts.