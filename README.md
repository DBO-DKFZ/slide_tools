Getting rid of pre-saved tiles for good
---------------------------------------

Pipelines involving WSIs usually pre-save tiles. This can have several downsides (storage requirements, unflexible, ...) but one huge upside: speed.  
This repo is a collection of useful tools to handle WSIs as fast as currently possible without pre-saving any tile and thereby getting rid of the downsides.

Motivation
------------
WSIs are saved in a hierarchical format and are not encoded as a whole but already consist of tiles internally (usually of the size 240x240 pixels).
This begs the question of why aren't we using these tiles? The reason mostly comes down to speed and pipeline complexity which are both ideally solved by this repo.

The biggest hit to loading speed comes from the fact that the internal tiles are usually encoded in [JPEG2000](https://en.wikipedia.org/wiki/JPEG_2000). Decoding JPEG2000 is much more taxing than decoding JPEG. This repo is based on [cuCIM](https://github.com/rapidsai/cucim) specifically because of the faster JPEG2000 decoding compared to [openslide](https://github.com/openslide/openslide-python).

This repo is designed to also handle annotations (GeoJSON from [QuPath](https://github.com/qupath/qupath)) and labels. The labels should follow this format:

```yaml
{
    "label_1": {
        "points": [[x_1, y_1], ...],  <-- pixel coordinatines at level=0
        "values": [v_11, ...]         <-- one scalar or vector per point 
    }
    "label_2": ...
}
```

Labels are interpolated (nearest or linear) to cover the whole slide.


Notebooks
---------
- [Basic usage](notebooks/basic_usage.ipynb)
- [Filtering tiles by label](notebooks/filtering_by_label.ipynb)
- [Tile-Level Dataset](notebooks/tile_level_dataset.ipynb)
- [Tile-Level DataModule](notebooks/tile_level_datamodule.ipynb)



Installation
------------

1) Make sure your current environment fulfills the requirements: `pip install -r requirements.txt`
2) Install the package in editable mode in your environment: `pip install -e .`


How to contribute
-----------------

There are several ways in which you could contribute:  
-  Create an issue because e.g. your use case is not covered yet or you found a bug  
-  Solve any already specified issue
-  Enhance the package by writing new use cases or enhance existing ones  

Every contribution has to happen inside a new branch from which you should then make a pull request.  

Please run [isort](https://github.com/PyCQA/isort) and [black](https://github.com/psf/black) before making pull requests:  
`isort -profile black slide_tools && black slide_tools`
