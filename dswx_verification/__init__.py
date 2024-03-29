import warnings
from importlib.metadata import PackageNotFoundError, version

from .labels import (reclassify_validation_dataset_to_dswx_frame,
                     resample_label_into_percentage)
from .latex import (get_main_beamer_tex_template, get_slide_tex_template,
                    render_latex_template)
from .mask_utils import (get_contiguous_areas_of_class_with_maximum_size,
                         get_number_of_pixels_in_hectare)
from .metrics import get_all_metrics_for_one_trial
from .random_sample import (generate_random_indices_for_classes,
                            get_equal_samples_per_label)
from .rio_tools import get_geopandas_features_from_array
from .val_db import (generate_linked_id_table_for_classified_imagery,
                     get_localized_validation_table,
                     get_path_of_validation_geojson,
                     get_validation_metadata_by_site_name)

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = None
    warnings.warn('package is not installed!\n'
                  'Install in editable/develop mode via (from the top of this repo):\n'
                  '   python -m pip install -e .\n', RuntimeWarning)

__all__ = ['generate_linked_id_table_for_classified_imagery',
           'get_path_of_validation_geojson',
           'get_localized_validation_table',
           'get_validation_metadata_by_site_name',
           'reclassify_validation_dataset_to_dswx_frame',
           'resample_label_into_percentage',
           'get_contiguous_areas_of_class_with_maximum_size',
           'get_number_of_pixels_in_hectare',
           'get_equal_samples_per_label',
           'generate_random_indices_for_classes',
           'get_all_metrics_for_one_trial',
           'get_geopandas_features_from_array',
           'get_main_beamer_tex_template',
           'get_slide_tex_template',
           'render_latex_template'
           ]
