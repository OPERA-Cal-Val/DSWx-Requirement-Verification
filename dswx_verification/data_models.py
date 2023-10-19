from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, model_validator

from .val_db import get_localized_validation_table


class VerificationParameters(BaseModel):
    pixels_sampled_per_trial: int
    pixels_distributed_with_respect_to_val: bool
    number_of_trials: int
    data_dir: str
    confidence_classes_to_strictly_include: Optional[list]
    rel_dswx_db_dir_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, yaml_file: str | Path) -> "VerificationParameters":
        with open(yaml_file) as f:
            params_dict = yaml.safe_load(f)

        verif_params = params_dict['verification_parameters']
        obj = VerificationParameters(**verif_params)
        return obj

    @model_validator(mode='after')
    def check_rel_dswx_db_dir_exists(self) -> 'VerificationParameters':
        if self.rel_dswx_db_dir_path is not None:
            if not self.rel_dswx_db_dir_path.exists():
                raise ValueError('The specified location of the local DSWx DB does not exist; '
                                 'if running this via papermill, make sure your relative path is with respect to '
                                 'the runtime directory.')
        return self

    @model_validator(mode='after')
    def check_if_local_data_exist(self) -> 'VerificationParameters':
        if self.rel_dswx_db_dir_path is not None:
            df_val = get_localized_validation_table()
            # TODO: may want to add warning if not all datasets exists; after PO.DAAC Delivery
            # Check at least one validation dataset path exists
            val_paths = [self.rel_dswx_db_dir_path / p for p in df_val['rel_local_val_path']]
            if not any([p.exists() for p in val_paths]):
                raise ValueError('The necessary validation datasets do not exist in the local directory'
                                 f'specified e.g. {val_paths[0]}')

            # Check at least one DSWx Path exists
            dswx_paths_group_str = df_val['rel_local_dswx_paths']
            dswx_paths_str = [paths for group in dswx_paths_group_str for paths in group.split(' ')]
            dswx_paths_path = [self.rel_dswx_db_dir_path / p for p in dswx_paths_str]
            if not any([p.exists() for p in dswx_paths_path]):
                raise ValueError('The necessary DSWx paths were not found; please verify you specified the relative '
                                 f'directory correctly e.g. {dswx_paths_path[0]}')
        return self
