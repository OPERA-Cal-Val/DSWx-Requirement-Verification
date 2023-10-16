from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, model_validator

from dswx_verification.val_db import get_localized_validation_table


class VerificationParameters(BaseModel):
    pixels_sampled_per_trial: int
    pixels_distributed_with_respect_to_val: bool
    number_of_trials: int
    data_dir: str
    confidence_classes_to_strictly_include: Optional[list]
    rel_dswx_db_dir_path: Optional[Path] = None
    dswx_db_dir_parent: Optional[Path] = None

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
            # Update the absolute path of the parent
            self.dswx_db_dir_parent = self.rel_dswx_db_dir_path.parent.resolve()

            df_val = get_localized_validation_table()

            correct_db_name = df_val['rel_local_val_path'][0].split('/')[0]
            db_name_incorrect = (self.rel_dswx_db_dir_path.stem != correct_db_name)
            if db_name_incorrect:
                raise ValueError(f'Local database directory must be called: {correct_db_name}')

            # TODO: may want to add warning if not all datasets exists; after PO.DAAC Delivery
            # Check at least one validation dataset path exists
            val_paths = df_val['rel_local_val_path']
            val_paths_trunc = ['/'.join(p.split('/')[1:]) for p in val_paths]
            val_paths = [self.rel_dswx_db_dir_path / p_trunc for p_trunc in val_paths_trunc]
            if not any([p.exists() for p in val_paths]):
                raise ValueError('None of the necessary validation datasets do not exist in the local directory'
                                 f'specified e.g. {val_paths[0]}')

            # Check at least one DSWx Path exists
            dswx_paths_group_str = df_val['rel_local_dswx_paths']
            dswx_paths_str = [paths for group in dswx_paths_group_str for paths in group.split(' ')]
            dswx_paths_path = [self.rel_dswx_db_dir_path / '/'.join(p.split('/')[1:])
                               for p in dswx_paths_str]
            if not any([p.exists() for p in dswx_paths_path]):
                raise ValueError('None of the necessary DSWx paths were found; please verify you specify the relative '
                                 f'directory correctly e.g. {dswx_paths_path[0]}')
        return self
