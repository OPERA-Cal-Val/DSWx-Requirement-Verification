from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, model_validator


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
        return self
