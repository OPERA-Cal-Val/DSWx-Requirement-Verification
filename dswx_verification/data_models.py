from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class VerificationParameters(BaseModel):
    pixels_sampled_per_trial: int
    pixels_distributed_with_respect_to_val: bool
    number_of_trials: int
    confidence_classes_to_strictly_include: Optional[list]
    data_dir = str

    @classmethod
    def from_yaml(cls, yaml_file: str | Path) -> "VerificationParameters":
        with open(yaml_file) as f:
            params_dict = yaml.safe_load(f)
        obj = VerificationParameters(**params_dict['verification_parameters'])
        return obj
