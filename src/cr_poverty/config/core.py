import os
from pathlib import Path
from typing import Dict, List, Literal

# Pydantic
from pydantic import BaseModel

# Yaml
import yaml

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ROOT = PROJECT_ROOT.parent
CONFIG_DIR = Path(PACKAGE_ROOT / "config")
LOGS_DIR = Path(PROJECT_ROOT / "logs")
TESTS_DIR = Path(PROJECT_ROOT / "tests")


# Pydantic model for common configuration
class GeneralConfig(BaseModel):
    RANDOM_STATE: int
    DATABRICKS_WORKSPACE_URL: str
    RUN_ON_DATABRICKS_WS: bool


# Pydantic model for processing configuration
class ProcessingConfig(BaseModel):
    # Transformation Parameters
    ID_COLS: List[str]
    IND_BOOL_COLS: List[str]
    IND_ORDERED_COLS: List[str]
    HH_BOOL_COLS: List[str]
    HH_ORDERED_COLS: List[str]
    HH_CONT_COLS: List[str]
    SQR_COLS: List[str]
    REDUNDANT_HH_COLS: List[str]
    REDUNDANT_ELEC_COLS: List[str]
    SELECTED_IND_AGG_FEATURES: List[str]


# Pydantic model for model configuration
class ModelConfig(BaseModel):
    MODEL_NAME: str
    OUTCOME_VARIABLE: str
    POVERTY_MAPPING: Dict[str, str]
    PARALLELISM: int
    TIMEOUT_SECONDS: int
    TEST_SIZE: float
    N_FOLDS: int
    N_ITER: int
    CALIBRATION_SIZE: float
    PERFORM_CALIBRATION: bool
    CALIBRATION_METHOD: Literal["sigmoid", "isotonic"]
    PLOT_LEARNING_CURVES: bool
    PLOT_SHAP_VALUES: bool
    SNAP_THRESHOLD: int
    SHAP_TYPE: Literal["beeswarm", "violin", "layered_violin", "bar"]
    SELECT_FROM_LATEST_CV_RUNS: bool
    MIN_TEST_ROC_AUC: float
    MIN_TEST_AURPC_LIFT: float
    CLF__N_ESTIMATORS: int
    CLF__MAX_DEPTH: int
    CLF__MIN_SAMPLES_SPLIT: int
    CLF__MIN_SAMPLES_LEAF: int


# Master config object
class Config(BaseModel):
    """Master config object."""

    general: GeneralConfig
    processing: ProcessingConfig
    model: ModelConfig


def fetch_config_from_yaml(cfg_path: Path = None) -> object:
    """
    Parse YAML containing the package configuration

    Args:
        cfg_path (Path): validated path to configuration yaml. Defaults to None.

    Returns:
        YAML: parsed yaml object
    """
    # Input check: check if cfg_path is a valid file
    if cfg_path is None or not os.path.isfile(cfg_path):
        raise OSError(f"Did not find config file at path: {cfg_path}")

    with open(cfg_path, "r") as conf_file:
        parsed_config = yaml.safe_load(
            conf_file
        )  # use the yaml package to load the config
        return parsed_config


def create_and_validate_config(cfg_paths: Dict) -> Dict:
    """
    Run validation on config values

    Args:
        parsed_config (YAML): parsed content of the config yaml. Defaults to None.

    Returns:
        Config: final config object
    """
    # Fetch the general conifgurations from the yaml file
    parsed_general_config = fetch_config_from_yaml(cfg_paths["general"])
    general_config = GeneralConfig(**parsed_general_config)

    # Fetch the processing configurations from the yaml file
    parsed_processing_config = fetch_config_from_yaml(cfg_paths["processing"])
    processing_config = ProcessingConfig(**parsed_processing_config)

    # Fetch the model configurations from the yaml file
    parsed_feature_config = fetch_config_from_yaml(cfg_paths["model"])
    model_config = ModelConfig(**parsed_feature_config)

    # Combine all validated objects into a single object
    _config = Config(
        general=general_config,
        processing=processing_config,
        model=model_config,
    )

    return _config


# Load the configuration
config = create_and_validate_config(
    {
        "general": Path(CONFIG_DIR, "general_config.yaml"),
        "processing": Path(CONFIG_DIR, "processing_config.yaml"),
        "model": Path(CONFIG_DIR, "model_config.yaml"),
    },
)
