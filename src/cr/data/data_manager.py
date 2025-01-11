from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession


from cr.config.core import config

if config.general.RUN_ON_DATABRICKS_WS:
    spark = SparkSession.builder.getOrCreate()
else:
    spark = DatabricksSession.builder.getOrCreate()


# External data storage location
RAW_DATA_EXTERNAL_LOCATION = (
    "abfss://costa-rica-poverty-data@hfdbsa.dfs.core.windows.net/"
)
TRAINING_DATA_DIR = f"{RAW_DATA_EXTERNAL_LOCATION}/raw-train"
SERVING_DATA_DIR = f"{RAW_DATA_EXTERNAL_LOCATION}/raw-serving"

TRAIN_DATA_NAME = config.processing.TRAIN_DATA_NAME

# Bronze data
BRONZE_SCHEMA_DICT = {
    env: f"{env}.bronze" for env in ["dev", "staging", "test", "prod"]
}

BRONZE_INCREMENTAL_TABLE = (
    f"{config.model.MODEL_NAME}_{config.genereal.TRAIN_DATA_NAME}"
)
BRONZE_FULL_TABLE = config.model.MODEL_NAME

# Silver Data
SILVER_SCHEMA_DICT = {
    env: f"{env}.silver" for env in ["dev", "staging", "test", "prod"]
}

SILVER_TABLE = config.model.MODEL_NAME

# Gold Data
BRONZE_SCHEMA_DICT = {env: f"{env}.gold" for env in ["dev", "staging", "test", "prod"]}

GOLD_TABLE = config.model.MODEL_NAME


# Model asset schema (stores feautres, inference, and metric tables as well as models (when we use UC), Volumes, and functions)
MODEL_ASSETS_SCHEMA_DICT = {
    env: f"{env}.cr_ml_assets" for env in ["dev", "staging", "test", "prod"]
}

INFERENCE_TABLE = f"inference_{config.model.MODEL_NAME}"
# METRICS_TABLE


"""
Create schemas
"""

# Bronze
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {BRONZE_SCHEMA_DICT[env]} LOCATION '{BRONZE_DATA_STORAGE_PATH_DICT[env]}'"
    )


# Silver
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {SILVER_SCHEMA_DICT[env]} LOCATION '{SILVER_DATA_STORAGE_PATH_DICT[env]}'"
    )


# Gold
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {GOLD_SCHEMA_DICT[env]} LOCATION '{GOLD_DATA_STORAGE_PATH_DICT[env]}'"
    )


# Model Assets
for env in ["dev", "staging", "test", "prod"]:
    # Create schema
    spark.sql(
        f"CREATE DATABASE IF NOT EXISTS {MODEL_ASSETS_SCHEMA_DICT[env]} LOCATION '{MODEL_ASSETS_DATA_STORAGE_PATH_DICT[env]}'"
    )
    # Create an empty (to be used as placeholder) inference table with the client id (cifidnbr) and adjusted effective date columns
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {MODEL_ASSETS_SCHEMA_DICT[env]}.{INFERENCE_TABLE}(cifidnbr string, adjusted_effective_date string) CLUSTER BY (adjusted_effective_date) LOCATION '{MODEL_ASSETS_DATA_STORAGE_PATH_DICT[env]}/{INFERENCE_TABLE}';"
    )


def get_schema_table_names(env: str) -> None:

    schema_table_name_dict = {
        "bronze_schema": BRONZE_SCHEMA_DICT[env],
        "bronze_all_members_table": BRONZE_ALL_MEMBERS_TABLE,
        "bronze_choreograph_table": BRONZE_CHOREOGRAPH_TABLE,
        "bronze_responder_table": BRONZE_RESPONDER_TABLE,
        "silver_schema": SILVER_SCHEMA_DICT[env],
        "silver_data_storage_path": SILVER_DATA_STORAGE_PATH_DICT[env],
        "silver_all_members_table": SILVER_ALL_MEMBERS_TABLE,
        "silver_choreograph_table": SILVER_CHOREOGRAPH_TABLE,
        "silver_all_members_table_serving": SILVER_ALL_MEMBERS_TABLE_SERVING,
        "silver_choreograph_table_serving": SILVER_CHOREOGRAPH_TABLE_SERVING,
        "silver_responder_table": SILVER_RESPONDER_TABLE,
        "gold_schema": GOLD_SCHEMA_DICT[env],
        "gold_data_storage_path": GOLD_DATA_STORAGE_PATH_DICT[env],
        "gold_table": GOLD_TABLE,
        "gold_table_serving": GOLD_TABLE_SERVING,
        "model_assets_schema": MODEL_ASSETS_SCHEMA_DICT[env],
        "model_assets_data_storage_path": MODEL_ASSETS_DATA_STORAGE_PATH_DICT[env],
        "inference_table": INFERENCE_TABLE,
    }

    return schema_table_name_dict
