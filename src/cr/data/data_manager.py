from cr.config.core import config

if config.general.RUN_ON_DATABRICKS_WS:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
else:
    from databricks.connect import DatabricksSession

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
    f"{config.model.MODEL_NAME}_{config.processing.TRAIN_DATA_NAME}"
)
BRONZE_FULL_TABLE = config.model.MODEL_NAME

# Silver Data
SILVER_SCHEMA_DICT = {
    env: f"{env}.silver" for env in ["dev", "staging", "test", "prod"]
}

SILVER_TABLE = config.model.MODEL_NAME

# Gold Data
GOLD_SCHEMA_DICT = {env: f"{env}.gold" for env in ["dev", "staging", "test", "prod"]}

GOLD_TABLE = config.model.MODEL_NAME


# Model asset schema (stores feautres, inference, and metric tables as well as models)
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
        f"CREATE DATABASE IF NOT EXISTS {BRONZE_SCHEMA_DICT[env]} LOCATION '{TRAINING_DATA_DIR}'"
    )


# Silver
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_SCHEMA_DICT[env]}")


# Gold
for env in ["dev", "staging", "test", "prod"]:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_SCHEMA_DICT[env]}")


# Model Assets
for env in ["dev", "staging", "test", "prod"]:
    # Create schema
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_ASSETS_SCHEMA_DICT[env]}")
    # Create an empty (to be used as placeholder) inference table with the client id (cifidnbr) and adjusted effective date columns
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {MODEL_ASSETS_SCHEMA_DICT[env]}.{INFERENCE_TABLE}(id string, time_period string) CLUSTER BY (time_period);"
    )


def get_schema_table_names(env: str) -> None:

    schema_table_name_dict = {
        "bronze_schema": BRONZE_SCHEMA_DICT[env],
        "bronze_incremental_table": BRONZE_INCREMENTAL_TABLE,
        "bronze_full_table": BRONZE_FULL_TABLE,
        "silver_schema": SILVER_SCHEMA_DICT[env],
        "silver_table": SILVER_TABLE,
        "gold_schema": GOLD_SCHEMA_DICT[env],
        "gold_table": GOLD_TABLE,
        "model_assets_schema": MODEL_ASSETS_SCHEMA_DICT[env],
        "inference_table": INFERENCE_TABLE,
    }

    return schema_table_name_dict
