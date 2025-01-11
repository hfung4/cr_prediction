from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

from cr_poverty.config.core import config


if config.general.RUN_ON_DATABRICKS_WS:
    spark = SparkSession.builder.getOrCreate()


"""
Schema and Table names in the Hive Meta Store (to be changed/replaced once the project is migrated to UC sandbox)
"""

DATA_STORAGE_EXTERNAL_LOCATION = (
    "s3://thrivent-prd-datalake-analytics-workspace-east/DSI"
)


# Bronze data
BRONZE_SCHEMA_DICT = {
    env: f"aw_ds_ptb_bronze_{env}" for env in ["dev", "staging", "test", "prod"]
}

BRONZE_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{BRONZE_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
}

BRONZE_ALL_MEMBERS_TABLE = "all_members_monthly_vw"
BRONZE_CHOREOGRAPH_TABLE = "choreograph"
BRONZE_RESPONDER_TABLE = "ltc_product_purchased"

# Silver Data
SILVER_SCHEMA_DICT = {
    env: f"aw_ds_ptb_silver_{env}" for env in ["dev", "staging", "test", "prod"]
}
SILVER_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{SILVER_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
}

SILVER_ALL_MEMBERS_TABLE = f"all_members_{config.model.MODEL_NAME}"
SILVER_CHOREOGRAPH_TABLE = f"choreograph_{config.model.MODEL_NAME}"
SILVER_RESPONDER_TABLE = f"responder_{config.model.MODEL_NAME}"

# Serving data (TODO: to be remove once we can use feature tore)
SILVER_ALL_MEMBERS_TABLE_SERVING = f"all_members_{config.model.MODEL_NAME}_serving"
SILVER_CHOREOGRAPH_TABLE_SERVING = f"choreograph_{config.model.MODEL_NAME}_serving"


# Gold Data
GOLD_SCHEMA_DICT = {
    env: f"aw_ds_ptb_gold_{env}" for env in ["dev", "staging", "test", "prod"]
}
GOLD_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{GOLD_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
}
GOLD_TABLE = config.model.MODEL_NAME

# Serving data (TODO: to be remove once we can use feature tore)
GOLD_TABLE_SERVING = f"{config.model.MODEL_NAME}_serving"


# Model asset schema (stores feautres, inference, and metric tables as well as models (when we use UC), Volumes, and functions)
MODEL_ASSETS_SCHEMA_DICT = {
    env: f"aw_ds_ptb_ml_assets_{env}" for env in ["dev", "staging", "test", "prod"]
}

MODEL_ASSETS_DATA_STORAGE_PATH_DICT = {
    env: f"{DATA_STORAGE_EXTERNAL_LOCATION}/{MODEL_ASSETS_SCHEMA_DICT[env]}"
    for env in ["dev", "staging", "test", "prod"]
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
