"""
* The bronze.py performs the following tasks in certain data projects:
* Ingest Data from external data storage such as AWS S3, Azure Blob, dbfs, or external databases.
* Register the data as a delta table (abstract representation of the underlying data in the Data Storage.
* Infer or apply a schema to the raw data for consistent downstream processing.
"""

import cr.data.data_manager as dm
from cr.config.core import config
from argparse import ArgumentParser
from delta.tables import DeltaTable

if not config.general.RUN_ON_DATABRICKS_WS:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
else:
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()


# Get user inputs for target deployment environment
parser = ArgumentParser()

parser.add_argument(
    "--env",
    type=str,
    required=False,
    default="dev",
    choices=["dev", "test", "staging", "prod"],
    help="The deploy environment name (e.g., 'dev', 'test', 'staging', 'prod').",
)
args = parser.parse_args()
env = args.env
schema_table_name_dict = dm.get_schema_table_names(env)

# Get incremental train data
bronze_inc = spark.read.csv(
    f"{dm.TRAINING_DATA_DIR}/{config.processing.TRAIN_DATA_NAME}.csv",
    header=True,
    inferSchema=True,
)

# Get time period for incremental data
incremental_data_time_period = (
    bronze_inc.toPandas().loc[:, config.processing.TIME_PERIOD_COL].unique()[0]
)


assert (
    config.general.TRAIN_DATA_TIME_PERIOD == incremental_data_time_period
), "Mismatch between time period in incremental train data and train data time period specified by user"

# Create bronze table
spark.sql(
    f"CREATE TABLE IF NOT EXISTS {schema_table_name_dict['bronze_schema']}.{schema_table_name_dict['bronze_full_table']}"
)

# Enable autoMerged: enable automatic schema evolution in case I am merging
# predictions_df with new features to the existing inference table
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

# Read existing bronze table as delta table object
bronze_delta = DeltaTable.forName(
    spark,
    f"{schema_table_name_dict['bronze_schema']}.{schema_table_name_dict['bronze_full_table']}",
)


if len(bronze_delta.toDF().schema.fields) == 0:
    # Overwrite bronze table with incremental data if bronze table is empty
    bronze_inc.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(
        f"{schema_table_name_dict['bronze_schema']}.{schema_table_name_dict['bronze_full_table']}"
    )
else:
    # Upsert predictions_df to inference_delta
    (
        bronze_delta.alias("tgt")
        .merge(
            bronze_inc.alias("upd"),
            # merge conditions
            "tgt.Id = upd.Id AND tgt.idhogar = upd.idhogar AND tgt.time_period = upd.time_period",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
