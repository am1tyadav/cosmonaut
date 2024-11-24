import os

import duckdb
import pandas as pd
from dotenv import load_dotenv

from cosmonaut import Cosmonaut

load_dotenv()


def create_prompt(inputs: pd.Series) -> str:
    text = inputs["sentence"]
    return f"Please classify the following sentence: {text}"


def extract_predicted_label(inputs: pd.Series) -> str:
    try:
        return inputs["predictions"]["predictions"][0]["labels"][0]
    except Exception:
        return "Error"


dirpath = os.path.dirname(__file__)
inputs = pd.read_csv(os.path.join(dirpath, "data.csv"))
classifier = Cosmonaut(os.path.join(dirpath, "config.yml"), create_prompt)
result_column = classifier.config.data.result_column


TOTAL_COUNT = len(inputs)
BATCH_SIZE = 5
offset = 0

conn = duckdb.connect()
conn.execute(
    f"CREATE TABLE preds (id BIGINT, language VARCHAR, "
    f"sentence VARCHAR, {result_column} VARCHAR)"
)


while offset < TOTAL_COUNT:
    df = conn.execute(
        "SELECT * FROM inputs ORDER BY id LIMIT ? OFFSET ?", [BATCH_SIZE, offset]
    ).fetch_df_chunk()
    offset += BATCH_SIZE

    results = classifier.run(df)
    df[result_column] = results[result_column].apply(extract_predicted_label)

    conn.execute("INSERT INTO preds SELECT * FROM df")


query = conn.sql("SELECT * FROM preds")
query.show()
