import time

import duckdb
import numpy as np
import pandas as pd
import polars as pl

# Generate synthetic data
num_rows = 100_000_000
categories = ["A", "B", "C", "D"]

# Filtering dataset
df_filter = pd.DataFrame(
    {"id": np.arange(num_rows), "value": np.random.randint(0, 1000, size=num_rows)}
)
df_filter.to_parquet("synthetic_data.parquet", index=False)

# GroupBy dataset
df_group = pd.DataFrame(
    {
        "id": np.arange(num_rows),
        "category": np.random.choice(categories, size=num_rows),
        "value": np.random.rand(num_rows) * 100,
    }
)
df_group.to_parquet("group_data.parquet", index=False)

# Join datasets
df_a = pd.DataFrame(
    {"id": np.arange(num_rows), "value_a": np.random.rand(num_rows) * 100}
)
df_b = pd.DataFrame(
    {"id": np.arange(num_rows), "value_b": np.random.rand(num_rows) * 100}
)
df_a.to_parquet("data_a.parquet", index=False)
df_b.to_parquet("data_b.parquet", index=False)

# Filtering benchmark
start = time.time()
pd.read_parquet("synthetic_data.parquet")[lambda df: df["value"] > 900]
print(f"Pandas filtering time: {time.time() - start:.2f} sec")

start = time.time()
duckdb.query("SELECT * FROM 'synthetic_data.parquet' WHERE value > 900").to_df()
print(f"DuckDB filtering time: {time.time() - start:.2f} sec")

start = time.time()
pl.read_parquet("synthetic_data.parquet").filter(pl.col("value") > 900)
print(f"Polars filtering time: {time.time() - start:.2f} sec")

# GroupBy benchmark
start = time.time()
pd.read_parquet("group_data.parquet").groupby("category")["value"].mean()
print(f"Pandas groupby time: {time.time() - start:.2f} sec")

start = time.time()
duckdb.query(
    "SELECT category, AVG(value) FROM 'group_data.parquet' GROUP BY category"
).to_df()
print(f"DuckDB groupby time: {time.time() - start:.2f} sec")

start = time.time()
pl.read_parquet("group_data.parquet").group_by("category").agg(pl.col("value").mean())
print(f"Polars groupby time: {time.time() - start:.2f} sec")

# Join benchmark
start = time.time()
pd.merge(pd.read_parquet("data_a.parquet"), pd.read_parquet("data_b.parquet"), on="id")
print(f"Pandas join time: {time.time() - start:.2f} sec")

start = time.time()
duckdb.query(
    """
    SELECT a.*, b.value_b
    FROM 'data_a.parquet' a
    JOIN 'data_b.parquet' b ON a.id = b.id
"""
).to_df()
print(f"DuckDB join time: {time.time() - start:.2f} sec")

start = time.time()
pl.read_parquet("data_a.parquet").join(
    pl.read_parquet("data_b.parquet"), on="id", how="inner"
)
print(f"Polars join time: {time.time() - start:.2f} sec")

# Outcome
# Pandas: Slower due to eager loading and single-threaded execution.
# DuckDB: Fast due to predicate pushdown and multi-threading.
# Polars: Very fast due to Rust backend and parallel execution.

# Results looks like this:
# Pandas filtering time: 2.55 sec
# DuckDB filtering time: 0.26 sec
# Polars filtering time: 0.24 sec
# Pandas groupby time: 5.78 sec
# DuckDB groupby time: 0.14 sec
# Polars groupby time: 1.85 sec
# Pandas join time: 19.91 sec
# DuckDB join time: 3.33 sec
# Polars join time: 5.20 sec

# Pandas will load the entire file into memory before filtering.
# DuckDB will apply predicate pushdown, reading only relevant parts of the file, making it significantly faster and more memory-efficient.
