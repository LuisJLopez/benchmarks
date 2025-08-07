import time

import duckdb
import numpy as np
import pandas as pd
import polars as pl

# Generate synthetic data
num_rows = 100_000_000
df = pd.DataFrame(
    {"id": np.arange(num_rows), "value": np.random.randint(0, 1000, size=num_rows)}
)

# Save to Parquet
df.to_parquet("synthetic_data.parquet", index=False)

# Benchmark Pandas
start_pandas = time.time()
df_pandas = pd.read_parquet("synthetic_data.parquet")
filtered_pandas = df_pandas[df_pandas["value"] > 900]
end_pandas = time.time()
print(f"Pandas filtering time: {end_pandas - start_pandas:.2f} seconds")

# Benchmark DuckDB
start_duckdb = time.time()
filtered_duckdb = duckdb.query(
    "SELECT * FROM 'synthetic_data.parquet' WHERE value > 900"
).to_df()
end_duckdb = time.time()
print(f"DuckDB filtering time: {end_duckdb - start_duckdb:.2f} seconds")

# Benchmark Polars
start_polars = time.time()
df_polars = pl.read_parquet("synthetic_data.parquet")
filtered_polars = df_polars.filter(pl.col("value") > 900)
end_polars = time.time()
print(f"Polars filtering time: {end_polars - start_polars:.2f} seconds")

# Outcome
# Pandas: Slower due to eager loading and single-threaded execution.
# DuckDB: Fast due to predicate pushdown and multi-threading.
# Polars: Very fast due to Rust backend and parallel execution.


# Results looks like this:
# Pandas filtering time: 2.33 seconds
# DuckDB filtering time: 0.23 seconds
# Polars filtering time: 0.61 seconds