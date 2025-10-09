from operator import methodcaller

import wandb
import numpy as np
import pandas as pd

api = wandb.Api()

# Replace with your entity/project/run IDs
runs: list[wandb.Run] = [
    api.run("mauroabidal/gpt-training/nd0y9fkq"),
    api.run("mauroabidal/gpt-training/u27cnz4r"),
    api.run("mauroabidal/gpt-training/isodsr6o"),
    api.run("mauroabidal/gpt-training/q85ppnls"),
]
# Fetch histories as DataFrames
get_history = methodcaller("history", samples=1_000_000)
dfs = (
    pd.concat(map(get_history, runs))
    .drop_duplicates("step", keep="last")
    .reset_index(drop=True)
)
dfs["_step"] = np.arange(len(dfs))
dfs["step"] = dfs["_step"]
# Start a new clean run and log the concatenated data
wandb.init(project="gpt-training", name="clean_run")

for _, row in dfs.iterrows():
    wandb.log(row.to_dict())

wandb.finish()
