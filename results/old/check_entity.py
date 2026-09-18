import wandb

api = wandb.Api()
print(f"Default entity: {api.default_entity}")

runs_choemj = api.runs("choemj-kaist/STORM")
print(f"Total runs in choemj-kaist/STORM: {len(runs_choemj)}")

runs_ai2lab = api.runs(f"{api.default_entity}/STORM")
print(f"Total runs in {api.default_entity}/STORM: {len(runs_ai2lab)}")
