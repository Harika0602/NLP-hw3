import subprocess

models = ["RankAggregated", "RankNearest", "RankFixed"]
datasets = ["ml-1m","lastfm"]
noise_types = ["random", "truncate", "duplicate"]
noise_ratio = 0.2

# Without noise
print("Running evaluations without noise...")
for model in models:
    for dataset in datasets:
        command = f"python run_test.py -m {model} -d {dataset}"
        subprocess.run(command, shell=True)

# With noise
print("Running evaluations with noise...")
for model in models:
    for dataset in datasets:
        for noise in noise_types:
            command = f"python run_test.py -m {model} -d {dataset} --noise_type {noise} --noise_ratio {noise_ratio}"
            subprocess.run(command, shell=True)
