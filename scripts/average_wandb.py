import wandb
import numpy as np

# Initialize the W&B API
api = wandb.Api()

run_path = input('Enter run name:')

# Reference the specific run using its path
run = api.run(run_path)

metrics = [
    "losses/self_attention_loss",
    "losses/hidden_state_loss",
    "losses/logit_loss",
    "losses/cross_attention_loss",
]

averages = dict()
for metric in metrics:
    # Retrieve the history of the desired metric
    metric_history = run.scan_history(keys=[metric])

    # Extract the metric values
    metric_values = [row[metric] for row in metric_history]

    # Calculate the average of the metric
    average_metric = np.mean(metric_values)
    averages[metric] = average_metric

    print(metric, average_metric)


# Compute the inverse sum
inverse_sum = sum(1 / np.array(list(averages.values())))

# Compute factors
factors = {key: (1 / value) / inverse_sum for key, value in averages.items()}

print(factors)




