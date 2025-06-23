import quantus
import numpy as np


def obtain_metrics_results():
    metrics = {
        "Robustness": quantus.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Faithfulness": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=224,
            perturb_baseline="black",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Localisation": quantus.RelevanceRankAccuracy(
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Complexity": quantus.Sparseness(
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Randomisation": quantus.RandomLogit(
            num_classes=1000,
            similarity_func=quantus.similarity_func.ssim,
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
    }


    results = {
    "Saliency": {
        "Robustness": [0.023706467548275694],
        "Faithfulness": [0.06749252841918861],
        "Localisation": [0.5122173871263156],
        "Complexity": [0.5503504513474646],
        "Randomisation": [0.8064057449830752],
    },
    "GradientShap": {
        "Robustness": [0.034456219962414575],
        "Faithfulness": [0.04583139237937677],
        "Localisation": [0.5046252238901434],
        "Complexity": [0.6088842825604118],
        "Randomisation": [0.7366918283019923],
    },
    "IntegratedGradients": {
        "Robustness": [0.02690529538428082],
        "Faithfulness": [-0.08233498797221532],
        "Localisation": [0.5071891576864163],
        "Complexity": [0.6107274870736773],
        "Randomisation": [0.7915174357470123],
    },
    "FusionGrad": {
        "Robustness": [5728.689449534697],
        "Faithfulness": [0.0004617348733465858],
        "Localisation": [0.493903066639846],
        "Complexity": [0.5320443076081544],
        "Randomisation": [0.006262477424033301],
    },
    }
    return metrics, results