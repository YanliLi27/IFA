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
    return metrics  # , results