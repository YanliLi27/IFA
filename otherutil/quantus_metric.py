import quantus
import numpy as np
from scipy.stats import pearsonr

def safe_pearson(*, a, b):  # 接受关键字参数 a, b
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return pearsonr(a, b)[0]

def obtain_metrics_results():
    metrics = {
        
        "Randomisation": quantus.RandomLogit(
            num_classes=2,
            similarity_func=quantus.similarity_func.ssim,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Faithfulness": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=32,
            perturb_baseline="black",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            similarity_func=safe_pearson,# quantus.similarity_func.correlation_pearson,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
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
        # Requires segmentation
        # "Localisation": quantus.RelevanceRankAccuracy(
        #     abs=False,
        #     normalise=False,
        #     aggregate_func=np.mean,
        #     return_aggregate=True,
        #     disable_warnings=True,
        # ),
        "Complexity": quantus.Sparseness(
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        
    }
    return metrics  # , results