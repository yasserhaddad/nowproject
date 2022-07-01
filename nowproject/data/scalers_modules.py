import numpy as np

from nowproject.scalers import (
    normalize_transform,
    normalize_inverse_transform,
    log_normalize_transform,
    log_normalize_inverse_transform,
    bin_transform,
    bin_inverse_transform,
    Scaler
)


def log_normalize_scaler():
    transform_kwargs = dict(feature_min=np.log10(0.025), 
                            feature_max=np.log10(100), 
                            threshold=np.log10(0.1))
    scaler = Scaler(log_normalize_transform, 
                    log_normalize_inverse_transform, 
                    transform_kwargs=transform_kwargs,
                    inverse_transform_kwargs=transform_kwargs)

    return scaler


def normalize_scaler():
    transform_kwargs = dict(feature_min=0.025, 
                            feature_max=100, 
                            threshold=0.1)
    scaler = Scaler(normalize_transform, 
                    normalize_inverse_transform, 
                    transform_kwargs=transform_kwargs,
                    inverse_transform_kwargs=transform_kwargs)

    return scaler


def bin_scaler():
    bins = [-0.001, 0.001, 0.2, 0.6, 1.4, 3.6, 6.4, 13.6, 26.4, 53.6, 86.4, 113.6]
    centres = [round((bins[i] + bins[i+1])/2, 1) for i in range(0, len(bins)-1)]
    def inverse_bins_fn(x):
        if x < 1:
            return centres[0]
        elif x >= 1 and x < len(centres):
            return centres[x-1]
        else:
            return centres[-1]
    inverse_bins = np.vectorize(lambda x: inverse_bins_fn(x), otypes=[float])
    transform_kwargs = dict(bins=bins)
    inverse_transform_kwargs = dict(inverse_bins=inverse_bins)
    scaler = Scaler(bin_transform, 
                    bin_inverse_transform, 
                    transform_kwargs=transform_kwargs,
                    inverse_transform_kwargs=inverse_transform_kwargs)

    return scaler
# scaler = LogScaler(epsilon=1e-5)
# model_settings["last_layer_activation"] = True

# scaler = dBScaler(threshold=0.1, inverse_threshold=-10.0, zero_value=-15.0)
# model_settings["last_layer_activation"] = False
