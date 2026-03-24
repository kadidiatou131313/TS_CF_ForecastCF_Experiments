from torch.utils.data import DataLoader
import numpy as np
import darts

import darts.models.forecasting.tft_model
import darts.models.forecasting.rnn_model
import darts.models.forecasting.nlinear
import darts.models.forecasting.dlinear
import darts.models.forecasting.nbeats
import darts.models.forecasting.linear_regression_model

import torch.nn
import pytorch_lightning.callbacks.early_stopping


class BaseForecaster:
    def __init__(self, config, target):
        raise NotImplementedError()

    def fit(self, data_train, data_train_time, data_val, data_val_time):
        raise NotImplementedError()

    def predict(self, data_test, data_test_time):
        raise NotImplementedError()

    def infer_data_shape_parameters(partial_config, data, data_time):
        raise NotImplementedError

    def generate_optuna_parameters(trial):
        raise NotImplementedError

    def generate_optuna_parameters_grid():
        raise NotImplementedError


class BaseDartsWrapper:
    constructor = None

    def __init__(self, config, target, **kwargs):
        config = self._configuration_completion(config)
        self.pred_len = config.get("forecasting_horizon", 1)
        self.config = config.copy()
        self.target = target

        model_config = self.config.copy()
        forbidden_keys = ["forecasting_horizon", "target"]
        for k in forbidden_keys:
            model_config.pop(k, None)

        self.model = self.constructor(**model_config)

    def _prepare_data(self, data, data_time):
        d = darts.TimeSeries.from_dataframe(data, value_cols=[x for x in data.columns])

        if len(data_time.columns) == 0:
            dt = darts.TimeSeries.from_times_and_values(
                times=data_time.index,
                values=np.empty((len(data_time), 0))
            )
        else:
            dt = darts.TimeSeries.from_dataframe(
                data_time,
                value_cols=[x for x in data_time.columns]
            )

        target_series = d[self.target]
        other_cols = [x for x in data.columns if x != self.target]

        if len(other_cols) == 0:
            covariates = None
        else:
            covariates = d[other_cols]

        return target_series, covariates, dt

    def _full_fit_params(
        self,
        data_train_target,
        data_train_past,
        data_train_future,
        data_val_target,
        data_val_past,
        data_val_future,
    ):
        config = dict()
        config["series"] = data_train_target

        if self.model.supports_past_covariates and data_train_past is not None:
            config["past_covariates"] = data_train_past

        if self.model.supports_future_covariates:
            config["future_covariates"] = data_train_future

        if isinstance(
            self.model,
            darts.models.forecasting.torch_forecasting_model.TorchForecastingModel
        ):
            config["val_series"] = data_val_target

            if self.model.supports_past_covariates and data_val_past is not None:
                config["val_past_covariates"] = data_val_past

            if self.model.supports_future_covariates:
                config["val_future_covariates"] = data_val_future

        return config

    def fit(self, data_train, data_train_time, data_val, data_val_time):
        data_train_target, data_train_past, data_train_future = self._prepare_data(
            data_train, data_train_time
        )
        data_val_target, data_val_past, data_val_future = self._prepare_data(
            data_val, data_val_time
        )

        fit_config = self._full_fit_params(
            data_train_target,
            data_train_past,
            data_train_future,
            data_val_target,
            data_val_past,
            data_val_future,
        )
        self.model.fit(**fit_config)

    def _full_predict_params(self, data_test_target, data_test_past, data_test_future):
        config = dict()
        config["series"] = data_test_target

        if self.model.supports_past_covariates and data_test_past is not None:
            config["past_covariates"] = data_test_past

        if self.model.supports_future_covariates:
            config["future_covariates"] = data_test_future

        config["forecast_horizon"] = self.pred_len
        config["stride"] = 1
        config["retrain"] = False
        config["overlap_end"] = False
        config["last_points_only"] = False
        return config

    def predict(self, data_test, data_test_time):
        data_test_target, data_test_past, data_test_future = self._prepare_data(
            data_test, data_test_time
        )
        test_config = self._full_predict_params(
            data_test_target,
            data_test_past,
            data_test_future
        )

        pred = self.model.historical_forecasts(**test_config)

        y_true = np.array([
            data_test_target[pred[i].time_index].values()
            for i in range(len(pred))
        ])
        y_pred = np.array([pred[i].values() for i in range(len(pred))])

        return y_pred, y_true

    def _configuration_completion(self, partial_config):
        return partial_config


class TorchDartsWrapper(BaseDartsWrapper):
    def _configuration_completion(self, partial_config):
        d = dict()
        d["batch_size"] = partial_config.get("batch_size", 32)
        d["n_epochs"] = partial_config.get("n_epochs", 30)
        d["random_state"] = partial_config.get("random_state", 0)
        d["nr_epochs_val_period"] = None
        d["pl_trainer_kwargs"] = dict()

        es_instance = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
            monitor=partial_config.get("pl_trainer_kwargs.callbacks.monitor", "val_loss"),
            patience=partial_config.get("pl_trainer_kwargs.callbacks.patience", 3),
            min_delta=partial_config.get("pl_trainer_kwargs.callbacks.min_delta", 0.0),
            mode=partial_config.get("pl_trainer_kwargs.callbacks.mode", "min"),
        )

        d["pl_trainer_kwargs"]["callbacks"] = (
            partial_config.get("pl_trainer_kwargs.callbacks", []) + [es_instance]
        )
        return d


class TFTDartsWrapper(TorchDartsWrapper):
    constructor = darts.models.forecasting.tft_model.TFTModel

    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon", 1)
        d["input_chunk_length"] = partial_config.get("input_chunk_length", 32)
        d["output_chunk_length"] = partial_config.get(
            "output_chunk_length",
            d["forecasting_horizon"]
        )
        d["hidden_size"] = partial_config.get("hidden_size", 16)
        d["lstm_layers"] = partial_config.get("lstm_layers", 1)
        d["num_attention_heads"] = partial_config.get("num_attention_heads", 4)
        d["dropout"] = partial_config.get("dropout", 0.1)
        d["add_encoders"] = {'position': {'future': ['relative']}}
        return d


class RNNDartsWrapper(TorchDartsWrapper):
    constructor = darts.models.forecasting.rnn_model.RNNModel

    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)

        d["input_chunk_length"] = partial_config.get("input_chunk_length", 32)
        d["training_length"] = partial_config.get(
            "training_length",
            d["input_chunk_length"]
        )
        d["output_chunk_length"] = partial_config.get("output_chunk_length", 1)

        d["model"] = partial_config.get("model", "GRU")
        d["hidden_dim"] = partial_config.get("hidden_dim", 25)
        d["n_rnn_layers"] = partial_config.get("n_rnn_layers", 1)
        d["dropout"] = partial_config.get("dropout", 0.1)

        return d


class NLinearDartsWrapper(TorchDartsWrapper):
    constructor = darts.models.forecasting.nlinear.NLinearModel

    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon", 1)
        d["input_chunk_length"] = partial_config.get("input_chunk_length", 32)
        d["output_chunk_length"] = partial_config.get(
            "output_chunk_length",
            d["forecasting_horizon"]
        )
        return d


class DLinearDartsWrapper(NLinearDartsWrapper):
    constructor = darts.models.forecasting.dlinear.DLinearModel

    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["kernel_size"] = partial_config.get("kernel_size", 25)
        return d


class NBEATSDartsWrapper(TorchDartsWrapper):
    constructor = darts.models.forecasting.nbeats.NBEATSModel

    def _configuration_completion(self, partial_config):
        d = super()._configuration_completion(partial_config)
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon", 1)
        d["input_chunk_length"] = partial_config.get("input_chunk_length", 32)
        d["output_chunk_length"] = partial_config.get(
            "output_chunk_length",
            d["forecasting_horizon"]
        )

        d["generic_architecture"] = partial_config.get("generic_architecture", True)
        d["num_stacks"] = partial_config.get("num_stacks", 2)
        d["num_blocks"] = partial_config.get("num_blocks", 2)
        d["num_layers"] = partial_config.get("num_layers", 2)
        d["layer_widths"] = partial_config.get("layer_widths", 128)
        d["dropout"] = partial_config.get("dropout", 0.1)

        return d


class LinearDartsWrapper(BaseDartsWrapper):
    constructor = darts.models.forecasting.linear_regression_model.LinearRegressionModel

    def _configuration_completion(self, partial_config):
        d = dict()
        d["forecasting_horizon"] = partial_config.get("forecasting_horizon", 1)
        d["lags"] = partial_config.get("lags", 32)
        return d


def build_base_configuration(forecaster_name, pred_len, seq_len):
    if forecaster_name == "RNNDartsWrapper":
        es_constructor = RNNDartsWrapper
        es_config = {
            "forecasting_horizon": pred_len,
            "input_chunk_length": seq_len
        }

    elif forecaster_name == "TFTDartsWrapper":
        es_constructor = TFTDartsWrapper
        es_config = {
            "forecasting_horizon": pred_len,
            "input_chunk_length": seq_len
        }

    elif forecaster_name == "NLinearDartsWrapper":
        es_constructor = NLinearDartsWrapper
        es_config = {
            "forecasting_horizon": pred_len,
            "input_chunk_length": seq_len
        }

    elif forecaster_name == "DLinearDartsWrapper":
        es_constructor = DLinearDartsWrapper
        es_config = {
            "forecasting_horizon": pred_len,
            "input_chunk_length": seq_len
        }

    elif forecaster_name == "NBEATSDartsWrapper":
        es_constructor = NBEATSDartsWrapper
        es_config = {
            "forecasting_horizon": pred_len,
            "input_chunk_length": seq_len
        }

    elif forecaster_name == "LinearDartsWrapper":
        es_constructor = LinearDartsWrapper
        es_config = {
            "forecasting_horizon": pred_len,
            "lags": seq_len
        }

    else:
        raise ValueError(f"Unknown forecaster_name: {forecaster_name}")

    return es_constructor, es_config