import torch
import numpy as np

class ForecastCF:
    def __init__(
        self,
        *,
        tolerance=1e-6,
        max_iter=100,
        optimizer_cls=torch.optim.Adam,
        lr=1e-3,
        pred_margin_weight=1.0,
        step_weights="local", 
        device=None,
        dtype=torch.float64
    ):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.pred_margin_weight = pred_margin_weight
        self.weighted_steps_weight = 1.0 - pred_margin_weight
        self.step_weights_init = step_weights
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        self.MISSING_MAX_BOUND = float("inf")
        self.MISSING_MIN_BOUND = -float("inf")

    def fit(self, model):
        self.model = model
        self.model.eval()
        return self

    def _get_preds(self, x):
        out = self.model((x, None, None))
        preds = out[0] if isinstance(out, tuple) else out


        while preds.dim() > 2:
            preds = preds.squeeze(-1)

        return preds


    def margin_mse(self, prediction, max_bound, min_bound):
   

  
        if not torch.is_tensor(max_bound):
            max_bound = torch.as_tensor(
                max_bound, device=prediction.device, dtype=prediction.dtype)
        if not torch.is_tensor(min_bound):
            min_bound = torch.as_tensor(
            min_bound, device=prediction.device, dtype=prediction.dtype)


        max_bound = torch.broadcast_to(max_bound, prediction.shape)
        min_bound = torch.broadcast_to(min_bound, prediction.shape)


        upper_mask = prediction > (max_bound + self.tolerance)
        lower_mask = prediction < (min_bound - self.tolerance)

        loss = torch.zeros((), device=prediction.device, dtype=prediction.dtype)

        if upper_mask.any():
            loss += torch.sum((prediction[upper_mask] - max_bound[upper_mask]) ** 2)

        if lower_mask.any():
            loss += torch.sum((min_bound[lower_mask] - prediction[lower_mask]) ** 2)

        return loss





    def compute_loss(self, original_sample, z_search, step_weights, max_bound, min_bound):
        preds = self._get_preds(z_search)
        
  
        forecast_margin_loss = self.margin_mse(preds, max_bound, min_bound)
        

        weighted_steps_loss = torch.mean(torch.abs(original_sample - z_search) * step_weights)

        total_loss = (self.pred_margin_weight * forecast_margin_loss + 
                      self.weighted_steps_weight * weighted_steps_loss)
        
        return total_loss, forecast_margin_loss, weighted_steps_loss, preds
    

    def transform(self, x, max_bound_lst=None, min_bound_lst=None):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n_samples = x.shape[0]
        result_samples = torch.zeros_like(x)
        
        for i in range(n_samples):
            max_b = max_bound_lst[i] if max_bound_lst is not None else self.MISSING_MAX_BOUND
            min_b = min_bound_lst[i] if min_bound_lst is not None else self.MISSING_MIN_BOUND


            max_b = torch.as_tensor(max_b, device=self.device, dtype=self.dtype)
            min_b = torch.as_tensor(min_b, device=self.device, dtype=self.dtype)

            
 
            if isinstance(self.step_weights_init, str) and self.step_weights_init == "local":
                step_weights = torch.ones((1, x.shape[1], x.shape[2]), device=self.device, dtype=self.dtype)
            else:
                step_weights = torch.as_tensor(self.step_weights_init, device=self.device, dtype=self.dtype)
            
            cf_sample, _ = self._transform_sample(x[i:i+1], step_weights, max_b, min_b)
            result_samples[i] = cf_sample

        return result_samples.detach().cpu().numpy()

    def _transform_sample(self, x, step_weights, max_bound, min_bound):
        z = x.clone().detach().requires_grad_(True)
        optimizer = self.optimizer_cls([z], lr=self.lr)
        
        it = 0
        while it < self.max_iter:
            optimizer.zero_grad()
            
            loss, f_margin, w_steps, preds = self.compute_loss(x, z, step_weights, max_bound, min_bound)
            

            if torch.all(preds <= max_bound + self.tolerance) and \
               torch.all(preds >= min_bound - self.tolerance):
                break

            loss.backward()
            optimizer.step()
            it += 1
            
        return z.detach(), loss.item()