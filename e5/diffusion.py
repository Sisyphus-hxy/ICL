"""Exact BPP-compatible DDIMScheduler, EMAModel, and cosine LR scheduler.

Matches diffusers==0.35.1 behavior used by official BPP without requiring diffusers
to be installed in the Blackwell SM120 training environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class DDIMSchedulerConfig:
    num_train_timesteps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "epsilon"
    timestep_spacing: str = "leading"


@dataclass
class DDIMStepOutput:
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
) -> torch.Tensor:
    def alpha_bar_fn(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDIMScheduler:
    """Bit-exact replica of diffusers.DDIMScheduler for BPP's diffusion configuration."""

    def __init__(
        self,
        num_train_timesteps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
    ):
        self.config = DDIMSchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            timestep_spacing=timestep_spacing,
        )
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"Unsupported beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0, dtype=torch.float32)
            if set_alpha_to_one
            else self.alphas_cumprod[0].clone()
        )
        self.num_inference_steps: Optional[int] = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) > "
                f"num_train_timesteps ({self.config.num_train_timesteps})"
            )
        self.num_inference_steps = num_inference_steps
        if self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = (
                np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(
                    np.int64
                )
                - 1
            )
        else:
            raise ValueError(f"Unsupported timestep_spacing: {self.config.timestep_spacing}")
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        return (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> DDIMStepOutput:
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before step()")

        t = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        prev_t = t - self.config.num_train_timesteps // self.num_inference_steps

        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        self.final_alpha_cumprod = self.final_alpha_cumprod.to(device=sample.device)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1.0 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - (beta_prod_t**0.5) * model_output
            ) / (alpha_prod_t**0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - (alpha_prod_t**0.5) * pred_original_sample
            ) / (beta_prod_t**0.5)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        variance = self._get_variance(t, prev_t)
        std_dev_t = eta * (variance**0.5)

        pred_sample_direction = (1.0 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        prev_sample = (alpha_prod_t_prev**0.5) * pred_original_sample + pred_sample_direction

        if eta > 0.0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample + std_dev_t * noise

        return DDIMStepOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )


class EMAModel:
    """Exact replica of diffusers.training_utils.EMAModel used by BPP."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        foreach: bool = True,
    ):
        params = list(parameters)
        self.shadow_params: List[torch.Tensor] = [p.clone().detach() for p in params]
        self.temp_stored_params: Optional[List[torch.Tensor]] = None
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value: Optional[float] = None
        self.foreach = foreach

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_ema_warmup:
            cur_decay_value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        else:
            cur_decay_value = (1.0 + step) / (10.0 + step)
        cur_decay_value = min(cur_decay_value, self.decay)
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameters = list(parameters)
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1.0 - decay

        if self.foreach:
            params_grad = [p for p in parameters if p.requires_grad]
            s_params_grad = [
                sp for sp, p in zip(self.shadow_params, parameters) if p.requires_grad
            ]
            if len(params_grad) < len(parameters):
                torch._foreach_copy_(
                    [sp for sp, p in zip(self.shadow_params, parameters) if not p.requires_grad],
                    [p for p in parameters if not p.requires_grad],
                    non_blocking=True,
                )
            if params_grad:
                torch._foreach_sub_(
                    s_params_grad,
                    torch._foreach_sub(s_params_grad, params_grad),
                    alpha=one_minus_decay,
                )
        else:
            for s_param, param in zip(self.shadow_params, parameters):
                if param.requires_grad:
                    s_param.sub_(one_minus_decay * (s_param - param))
                else:
                    s_param.copy_(param)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.to(param.device).data)

    @torch.no_grad()
    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    @torch.no_grad()
    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        if self.temp_stored_params is None:
            raise RuntimeError("EMAModel.restore() called without matching store()")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.to(param.device).data)
        self.temp_stored_params = None

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "EMAModel":
        self.shadow_params = [
            p.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if p.is_floating_point()
            else p.to(device=device, non_blocking=non_blocking)
            for p in self.shadow_params
        ]
        return self

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": [p.detach().cpu() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.min_decay = state_dict["min_decay"]
        self.optimization_step = state_dict["optimization_step"]
        self.update_after_step = state_dict["update_after_step"]
        self.use_ema_warmup = state_dict["use_ema_warmup"]
        self.inv_gamma = state_dict["inv_gamma"]
        self.power = state_dict["power"]
        device = self.shadow_params[0].device if self.shadow_params else torch.device("cpu")
        self.shadow_params = [p.to(device) for p in state_dict["shadow_params"]]


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Exact replica of diffusers.optimization.get_cosine_schedule_with_warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
