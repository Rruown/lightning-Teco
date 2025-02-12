# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Mapping, Optional, Union

from lightning_utilities import module_available
from torch import Tensor
from torch.optim import LBFGS, Optimizer


from lightning_teco.lightning_api import (Steppable,
                                          LightningModule,
                                          LightningModule,
                                          GradClipAlgorithmType,
                                          MisconfigurationException,
                                          is_overridden,
                                          WarningCache)

from lightning_teco.pytorch.plugins.precision import _PRECISION_INPUT, SDAAPrecisionPlugin

warning_cache = WarningCache()


class SDAADeepSpeedPrecisionPlugin(SDAAPrecisionPlugin):
    """Plugin that enables mixed precision support on SDAAs.

    Args:
        precision (_PRECISION_INPUT, optional): Precision input. Defaults to "32-true".

    Raises:
        OSError: Unsupported Synapse version.
        ValueError: Invalid precision value.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT = "32-true",
        device: str = "sdaa",
    ) -> None:
        if not module_available('deepspeed'):
            raise MisconfigurationException(
                "To use the `SDAADeepSpeedPrecisionPlugin`, you must have sdaa DeepSpeed installed."
            )
        super().__init__(device=device, precision=precision)

    def backward(
        self,
        tensor: Tensor,
        model: "LightningModule",
        optimizer: Optional[Steppable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs back-propagation using DeepSpeed's engine.

        Args:
            tensor: the loss tensor
            model: the model to be optimized
            optimizer: ignored for DeepSpeed
            *args: additional positional arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call
            **kwargs: additional keyword arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call

        """
        import deepspeed
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since DeepSpeed handles"
                " the backward logic internally."
            )
        deepspeed_engine: deepspeed.DeepSpeedEngine = model.trainer.model
        deepspeed_engine.backward(tensor, *args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Steppable,
        model: "LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        import deepspeed
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                "DeepSpeed and the LBFGS optimizer are not compatible.")
        closure_result = closure()
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if model.automatic_optimization and skipped_backward:
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not supported by `DeepSpeed`"
            )
        # DeepSpeed handles the optimizer step internally
        deepspeed_engine: deepspeed.DeepSpeedEngine = model.trainer.model
        return deepspeed_engine.step(**kwargs)

