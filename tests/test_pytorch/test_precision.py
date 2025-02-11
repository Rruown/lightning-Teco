# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import importlib
import json
import os
import re
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from lightning_utilities import module_available
from typing_extensions import get_args

if module_available("lightning"):
    from lightning.pytorch import Callback, LightningModule, Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
    from lightning.pytorch.plugins import MixedPrecision
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel
    from pytorch_lightning.plugins import MixedPrecision


from lightning_teco.pytorch.accelerator import SDAAAccelerator
from lightning_teco.pytorch.plugins import SDAAPrecisionPlugin
from lightning_teco.pytorch.plugins.precision import _PRECISION_INPUT
from lightning_teco.pytorch.strategies import SDAADDPStrategy, SingleSDAAStrategy

supported_precision = get_args(_PRECISION_INPUT)


def run_training(tmpdir, model, plugin, callback=None):
    """Runs a model and returns loss."""
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=SDAAAccelerator(),
        devices=1,
        strategy=SingleSDAAStrategy(),
        plugins=plugin,
        callbacks=callback,
    )
    trainer.fit(model)
    return trainer.callback_metrics["val_loss"], trainer.callback_metrics["train_loss"]


class BaseBM(BoringModel):
    """Model to test with precision Plugin."""

    def forward(self, x):
        """Forward."""
        # Input is in fp32
        identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)

        # torch.mm is computed in bf16.
        x = torch.mm(x, identity)

        return self.layer(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = super().training_step(batch, batch_idx)
        self.log("train_loss", loss.get("loss"), prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = super().validation_step(batch, batch_idx)
        self.log("val_loss", loss.get("x"), prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss = super().test_step(batch, batch_idx)
        self.log("test_loss", loss.get("y"), prog_bar=True, sync_dist=True)
        return loss


class BMAutocastCM(BaseBM):
    """Model for torch.autocast context manager."""

    def forward(self, x):
        """Forward."""
        with torch.autocast(device_type="sdaa", dtype=torch.float16):
            return super().forward(x)


class BMAutocastDecorator(BaseBM):
    """Model for torch.autocast decorator."""

    @torch.autocast(device_type="sdaa", dtype=torch.float16)
    def forward(self, x):
        """Forward."""
        return super().forward(x)


class BMPluginActive(BaseBM):
    """Model to check active autocast CM when using a precision plugin."""

    def forward(self, x):
        """Forward."""
        return super().forward(x)


@pytest.mark.parametrize("precision_plugin", [False, True])
def test_autocast_enable_disable(tmpdir, precision_plugin):
    """Tests autocast granular control with SDAAPrecisionPlugin."""

    class BMAutocastGranularControl(BaseBM):
        """Tests autocast control with enabled arg."""

        def forward(self, x):
            """Forward."""
            with torch.autocast(device_type="sdaa", dtype=torch.float16, enabled=True):
                # Downcasting is lazy.
                # Operands will be downcasted if operator supports float16
                assert x.dtype == torch.float32
                identity = torch.eye(
                    x.shape[1], device=x.device, dtype=x.dtype)
                x = torch.mm(x, identity)
                assert x.dtype == torch.float16

                # In the disabled subregion, inputs from the surrounding region
                # should be cast to required dtype before use
                x = x.to(torch.float32)
                identity = identity.to(torch.float32)
                with torch.autocast(device_type="sdaa", dtype=torch.float16, enabled=False):
                    x = torch.mm(x, identity)
                    assert x.dtype == torch.float32

                # Re-entering autocast enabled region
                x = torch.mm(x, identity)
                assert x.dtype == torch.float16
            return self.layer(x)

    precision_plugin = SDAAPrecisionPlugin(
        precision="fp16-mixed") if precision_plugin else None
    assert run_training(tmpdir, BMAutocastGranularControl(),
                        precision_plugin) is not None


@pytest.mark.standalone_only()
def test_sdaa_precision_with_ddp_strategy(tmpdir, arg_sdaas):
    """Negative test for  inference not supported with SDAADDPStrategy."""
    model = BoringModel()
    dm = BoringDataModule()
    plugin = SDAAPrecisionPlugin(precision="fp32")

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=SDAADDPStrategy(),
        plugins=plugin,
    )

    trainer.test(model, dm)


@pytest.mark.parametrize(
    ("precision", "expectation"),
    [
        (
            "fp8",
            pytest.raises(
                ValueError, match="not supported"
            ),
        )
    ],
)
def test_sdaa_precision_not_supported(precision, expectation):
    """Test fp8 with unsupported device."""
    with expectation:
        SDAAPrecisionPlugin(precision=precision)


@pytest.mark.parametrize(
    ("plugin", "params"),
    [
        (
            MixedPrecision,
            {"device": "sdaa", "precision": "bf16-mixed"},
        ),
        (
            SDAAPrecisionPlugin,
            {},
        ),
        (
            SDAAPrecisionPlugin,
            {"precision": "bf16-mixed"},
        ),
        (
            SDAAPrecisionPlugin,
            {"precision": "bf16"},
        ),
        (
            SDAAPrecisionPlugin,
            {"precision": "32-true"},
        ),
        (
            SDAAPrecisionPlugin,
            {"precision": "32"},
        ),
        (
            SDAAPrecisionPlugin,
            {"precision": "16-mixed"},
        )
    ],
)
def test_precision_plugin_init(plugin, params):
    """Tests precision plugins are instantiated correctly."""
    _plugin = plugin(**params)

    # Common params
    assert _plugin.device == "sdaa"
    assert _plugin.precision == params.get("precision", "32-true")


def test_precision_plugin_invalid_precision_init():
    """Tests precision plugins are instantiated correctly."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`Trainer(accelerator='sdaa', precision='f16-mixed')` is not supported. "
            f"`precision` must be one of: {supported_precision}."
        ),
    ):
        SDAAPrecisionPlugin(precision="f16-mixed")


@pytest.mark.parametrize(
    ("precision"),
    [
        "32",
        "32-true",
        "bf16",
        "bf16-mixed",
    ],
)
def test_sdaa_precision_supported_precision(precision):
    """Tests supported precisions with SDAA Precision Plugin."""
    with nullcontext():
        SDAAPrecisionPlugin(precision=precision)


@pytest.mark.parametrize(
    ("plugin", "params"),
    [
        (
            MixedPrecision,
            {"device": "sdaa", "precision": "bf16-mixed"},
        ),
        (
            SDAAPrecisionPlugin,
            {"precision": "bf16-mixed"},
        ),
        pytest.param(
            SDAAPrecisionPlugin,
            {"precision": "16-mixed"},
        ),
    ],
)
def test_precision_plugin_fit(tmpdir, plugin, params):
    """Tests precision plugins with trainer.fit."""

    class TestCallback(Callback):
        def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            assert trainer.precision == params.get("precision", "32-true")
            raise SystemExit

    seed_everything(42)
    _model = BoringModel()
    _plugin = plugin(**params)
    if isinstance(_plugin, SDAAPrecisionPlugin) and params.get("precision") == "fp8":
        _plugin.convert_modules(_model, replace_layers=True)

    with pytest.raises(SystemExit):
        run_training(tmpdir, _model, _plugin, TestCallback())


@pytest.mark.parametrize(
    ("model", "plugin", "params"),
    [
        (BMAutocastCM, None, None),
        (BMAutocastDecorator, None, None),
        (BMPluginActive, MixedPrecision, {
         "device": "sdaa", "precision": "16-mixed"}),
        pytest.param(
            BMPluginActive,
            SDAAPrecisionPlugin,
            {"precision": "16-mixed"},
        ),
    ]
)
def test_mixed_precision_autocast_to_precision_active(tmpdir, model, plugin, params):
    """Tests autocast is active with torch.autocast context manager."""
    seed_everything(42)
    _model = model()
    _plugin = plugin(**params) if plugin and params else None
    run_training(tmpdir, _model, _plugin)


def test_mixed_precision_compare_accuracy(tmpdir):
    """Test and compare accuracy for mixed precision training methods."""
    model_plugin_list = [
        (BMAutocastCM, None, None),
        (BMAutocastDecorator, None, None),
        (BaseBM, MixedPrecision, {
         "device": "sdaa", "precision": "bf16-mixed"}),
        (BaseBM, SDAAPrecisionPlugin, {"precision": "bf16-mixed"}),
    ]

    loss_list = []
    for item in model_plugin_list:
        seed_everything(42)
        model, plugin, params = item
        model = model()
        _plugin = plugin(**params) if plugin and params else None
        loss_list.append(torch.tensor(run_training(tmpdir, model, _plugin)))

    assert all(torch.allclose(
        loss_list[0], loss_tensor, rtol=1e-2, atol=1e-2) for loss_tensor in loss_list[1:])


@pytest.mark.standalone_only()
@pytest.mark.parametrize(
    ("int64_support", "expectation"),
    [
        ("True", nullcontext()),
    ],
)
def test_sdaa_precision_long_type(int64_support, expectation):
    """Tests native support for long tensor."""
    with expectation:
        torch.tensor(torch.iinfo(torch.int64).max,
                     dtype=torch.int64, device=torch.device("sdaa"))


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.bool,
    ],
)
def test_sdaa_supported_dtypes_tensor_creation(dtype):
    """Tests tensors with supported dtypes can be created on sdaa."""
    with nullcontext():
        torch.tensor(42, dtype=dtype, device=torch.device("sdaa"))


@pytest.mark.parametrize("intype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float32])
def test_sdaa_dtypes_op_output_dtype(intype):
    """Test dtypes type promotion."""
    t1 = torch.tensor([[1, 2], [2, 1]], dtype=intype,
                      device=torch.device("sdaa"))
    t2 = torch.tensor([[2, 1], [1, 2]], dtype=intype,
                      device=torch.device("sdaa"))

    # Operands are promoted as per torch.promote_types
    t3 = t1.mm(t2)
    t4 = t1.add(t2)
    t5 = t1.div(t2)
    assert t3.dtype == torch.promote_types(t1.dtype, t2.dtype)
    assert t4.dtype == torch.promote_types(t1.dtype, t2.dtype)
    # integer div always promoted to float32.
    assert (
        t5.dtype == torch.promote_types(t1.dtype, t2.dtype)
        if t1.is_floating_point() or t2.is_floating_point()
        else torch.float32
    )

    # torch.autocast only affects torch.float16, torch.bfloat16, torch.float32
    with torch.autocast(device_type="sdaa", dtype=torch.bfloat16):
        # Computes in lower precision if operands in (bf16, fp32) else operand dtype
        t3 = t1.mm(t2)
        # Promoted to highest dtype between operands
        t4 = t1.add(t2)

    assert t3.dtype == intype if intype not in (
        torch.bfloat16, torch.float32) else torch.bfloat16
    assert t4.dtype == intype


@pytest.mark.parametrize("intype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_sdaa_dtypes_compare_cpu_accuracy(intype, tmpdir):
    """Test dtypes type promotion."""

    class TestModel(BaseBM):
        def forward(self, x):
            # Perform some operations in given dtype
            x = x.to(intype)
            identity = torch.eye(x.shape[1], device=x.device, dtype=intype)
            x = torch.addmm(x, x, identity)

            return super().forward(x.to(torch.float32))

    metrics = []
    for accelerator in [SDAAAccelerator(), "cpu"]:
        seed_everything(42)
        trainer = Trainer(
            default_root_dir=tmpdir,
            accelerator=accelerator,
            devices=1,
            strategy=SingleSDAAStrategy() if isinstance(
                accelerator, SDAAAccelerator) else "auto",
            fast_dev_run=1,
        )

        trainer.fit(TestModel())
        metrics.append(trainer.logged_metrics)

    # Compare metrics between cpu and sdaa
    assert torch.isclose(metrics[0].get("train_loss"), metrics[1].get(
        "train_loss"), atol=1e-5, rtol=1e-5)
    assert torch.isclose(metrics[0].get("val_loss"), metrics[1].get(
        "val_loss"), atol=1e-5, rtol=1e-5)


def test_sdaa_precision_plugin_grads_dtype(tmpdir):
    """Tests dtype of gradients on sdaa match with those on cpu with SDAAPrecisionPlugin."""

    class TestModel(BoringModel):
        """Test model."""

        def __init__(self):
            """Init."""
            super().__init__()
            self.linear_hook_handle = self.layer.register_full_backward_hook(
                self.layer_backward_hook)
            self.grad_dict: dict = {}

        def back_hook(self, layer_name, grad_input, grad_output):
            """Back hook."""
            if layer_name not in self.grad_dict:
                self.grad_dict[layer_name] = {}
                self.grad_dict[layer_name]["grad_input"] = []
                self.grad_dict[layer_name]["grad_output"] = []
            self.grad_dict[layer_name]["grad_input"].append(grad_input)
            self.grad_dict[layer_name]["grad_output"].append(grad_output)

        def layer_backward_hook(self, module, grad_input, grad_output):
            """Layer backward hook."""
            assert isinstance(module, torch.nn.Linear)
            self.back_hook("Linear", grad_input, grad_output)

        def forward(self, x):
            """Forward."""
            x.requires_grad_(True)
            return super().forward(x)

    grad_dict = {}
    for accelerator, strategy, precision_plugin in [
        ("cpu", "auto", MixedPrecision(device="cpu", precision="bf16-mixed")),
        (SDAAAccelerator(), SingleSDAAStrategy(),
         SDAAPrecisionPlugin(precision="bf16-mixed")),
    ]:
        seed_everything(42)
        model = TestModel()
        dm = BoringDataModule()
        trainer = Trainer(
            default_root_dir=tmpdir,
            accelerator=accelerator,
            devices=1,
            strategy=strategy,
            plugins=precision_plugin,
            fast_dev_run=1,
        )

        trainer.fit(model, dm)
        accelerator_str = "sdaa" if isinstance(
            accelerator, SDAAAccelerator) else accelerator
        grad_dict[accelerator_str] = model.grad_dict

    for (kcpu, vcpu), (ksdaa, vsdaa) in zip(grad_dict["cpu"]["Linear"].items(), grad_dict["sdaa"]["Linear"].items()):
        # Ensure comparing same grad_type grad_input / grad_output for both devices
        assert kcpu == ksdaa
        for (grad_cpu,), (grad_sdaa,) in zip(vcpu, vsdaa):
            # Check grad dtype
            assert grad_cpu.dtype == grad_sdaa.dtype
