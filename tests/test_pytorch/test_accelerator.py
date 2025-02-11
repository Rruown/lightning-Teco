import os
from contextlib import nullcontext
from typing import Any, Optional, Union
from unittest import mock

import pytest
import torch
from lightning_teco.utils.resources import num_sdaa_devices
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.utilities.types import ReduceOp
    from lightning.pytorch import Callback, Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.types import ReduceOp
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_teco.pytorch.accelerator import SDAAAccelerator
from lightning_teco.pytorch.plugins import SDAAPrecisionPlugin
from lightning_teco.pytorch.strategies import SDAADDPStrategy, SingleSDAAStrategy

from tests.helpers import ClassifDataModule, ClassificationModel


def test_availability():
    assert SDAAAccelerator.is_available()


@pytest.mark.standalone()
def test_all_stages(tmpdir, arg_sdaas):
    """Tests all the model stages using BoringModel on SDAA."""
    model = BoringModel()

    _strategy = SingleSDAAStrategy()
    if arg_sdaas > 1:
        parallel_sdaas = [torch.device("sdaa")] * arg_sdaas
        _strategy = SDAADDPStrategy(parallel_devices=parallel_sdaas)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=SDAAAccelerator(),
        strategy=_strategy,
        devices=arg_sdaas,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


@pytest.mark.parametrize(
    ("device", "strategy"),
    [
        (
            "auto",
            "auto",
        ),
        (1, "auto"),
        pytest.param(
            2,
            "auto",
            marks=pytest.mark.skipif(
                SDAAAccelerator.auto_device_count() < 2, reason="Test requires multiple sdaa devices."
            ),
        ),
        ("auto", SingleSDAAStrategy),  # Test assumes only 1 card is available,
    ],
)
def test_sdaa_accelerator_trainer_init_devices_and_strategy(device, strategy, arg_sdaas):
    """Tests correct number of devices and strategy is picked when using "auto"."""
    # Mock the node to only have "sdaas" number of cards
    num_sdaas = 1 if arg_sdaas == 1 or (
        device == "auto" and strategy == SingleSDAAStrategy) else 2
    with mock.patch.object(
        SDAAAccelerator, "get_parallel_devices", return_value=[torch.device("sdaa")] * num_sdaas
    ), mock.patch.object(SDAAAccelerator, "auto_device_count", return_value=num_sdaas):
        trainer = Trainer(
            accelerator=SDAAAccelerator(), devices=device, strategy=strategy() if strategy != "auto" else strategy
        )
        assert trainer.num_devices == num_sdaas
        trainer_strategy = (
            strategy if strategy != "auto" else SingleSDAAStrategy if num_sdaas == 1 else SDAAParallelStrategy
        )
        assert isinstance(trainer.strategy, trainer_strategy)


def test_device_name():
    assert "GAUDI" in SDAAAccelerator.get_device_name()


def test_accelerator_selected():
    trainer = Trainer(accelerator=SDAAAccelerator(),
                      strategy=SingleSDAAStrategy())
    assert isinstance(trainer.accelerator, SDAAAccelerator)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_optimization(tmpdir):
    seed_everything(42)

    dm = ClassifDataModule(length=1024)
    model = ClassificationModel()

    _strategy = SingleSDAAStrategy()

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, max_steps=10, accelerator=SDAAAccelerator(), devices=1, strategy=_strategy
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None


def test_stages_correct(tmpdir):
    """Ensure all stages correctly are traced correctly by asserting the output for each stage."""

    class StageModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            loss = loss.get("loss")
            # tracing requires a loss value that depends on the model.
            # force it to be a value but ensure we use the loss.
            loss = (loss - loss) + torch.tensor(1)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            x = loss.get("x")
            x = (x - x) + torch.tensor(2)
            return {"x": x}

        def test_step(self, batch, batch_idx):
            loss = super().test_step(batch, batch_idx)
            y = loss.get("y")
            y = (y - y) + torch.tensor(3)
            return {"y": y}

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            output = super().predict_step(batch, batch_idx)
            return (output - output) + torch.tensor(4)

    class TestCallback(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["loss"].item() == 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["x"].item() == 2

        def on_test_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["y"].item() == 3

        def on_predict_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert torch.all(outputs == 4).item()

    model = StageModel()
    _strategy = SingleSDAAStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=SDAAAccelerator(),
        devices=1,
        strategy=_strategy,
        callbacks=TestCallback(),
    )
    trainer.fit(model)
    trainer.test(model)
    trainer.validate(model)
    trainer.predict(model)


@pytest.mark.parametrize("accelerator", ["auto", "sdaa", SDAAAccelerator()])
def test_sdaa_accelerator_trainer_init_accelerator(accelerator):
    trainer = Trainer(accelerator=accelerator, strategy=SingleSDAAStrategy())
    assert isinstance(trainer.accelerator, SDAAAccelerator)


@pytest.mark.parametrize("sdaas", [1])
def test_inference_only(tmpdir, sdaas):
    model = BoringModel()

    _strategy = SingleSDAAStrategy()
    if sdaas > 1:
        parallel_sdaas = [torch.device("sdaa")] * sdaas
        _strategy = SDAADDPStrategy(parallel_devices=parallel_sdaas)
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, accelerator=SDAAAccelerator(), devices=sdaas, strategy=_strategy
    )
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


def test_sdaa_auto_device_count():
    assert SDAAAccelerator.auto_device_count() == SDAAAccelerator.auto_device_count()


def test_sdaa_unsupported_device_type():
    with pytest.raises(MisconfigurationException, match="`devices` for `SDAAAccelerator` must be int, string or None."):
        Trainer(accelerator=SDAAAccelerator(), devices=[1])


def test_multi_optimizers_with_sdaa(tmpdir):
    class MultiOptimizerModel(BoringModel):
        def configure_optimizers(self):
            opt_a = torch.optim.Adam(self.layer.parameters(), lr=0.001)
            opt_b = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            return opt_a, opt_b

        def training_step(self, batch, batch_idx):
            opt1, opt2 = self.optimizers()
            loss = self.loss(self.step(batch))
            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            loss = self.loss(self.step(batch))
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

    model = MultiOptimizerModel()
    model.automatic_optimization = False
    model.val_dataloader = None
    _strategy = SingleSDAAStrategy()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=1,
        strategy=_strategy,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)


def test_sdaa_device_stats_monitor():
    sdaa_stats = SDAAAccelerator().get_device_stats("sdaa")
    fields = [
        "Limit",
        "InUse",
        "MaxInUse",
        "NumAllocs",
        "NumFrees",
        "ActiveAllocs",
        "MaxAllocSize",
        "TotalSystemAllocs",
        "TotalSystemFrees",
        "TotalActiveAllocs",
    ]
    for f in fields:
        assert any(f in h for h in sdaa_stats)


class MockSDAADDPStrategy(SDAADDPStrategy):
    def __init__(
        self,
        reduce_op="sum",
        **kwargs: Any,
    ):
        super().__init__()
        self.reduce_op = reduce_op
        self.logged_messages = []

    def reduce(
        self, tensor: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum"
    ) -> torch.Tensor:
        return super().reduce(tensor, group, self.reduce_op)


@pytest.mark.standalone()
@pytest.mark.skipif(num_sdaa_devices() < 2, reason="Test requires multiple SDAA devices")
@pytest.mark.parametrize(
    ("reduce_op", "expectation"),
    [
        ("sum", nullcontext()),
        ("max", nullcontext()),
        ("min", nullcontext()),
        ("mean", nullcontext()),
        ("product", nullcontext()),
        (ReduceOp.SUM, nullcontext()),
        (ReduceOp.MIN, nullcontext()),
        (ReduceOp.MAX, nullcontext()),
        (ReduceOp.AVG, nullcontext()),
        (ReduceOp.PRODUCT, nullcontext()),
    ],
    ids=[
        "sum",
        "max",
        "min",
        "mean",
        "product",
        "ReduceOp.SUM",
        "ReduceOp.MIN",
        "ReduceOp.MAX",
        "ReduceOp.AVG",
        "ReduceOp.PRODUCT",
    ],
)
def test_sdaa_supported_reduce_op(tmpdir, arg_sdaas, reduce_op, expectation):
    """Tests all reduce in SDAADDP strategy."""
    seed_everything(42)
    _model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=MockSDAADDPStrategy(reduce_op=reduce_op),
        fast_dev_run=True,
        plugins=SDAAPrecisionPlugin(precision="bf16-mixed"),
    )
    with expectation:
        trainer.fit(_model)


@pytest.mark.standalone()
@pytest.mark.skipif(num_sdaa_devices() < 2, reason="Test requires multiple SDAA devices")
@pytest.mark.parametrize(
    ("reduce_op", "logged_value_epoch", "logged_value_step"),
    [
        # Epoch = Sum(42, 43, 44), Step = Sum(44)
        ("sum", 129.0, 44.0),
        # Epoch = Max(42, 43, 44), Step = Max(44)
        ("max", 44.0, 44.0),
        # Epoch = Min(42, 43, 44), Step = Min(44)
        ("min", 42.0, 44.0),
        # Epoch = Mean(42, 43, 44), Step = Mean(44)
        ("mean", 43.0, 44.0),
    ],
)
def test_sdaa_reduce_op_logging(tmpdir, arg_sdaas, reduce_op, logged_value_epoch, logged_value_step):
    """Test reduce_op with logger and sync_dist."""
    # Logger has its own reduce_op sanity check.
    # It only accepts following string reduce_ops {min, max, mean, sum}
    # Each ddp process logs 3 values: 42, 43, and 44.
    # logger performs reduce depending on the reduce_op
    if reduce_op == "sum":
        logged_value_epoch *= arg_sdaas
        logged_value_step *= arg_sdaas

    class BaseBM(BoringModel):
        """Model to test with reduce ops."""

        def __init__(self, reduce_op=None):
            """Init."""
            super().__init__()
            self.reduce_op = reduce_op
            self.logged_value_start = 42

        def training_step(self, batch, batch_idx):
            """Training step."""
            # Each ddp process logs 3 values: 42, 43, and 44.
            # logger performs reduce depending on the reduce_op
            loss = super().training_step(batch, batch_idx)
            self.log(
                "logged_value",
                self.logged_value_start,
                prog_bar=True,
                sync_dist=True,
                reduce_fx=self.reduce_op,
                on_epoch=True,
            )
            self.logged_value_start += 1
            return loss

    seed_everything(42)
    _model = BaseBM(reduce_op=reduce_op)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=MockSDAADDPStrategy(reduce_op=reduce_op),
        max_epochs=1,
        fast_dev_run=3,
    )
    trainer.fit(_model)

    assert torch.allclose(
        trainer.callback_metrics.get("logged_value_epoch"), torch.tensor(logged_value_epoch), atol=1e-4
    )
    assert torch.allclose(trainer.callback_metrics.get(
        "logged_value_step"), torch.tensor(logged_value_step), atol=1e-4)
