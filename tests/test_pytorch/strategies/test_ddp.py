from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.fabric.plugins.environments import LightningEnvironment
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.plugins import CheckpointIO
    from lightning.pytorch.strategies import StrategyRegistry
elif module_available("pytorch_lightning"):
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.plugins import CheckpointIO
    from pytorch_lightning.strategies import StrategyRegistry

from lightning_teco.pytorch.accelerator import SDAAAccelerator
from lightning_teco.pytorch.plugins.io_plugin import SDAACheckpointIO
from lightning_teco.pytorch.strategies import SDAADDPStrategy


@pytest.mark.standalone()
def test_sdaa_ddp_strategy_init():
    bucket_cap_mb = 100
    gradient_as_bucket_view = True
    static_graph = True
    find_unused_parameters = True
    strategy = SDAADDPStrategy(
        parallel_devices=[torch.device("sdaa")] * 2,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        find_unused_parameters=find_unused_parameters,
    )
    # SDAA specific params
    assert strategy._get_process_group_backend() == "tccl"
    assert strategy.root_device == torch.device("sdaa")
    assert len(strategy.parallel_devices) == 2
    assert isinstance(strategy.checkpoint_io, SDAACheckpointIO)

    # DDP params
    assert strategy._ddp_kwargs["bucket_cap_mb"] == bucket_cap_mb
    assert strategy._ddp_kwargs["gradient_as_bucket_view"] == gradient_as_bucket_view
    assert strategy._ddp_kwargs["static_graph"] == static_graph
    assert strategy._ddp_kwargs["find_unused_parameters"] == find_unused_parameters


@pytest.mark.standalone()
def test_sdaa_ddp_strategy_device_not_sdaa(tmpdir):
    """Tests sdaa required with SDAADDPStrategy."""
    trainer = Trainer(
        default_root_dir=tmpdir, accelerator="cpu", strategy=SDAADDPStrategy(), devices=1, fast_dev_run=True
    )
    with pytest.raises(RuntimeError, match="No backend type associated with device type cpu"):
        trainer.fit(BoringModel())


@pytest.mark.standalone()
def test_sdaa_ddp_custom_strategy_registry():
    """Test custom parallel strategy registry."""

    class CustomCPIO(CheckpointIO):
        def save_checkpoint(self, checkpoint, path):
            pass

        def load_checkpoint(self, path):
            pass

        def remove_checkpoint(self, path):
            pass

    class CustomDDPStrategy(SDAADDPStrategy):
        strategy_name = "custom_sdaa_ddp"

    StrategyRegistry.register(
        "sdaa_ddp_custom_strategy",
        CustomDDPStrategy,
        description="custom SDAA Parallel strategy",
        checkpoint_io=CustomCPIO(),
    )
    trainer = Trainer(strategy="sdaa_ddp_custom_strategy",
                      accelerator=SDAAAccelerator(), devices=1)
    assert isinstance(trainer.strategy, CustomDDPStrategy)
    assert isinstance(trainer.strategy.checkpoint_io, CustomCPIO)
    assert trainer.strategy.strategy_name == "custom_sdaa_ddp"


@pytest.mark.standalone()
def test_sdaa_ddp_tensor_init_context():
    """Test that the module under the init-context gets moved to the right device."""
    strategy = SDAADDPStrategy(parallel_devices=[torch.device(
        "sdaa")], cluster_environment=LightningEnvironment())
    with strategy.tensor_init_context():
        module = torch.nn.Linear(2, 2)
    assert module.weight.device.type == module.bias.device.type == "sdaa"


@pytest.mark.standalone()
@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_sdaa_ddp_strategy_trainer_stages(tmpdir, stage, arg_sdaas):
    """Test trainer stages with sdaa_parallel_strategy."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=SDAADDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas),
        fast_dev_run=True,
    )
    with nullcontext():
        trainer_fn = getattr(trainer, stage)
        trainer_fn(model)


@pytest.mark.standalone()
@pytest.mark.parametrize(
    "reduce_op",
    [
        "sum",
        "max",
        "min",
        "mean",
    ],
)
def test_sdaa_ddp_reduce(tmpdir, arg_sdaas, reduce_op):
    """Test reduce_op with logger and sync_dist."""
    seed_everything(42)
    logged_value_arr = [torch.rand(1) for _ in range(arg_sdaas)]
    torch_function = getattr(torch, reduce_op)
    expected_value = torch_function(torch.stack(logged_value_arr))

    class BaseBM(BoringModel):
        """Model to test with reduce ops."""

        def __init__(self, reduce_op=None):
            """Init."""
            super().__init__()
            self.reduce_op = reduce_op
            self.reduced_value = None
            self.logged_value = None

        def training_step(self, batch, batch_idx):
            """Training step."""
            self.logged_value = logged_value_arr[self.trainer.strategy.local_rank]
            self.reduced_value = self.trainer.strategy.reduce(
                self.logged_value, reduce_op=reduce_op)
            return super().training_step(batch, batch_idx)

    seed_everything(42)
    _model = BaseBM(reduce_op=reduce_op)
    _strategy = SDAADDPStrategy(
        parallel_devices=[torch.device("sdaa")] * arg_sdaas)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=_strategy,
        fast_dev_run=True,
    )
    trainer.fit(_model)
    assert expected_value.item() == _model.reduced_value.item()
