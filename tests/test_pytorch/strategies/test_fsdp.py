import os
import operator
from functools import partial
from typing import Optional
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from lightning_utilities import module_available, compare_version
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, wrap
from torchmetrics import Accuracy

if module_available("lightning"):
    import lightning
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
    _LIGHTNING_GREATER_EQUAL_2_3_0 = compare_version(
        lightning.__version__, operator.ge, "2.3.0")
elif module_available("pytorch_lightning"):
    import pytorch_lightning
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel
    _LIGHTNING_GREATER_EQUAL_2_3_0 = compare_version(
        pytorch_lightning.__version__, operator.ge, "2.3.0")

from pytorch_lightning.strategies import DDPStrategy
from lightning_teco.pytorch.accelerator import SDAAAccelerator
from lightning_teco.pytorch.plugins.fsdp_precision import SDAAFSDPPrecision, SDAAPrecisionPlugin
from lightning_teco.pytorch.strategies import SDAADDPStrategy, SDAAFSDPStrategy

if not _LIGHTNING_GREATER_EQUAL_2_3_0:
    pytestmark = pytest.mark.skip(
        reason="The tests require lightning version 2.3.0 or above")


class TestFSDPModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer: Optional[nn.Module] = None

    def _init_model(self) -> None:
        self.layer = torch.nn.Sequential(torch.nn.Linear(
            32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def configure_model(self) -> None:
        if self.layer is None:
            self._init_model()
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return

        self.layer = wrap(self.layer)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.layer.parameters(), lr=0.1)

    def on_train_batch_start(self, batch, batch_idx):
        assert batch.dtype == torch.float32

    def on_train_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def on_predict_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        assert isinstance(self.layer, FullyShardedDataParallel)
        assert isinstance(
            self.trainer.strategy.precision_plugin, SDAAFSDPPrecision)

        if self.trainer.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif self.trainer.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision {self.trainer.precision}")

        assert self.layer.mixed_precision.param_dtype == param_dtype
        assert self.layer.mixed_precision.reduce_dtype == reduce_dtype
        assert self.layer.mixed_precision.buffer_dtype == buffer_dtype


class TestBoringModel(BoringModel):
    def __init__(self, wrap_min_params: int = 2):
        super().__init__()

        self.save_hyperparameters()
        self.layer = torch.nn.Sequential(torch.nn.Linear(
            32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        self.should_be_wrapped = [wrap_min_params < (
            32 * 32 + 32), None, wrap_min_params < (32 * 2 + 2)]

    def configure_optimizers(self):
        parameters = self.parameters()

        return torch.optim.AdamW(parameters, lr=0.1)


class TestFSDPModelAutoWrapped(TestBoringModel):
    def on_train_batch_start(self, batch, batch_idx):
        assert batch.dtype == torch.float32

    def on_train_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def on_predict_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        assert isinstance(self.layer, torch.nn.Sequential)
        assert isinstance(
            self.trainer.strategy.precision_plugin, SDAAFSDPPrecision)

        if self.trainer.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif self.trainer.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision {self.trainer.precision}")

        for layer_num in [0, 2]:
            if not self.should_be_wrapped[layer_num]:
                # this layer is not wrapped
                assert not isinstance(
                    self.layer[layer_num], FullyShardedDataParallel)
                continue
            assert isinstance(self.layer[layer_num], FullyShardedDataParallel)
            assert self.layer[layer_num].mixed_precision.param_dtype == param_dtype
            assert self.layer[layer_num].mixed_precision.reduce_dtype == reduce_dtype
            assert self.layer[layer_num].mixed_precision.buffer_dtype == buffer_dtype


def test_fsdp_custom_mixed_precision():
    """Test to ensure that passing a custom mixed precision config works."""
    config = MixedPrecision()
    strategy = SDAAFSDPStrategy(mixed_precision=config)
    assert strategy.mixed_precision_config == config


@pytest.mark.skipif(SDAAAccelerator.auto_device_count() <= 1, reason="Test requires multiple SDAA devices")
def test_fsdp_strategy_sync_batchnorm(tmpdir, arg_sdaas):
    """Test to ensure that sync_batchnorm works when using FSDP on SDAA."""
    if arg_sdaas <= 1:
        pytest.skip(reason="Test requires multiple cards")

    model = TestBoringModel()
    config = CPUOffload()

    trainer = Trainer(
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=SDAAFSDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas,
            cpu_offload=config,
            precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
        ),
        max_epochs=1,
        enable_checkpointing=False,
        sync_batchnorm=True,
    )

    trainer.fit(model)


@pytest.mark.parametrize("strategy", ["SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"])
def test_fsdp_simple_model(strategy, arg_sdaas):
    model = TestBoringModel()

    trainer = Trainer(
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=SDAAFSDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas,
            sharding_strategy=strategy,
            precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
        ),
        max_epochs=1,
        enable_checkpointing=False,
        sync_batchnorm=True,
    )

    trainer.fit(model)


@pytest.mark.xfail(run=False, reason="To be fixed.Failure post 1.17 upgrade.")
@pytest.mark.parametrize("strategy", ["SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"])
def test_fsdp_simple_model_activation_cp(strategy, arg_sdaas):
    model = BoringModel()

    trainer = Trainer(
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        num_sanity_val_steps=0,
        strategy=SDAAFSDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas,
            sharding_strategy=strategy,
            precision_plugin=SDAAFSDPPrecision("32-true"),
            activation_checkpointing_policy={torch.nn.Linear},
        ),
        max_epochs=1,
        fast_dev_run=1,
    )

    trainer.fit(model)


@pytest.mark.xfail(run=False, reason="Failure in applying autocast during recompute.")
@pytest.mark.parametrize("strategy", ["SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"])
def test_fsdp_simple_model_activation_cp_mixed_precision(strategy, arg_sdaas):
    model = BoringModel()

    trainer = Trainer(
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        num_sanity_val_steps=0,
        strategy=SDAAFSDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas,
            sharding_strategy=strategy,
            precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
            activation_checkpointing_policy={torch.nn.Linear},
        ),
        max_epochs=1,
        fast_dev_run=1,
    )

    trainer.fit(model)


@pytest.mark.skipif(SDAAAccelerator.auto_device_count() <= 1, reason="Test requires multiple SDAA devices.")
@pytest.mark.standalone()
def test_fsdp_strategy_simple_model_compile(tmpdir, arg_sdaas):
    """Test to ensure that sync_batchnorm works when using FSDP and SDAA."""
    if arg_sdaas <= 1:
        pytest.skip(reason="Test requires multiple cards")

    model = TestBoringModel()
    config = CPUOffload()
    compiled_model = torch.compile(model, backend="sdaa_backend")

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=SDAAFSDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas,
            cpu_offload=config,
            precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
        ),
        max_epochs=1,
        sync_batchnorm=False,
        enable_checkpointing=False,
    )
    trainer.fit(compiled_model)


@pytest.mark.standalone()
def test_fsdp_modules_without_parameters(tmpdir, arg_sdaas):
    """Test that TorchMetrics get moved to the device despite not having any parameters."""

    class MetricsModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy("multiclass", num_classes=10)
            assert self.metric.device == self.metric.tp.device == torch.device(
                "cpu")

        def setup(self, stage) -> None:
            assert self.metric.device == self.metric.tp.device == torch.device(
                "cpu")

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            assert self.metric.device == self.metric.tp.device == torch.device(
                "sdaa", 0)
            self.metric(torch.ones(10, device=self.device),
                        torch.ones(10, device=self.device))
            return loss

    model = MetricsModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=SDAAFSDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas,
            cpu_offload=True,
            precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
        ),
        max_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(model)


# # @pytest.skip(reason="not support save checkpoint for fsdp")
# @pytest.mark.parametrize("state_dict_type", ["sharded", "full"])
# @pytest.mark.standalone()
# def test_fsdp_strategy_checkpoint(tmpdir, arg_sdaas, state_dict_type):
#     """Test to ensure that checkpoint is saved and loaded correctly when using a SDAA."""
#     model = TestFSDPModel()
#     trainer = Trainer(
#         default_root_dir=tmpdir,
#         accelerator=SDAAAccelerator(),
#         strategy=SDAAFSDPStrategy(
#             parallel_devices=[torch.device("sdaa")] * arg_sdaas,
#             precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
#             state_dict_type=state_dict_type,
#         ),
#         max_epochs=1,
#     )

#     trainer.fit(model)
#     trainer.save_checkpoint(os.path.join(tmpdir, "last.ckpt"))

#     trainer = Trainer(
#         default_root_dir=tmpdir,
#         accelerator=SDAAAccelerator(),
#         strategy=SDAAFSDPStrategy(
#             parallel_devices=[torch.device("sdaa")] * arg_sdaas,
#             precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
#             state_dict_type=state_dict_type,
#         ),
#         max_epochs=1,
#     )

#     trainer.fit(model, ckpt_path=os.path.join(tmpdir, "last.ckpt"))


@pytest.mark.standalone()
@pytest.mark.parametrize("wrap_min_params", [1024])
def test_fsdp_strategy_full_state_dict(tmpdir, wrap_min_params, arg_sdaas):
    """Test to ensure that the full state dict is extracted when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all.

    """
    model = TestFSDPModelAutoWrapped(wrap_min_params=wrap_min_params)
    correct_state_dict = model.state_dict()  # State dict before wrapping

    strategy = SDAAFSDPStrategy(
        parallel_devices=[torch.device("sdaa")] * arg_sdaas,
        auto_wrap_policy=partial(
            size_based_auto_wrap_policy, min_num_params=wrap_min_params),
        precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=strategy,
        max_epochs=1,
        barebones=True,
    )
    trainer.fit(model)

    full_state_dict = trainer.strategy.lightning_module_state_dict()

    if trainer.global_rank != 0:
        assert len(full_state_dict) == 0
        return

    # State dict should contain same number of keys
    assert len(correct_state_dict) == len(full_state_dict)
    # OrderedDict should return the same keys in the same order
    assert all(_ex == _co for _ex, _co in zip(
        full_state_dict.keys(), correct_state_dict.keys()))


def test_fsdp_strategy_cpu_offload():
    """Test the different ways cpu offloading can be enabled."""
    # bool
    strategy = SDAAFSDPStrategy(cpu_offload=True)
    assert strategy.cpu_offload == CPUOffload(offload_params=True)

    # dataclass
    config = CPUOffload()
    strategy = SDAAFSDPStrategy(cpu_offload=config)
    assert strategy.cpu_offload == config


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
    ],
)
def test_configure_model(tmpdir, arg_sdaas, precision, expected_dtype):
    """Test that the module under configure_model gets moved to the right device and dtype."""
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=SDAAFSDPStrategy(
            parallel_devices=[torch.device("sdaa")] * arg_sdaas,
            sharding_strategy="SHARD_GRAD_OP",
            precision_plugin=SDAAFSDPPrecision(precision),
        ),
        max_epochs=1,
        enable_checkpointing=False,
        sync_batchnorm=True,
    )

    class MyModel(BoringModel):
        def configure_model(self):
            self.layer = torch.nn.Linear(32, 2)
            expected_device = torch.device("cpu")
            assert self.layer.weight.device == expected_device
            assert self.layer.weight.dtype == expected_dtype

        def configure_optimizers(self):
            return torch.optim.AdamW(self.layer.parameters(), lr=0.1)

        def on_fit_start(self):
            assert self.layer.weight.device == torch.device(
                "sdaa", torch.sdaa.current_device())
            assert self.layer.weight.dtype == expected_dtype

    model = MyModel()
    trainer.fit(model)


def test_fsdp_sharding_strategy():
    """Test the different ways the sharding strategy can be set."""
    from torch.distributed.fsdp import ShardingStrategy

    # default
    strategy = SDAAFSDPStrategy()
    assert strategy.sharding_strategy == ShardingStrategy.FULL_SHARD

    # enum
    strategy = SDAAFSDPStrategy(
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
    assert strategy.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP

    # string
    strategy = SDAAFSDPStrategy(sharding_strategy="NO_SHARD")
    assert strategy.sharding_strategy == ShardingStrategy.NO_SHARD
    strategy = SDAAFSDPStrategy(sharding_strategy="no_shard")
    assert strategy.sharding_strategy == ShardingStrategy.NO_SHARD


def test_fsdp_activation_checkpointing():
    """Test that the FSDP strategy can apply activation checkpointing to the given layers."""

    class Block1(nn.Linear):
        pass

    class Block2(nn.Linear):
        pass

    class Model(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Sequential(Block1(4, 4), Block1(5, 5))
            self.layer1 = Block2(2, 2)
            self.layer2 = nn.Linear(3, 3)

    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    strategy = SDAAFSDPStrategy(activation_checkpointing_policy={Block1})
    assert set(strategy._activation_checkpointing_kwargs) == {
        "auto_wrap_policy"}
    assert isinstance(
        strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

    strategy = SDAAFSDPStrategy(
        activation_checkpointing_policy=ModuleWrapPolicy({Block1, Block2}))
    assert set(strategy._activation_checkpointing_kwargs) == {
        "auto_wrap_policy"}
    assert isinstance(
        strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

    model = Model()
    strategy._parallel_devices = [torch.device("sdaa", 0)]
    strategy._lightning_module = model
    strategy._process_group = Mock()
    with mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", new=MagicMock), mock.patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
    ) as apply_mock:
        wrapped = strategy._setup_model(model)
    apply_mock.assert_called_with(
        wrapped, checkpoint_wrapper_fn=ANY, **strategy._activation_checkpointing_kwargs)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        pytest.param(
            "bf16-mixed",
            (torch.float32, torch.bfloat16, torch.bfloat16),
        ),
        pytest.param(
            "32-true",
            (torch.float32, torch.float32, torch.float32),
        ),
    ],
)
def test_fsdp_precision_config(precision, expected):
    plugin = SDAAFSDPPrecision(precision=precision)
    config = plugin.mixed_precision_config

    assert config.param_dtype == expected[0]
    assert config.buffer_dtype == expected[1]
    assert config.reduce_dtype == expected[2]


@pytest.mark.parametrize("wrap_min_params", [1024])
@pytest.mark.standalone()
def test_fsdp_strategy_save_optimizer_states(tmpdir, wrap_min_params, arg_sdaas):
    """Test to ensure that the full state dict and optimizer states is saved when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all. If the model can
    be restored to DDP, it means that the optimizer states were saved correctly.

    """
    if arg_sdaas <= 1:
        pytest.skip(reason="Test requires multiple cards")

    model = TestFSDPModelAutoWrapped(wrap_min_params=wrap_min_params)
    strategy = SDAAFSDPStrategy(
        parallel_devices=[torch.device("sdaa")] * arg_sdaas,
        auto_wrap_policy=partial(
            size_based_auto_wrap_policy, min_num_params=wrap_min_params),
        precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
    )

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=strategy,
        max_epochs=1,
    )

    trainer.fit(model)
    model_path = os.path.join(tmpdir, "last.ckpt")
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)

    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank != 0:
        assert len(model_state_dict) == 0

    if trainer.global_rank != 0:
        assert len(optimizer_state_dict) == 0

    # restore model to ddp
    parallel_sdaas = [torch.device("sdaa")] * arg_sdaas
    _strategy = SDAADDPStrategy(parallel_devices=parallel_sdaas)
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, accelerator=SDAAAccelerator(), devices=arg_sdaas, strategy=_strategy
    )
    model = TestBoringModel()

    # This step will restore the model and optimizer states
    trainer.fit(model, ckpt_path=model_path)

    # Get the model and optimizer states from the restored ddp model
    restored_model_state_dict = trainer.strategy.lightning_module_state_dict()
    restored_optimizer_state_dict = trainer.strategy.optimizer_state(
        model.optimizers())

    if trainer.global_rank == 0:
        # assert everything is the same
        assert len(model_state_dict) == len(restored_model_state_dict)
        assert len(optimizer_state_dict) == len(restored_optimizer_state_dict)

        torch.testing.assert_close(
            model_state_dict, restored_model_state_dict, atol=0, rtol=0)
        torch.testing.assert_close(
            optimizer_state_dict, restored_optimizer_state_dict, atol=0, rtol=0)

    trainer.strategy.barrier()


@pytest.mark.parametrize("wrap_min_params", [2, 1024, 100000000])
@pytest.mark.standalone()
def test_fsdp_strategy_load_optimizer_states(tmpdir, wrap_min_params, arg_sdaas):
    """Test to ensure that the full state dict and optimizer states can be load when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all. If the DDP model
    can be restored to FSDP, it means that the optimizer states were restored correctly.

    """
    if arg_sdaas <= 1:
        pytest.skip(reason="Test requires multiple cards")

    # restore model to ddp
    model = TestBoringModel()
    parallel_sdaas = [torch.device("sdaa")] * arg_sdaas
    _strategy = SDAADDPStrategy(parallel_devices=parallel_sdaas)
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, accelerator=SDAAAccelerator(), devices=arg_sdaas, strategy=_strategy
    )
    # This step will restore the model and optimizer states
    trainer.fit(model)
    model_path = os.path.join(tmpdir, "last.ckpt")
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)

    # Get the model and optimizer states from the restored ddp model
    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    # Build a new FSDP model
    model = TestFSDPModelAutoWrapped(wrap_min_params=wrap_min_params)

    strategy = SDAAFSDPStrategy(
        parallel_devices=[torch.device("sdaa")] * arg_sdaas,
        auto_wrap_policy=partial(
            size_based_auto_wrap_policy, min_num_params=wrap_min_params),
        precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=strategy,
        max_epochs=1,
    )

    trainer.fit(model, ckpt_path=model_path)

    restored_model_state_dict = trainer.strategy.lightning_module_state_dict()
    restored_optimizer_state_dict = trainer.strategy.optimizer_state(
        model.optimizers())

    if trainer.global_rank != 0:
        assert len(restored_model_state_dict) == 0

    if trainer.global_rank != 0:
        assert len(restored_optimizer_state_dict) == 0

    if trainer.global_rank == 0:
        # assert everything is the same
        assert len(model_state_dict) == len(restored_model_state_dict)
        assert len(optimizer_state_dict) == len(restored_optimizer_state_dict)
        torch.testing.assert_close(
            model_state_dict, restored_model_state_dict, atol=0, rtol=0)
        torch.testing.assert_close(
            optimizer_state_dict, restored_optimizer_state_dict, atol=0, rtol=0)

    trainer.strategy.barrier()


def test_dummy_fsdp_string_init(tmpdir):
    """Test that TorchMetrics get moved to the device despite not having any parameters."""

    class DummyFSDPStrategy(SDAAFSDPStrategy):
        strategy_name = "dummy_sdaa_fsdp"

    model = BoringModel()
    dm = BoringDataModule()
    _strategy = DummyFSDPStrategy(
        precision_plugin=SDAAFSDPPrecision("bf16-mixed"))

    # Check strategy string init name
    assert _strategy.strategy_name == "dummy_sdaa_fsdp"

    # Trainer is able to run fit with dummy policy without policy registration issue
    trainer = Trainer(
        accelerator=SDAAAccelerator(), devices=1, strategy=_strategy, fast_dev_run=True, enable_model_summary=True
    )
    trainer.fit(model, dm)


class AccuracyTestModel(BoringModel):
    """Model to test with precision Plugin."""

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = super().training_step(batch, batch_idx)
        self.log("train_loss", loss.get("loss").to(
            torch.bfloat16), prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = super().validation_step(batch, batch_idx)
        self.log("val_loss", loss.get("x").to(
            torch.bfloat16), prog_bar=True, sync_dist=True)
        return loss


def run_training(root_dir, model, dm, strategy, arg_sdaas):
    seed_everything(42)
    trainer = Trainer(
        default_root_dir=root_dir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=strategy,
        plugins=None if isinstance(strategy, SDAAFSDPStrategy) else SDAAPrecisionPlugin(
            precision="bf16-mixed"),
        fast_dev_run=True,
    )
    trainer.fit(model(), dm())
    return trainer.callback_metrics["val_loss"], trainer.callback_metrics["train_loss"]


@pytest.mark.standalone()
def test_sdaa_parallel_precision_accuracy(tmpdir, arg_sdaas):
    def run_training(root_dir, device, model, dm, strategy, num_devices):
        seed_everything(42)
        trainer = Trainer(
            default_root_dir=root_dir,
            accelerator=device,
            devices=num_devices,
            strategy=strategy,
            plugins=None,
            fast_dev_run=True,
        )
        trainer.fit(model(), dm())
        return trainer.callback_metrics["val_loss"], trainer.callback_metrics["train_loss"]

    parallel_sdaas = [torch.device("sdaa")] * arg_sdaas
    val_loss, train_loss = val_loss, train_loss = run_training(
        tmpdir, "sdaa", AccuracyTestModel, BoringDataModule, SDAADDPStrategy(
            parallel_devices=parallel_sdaas), arg_sdaas
    )

    expected_train_loss = torch.tensor(0.9643)
    expected_val_loss = torch.tensor(2.3879)
    assert torch.allclose(train_loss, expected_train_loss,
                          rtol=1e-4, atol=1e-4), f"train_loss: {train_loss}, expected_train_loss: {expected_train_loss}"
    assert torch.allclose(val_loss, expected_val_loss, rtol=1e-4,
                          atol=1e-2), f"val_loss: {val_loss}, expected_val_loss: {expected_val_loss}"


@pytest.mark.standalone()
def test_sdaa_fsdp_precision_accuracy(tmpdir, arg_sdaas):
    fsdp_strategy = SDAAFSDPStrategy(
        parallel_devices=[torch.device("sdaa")] * arg_sdaas,
        sharding_strategy="FULL_SHARD",
        precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
    )
    val_loss, train_loss = run_training(
        tmpdir, AccuracyTestModel, BoringDataModule, fsdp_strategy, arg_sdaas)
    # sdaas == 1
    expected_train_loss = torch.tensor(0.9688)
    expected_val_loss = torch.tensor(0.6016)
    if arg_sdaas == 2:
        expected_train_loss = torch.tensor(1.0)
        expected_val_loss = torch.tensor(2.5781)
    assert torch.allclose(train_loss, expected_train_loss,
                          rtol=1e-4, atol=1e-4)
    assert torch.allclose(val_loss, expected_val_loss, rtol=1e-2, atol=1e-2)


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
def test_sdaa_fsdp_reduce(tmpdir, arg_sdaas, reduce_op):
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
    _strategy = SDAAFSDPStrategy(
        parallel_devices=[torch.device("sdaa")] * arg_sdaas,
        sharding_strategy="FULL_SHARD",
        precision_plugin=SDAAFSDPPrecision("bf16-mixed"),
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=SDAAAccelerator(),
        devices=arg_sdaas,
        strategy=_strategy,
        fast_dev_run=True,
    )
    trainer.fit(_model)
    assert expected_value.item() == _model.reduced_value.item()
