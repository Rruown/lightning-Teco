import json
import os
from typing import Any, Dict

import pytest
import torch
from lightning_utilities import module_available
from torch import Tensor
import torch.distributed
from torch.utils.data import DataLoader, Dataset

if module_available("lightning"):
    from lightning.pytorch import LightningModule, Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule, Trainer, seed_everything
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_teco.pytorch.accelerator import SDAAAccelerator
from lightning_teco.pytorch.plugins import SDAADeepSpeedPrecisionPlugin
from lightning_teco.pytorch.strategies import SDAADeepSpeedStrategy

from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


class ModelParallelBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def configure_sharded_model(self) -> None:
        self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.configure_sharded_model()


class ModelParallelBoringModelNoSchedulers(ModelParallelBoringModel):
    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


class ModelParallelBoringModelManualOptim(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def configure_sharded_model(self) -> None:
        self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.configure_sharded_model()

    @property
    def automatic_optimization(self) -> bool:
        return False


@pytest.fixture()
def deepspeed_base_config():
    return {
        "train_batch_size": 2,
        "fp16": {"enabled": True},
        "bf16": {"enabled": False},
        "train_micro_batch_size_per_gpu": 2,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0.02,
                "warmup_max_lr": 0.05,
                "warmup_num_steps": 4,
                "total_num_steps": 2,
                "warmup_type": "linear",
            },
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {"stage": 0},
        "zero_force_ds_cpu_optimizer": False,  # TBD : Fix custom optim offload support
    }


def config_generator(
    deepspeed_base_config,
    stage,
    cpu_offload,
    activation_checkpoints,
    partition_activations,
    contiguous_checkpointing,
    checkpoint_in_cpu,
):
    deepspeed_config = {**deepspeed_base_config}

    deepspeed_config["zero_optimization"]["stage"] = stage
    if stage == "infinity":
        deepspeed_config["zero_optimization"]["stage"] = 3
        deepspeed_config["zero_optimization"]["offload_param"] = {
            "device": "cpu"}

    if cpu_offload:
        deepspeed_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu"}
        deepspeed_config["zero_optimization"]["contiguous_gradients"] = True
        deepspeed_config["zero_optimization"]["overlap_comm"] = True

    if stage != 0 and activation_checkpoints:
        deepspeed_config["activation_checkpointing"] = {
            "partition_activations": partition_activations,
            "contiguous_memory_optimization": False,
            "cpu_checkpointing": checkpoint_in_cpu,
        }

    if stage == 0:
        deepspeed_config["fp16"]["enabled"] = False

    return deepspeed_config


@pytest.fixture()
def deepspeed_config():
    return {
        "optimizer": {"type": "SGD", "params": {"lr": 3e-5}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {"last_batch_iteration": -1, "warmup_min_lr": 0, "warmup_max_lr": 3e-5, "warmup_num_steps": 100},
        },
    }


@pytest.fixture()
def deepspeed_zero_config(deepspeed_config):
    return {**deepspeed_config, "zero_allow_untested_optimizer": True, "zero_optimization": {"stage": 2}}


@pytest.fixture()
def deepspeed_zero_autotuning_config():
    return {
        "fp16": {"enabled": True},
        "autotuning": {
            "enabled": True,
            "arg_mappings": {
                "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
                "gradient_accumulation_steps ": "--gradient_accumulation_steps",
            },
        },
    }


@pytest.mark.standalone()
@pytest.mark.skipif(SDAAAccelerator.auto_device_count() <= 1, reason="Test requires multiple SDAA devices")
def test_sdaa_deepspeed_strategy_env(tmpdir, monkeypatch, deepspeed_config):
    """Test to ensure that the strategy can be passed via a string with an environment variable."""
    config_path = os.path.join(tmpdir, "temp.json")
    with open(config_path, "w") as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", config_path)

    trainer = Trainer(
        accelerator="sdaa", fast_dev_run=True, default_root_dir=tmpdir, strategy=SDAADeepSpeedStrategy()
    )  # strategy="sdaa_deepspeed")

    strategy = trainer.strategy
    assert isinstance(strategy, SDAADeepSpeedStrategy)
    assert len(trainer.strategy.parallel_devices) > 1
    assert trainer.strategy.parallel_devices[0].type == "sdaa"


@pytest.mark.standalone()
@pytest.mark.parametrize(
    "precision",
    [
        "fp16-mixed",
    ],
)
def test_sdaa_deepspeed_precision_choice(tmpdir, precision):
    """Tests precision plugin with supported precisions."""
    _plugins = [SDAADeepSpeedPrecisionPlugin(precision=precision)]
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        accelerator="sdaa",
        strategy=SDAADeepSpeedStrategy(),  # strategy="sdaa_deepspeed",
        plugins=_plugins,
    )

    assert isinstance(trainer.strategy, SDAADeepSpeedStrategy)
    assert isinstance(trainer.strategy.precision_plugin,
                      SDAADeepSpeedPrecisionPlugin)
    assert trainer.strategy.precision_plugin.precision == precision


@pytest.mark.standalone()
def test_sdaa_deepspeed_with_invalid_config_path():
    """Test to ensure if we pass an invalid config path we throw an exception."""
    with pytest.raises(
        MisconfigurationException, match="You passed in a path to a DeepSpeed config but the path does not exist"
    ):
        SDAADeepSpeedStrategy(config="invalid_path.json")


@pytest.mark.standalone()
def test_deepspeed_defaults():
    """Ensure that defaults are correctly set as a config for DeepSpeed if no arguments are passed."""
    strategy = SDAADeepSpeedStrategy()
    assert strategy.config is not None
    assert isinstance(strategy.config["zero_optimization"], dict)


class SampleDataset(Dataset):
    def __init__(self, batch_size, data_size):
        x = torch.ones([batch_size, data_size],
                       dtype=torch.float, device="sdaa")
        y = torch.zeros([batch_size, data_size],
                        dtype=torch.float, device="sdaa")
        self.train_data = (x, y)

    def __getitem__(self, index):
        """Get a sample."""
        return (self.train_data[0][index], self.train_data[1][index])

    def __len__(self):
        """Get length of dataset."""
        return self.train_data[0].size(0)


class SampleLayer(torch.nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones([data_size], dtype=torch.float))

    def forward(self, input):
        return input * self.w


class SampleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = SampleLayer(10)
        self.l2 = SampleLayer(10)
        self.l3 = SampleLayer(10)
        self.l4 = SampleLayer(10)

    def forward(self, x):
        l1_out = self.l1(x)
        l2_out = checkpoint(self.l2, l1_out)
        l3_out = checkpoint(self.l3, l2_out)
        return checkpoint(self.l4, l3_out)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.sum(torch.abs(y - logits)) / (2 * 10 * 50)
        self.log("train_loss", loss.item(), sync_dist=True, reduce_fx="sum")
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.sum(torch.abs(y - logits)) / (2 * 10 * 50)
        self.log("valid_loss", loss, sync_dist=True, reduce_fx="sum")
        return {"loss": loss, "logits": logits}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.sum(torch.abs(y - logits)) / (2 * 10 * 50)
        self.log("test_loss", loss, sync_dist=True, reduce_fx="sum")
        return {"loss": loss, "logits": logits}

    def configure_optimizers(self):
        from torch.optim.adamw import AdamW as AdamW

        return torch.optim.AdamW(self.parameters())

    def train_dataloader(self):
        return DataLoader(SampleDataset(16, 10), batch_size=2)

    def val_dataloader(self):
        return DataLoader(SampleDataset(16, 10), batch_size=2)

    def test_dataloader(self):
        return DataLoader(SampleDataset(16, 10), batch_size=2)


@pytest.mark.standalone()
def test_deepspeed_config(tmpdir):
    """Test to ensure deepspeed config works correctly.

    DeepSpeed config object including optimizers/schedulers and saves the model weights to load correctly.

    """

    class TestCB(Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            from torch.optim.lr_scheduler import StepLR

            assert isinstance(trainer.optimizers[0], DeepSpeedZeroOptimizer)
            assert isinstance(trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert isinstance(
                trainer.lr_scheduler_configs[0].scheduler, StepLR)
            assert trainer.lr_scheduler_configs[0].interval == "epoch"

    model = BoringModel()
    lr_monitor = LearningRateMonitor()
    _plugins = [SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")]
    trainer = Trainer(
        accelerator="sdaa",
        strategy=SDAADeepSpeedStrategy(
            parallel_devices=[torch.device("sdaa")]),
        default_root_dir=tmpdir,
        devices=1,
        log_every_n_steps=1,
        limit_train_batches=4,
        limit_val_batches=4,
        limit_test_batches=4,
        max_epochs=2,
        plugins=_plugins,
        callbacks=[TestCB(), lr_monitor],
        logger=CSVLogger(tmpdir),
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model)
    trainer.test(model)
    assert list(lr_monitor.lrs) == ["lr-SGD"]
    assert len(set(lr_monitor.lrs["lr-SGD"])) == trainer.max_epochs


class SomeDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        """Get a sample."""
        return self.data[index]

    def __len__(self):
        """Get length of dataset."""
        return self.len


class SomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(SomeDataset(32, 64), batch_size=2)

    def val_dataloader(self):
        return DataLoader(SomeDataset(32, 64), batch_size=2)


@pytest.mark.standalone()
def test_multi_optimizer_with_sdaa_deepspeed(tmpdir):
    """Test to validate multi optimizer support with deepspeed."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def configure_optimizers(self):
            optimizer1 = torch.optim.AdamW(self.parameters())
            optimizer2 = torch.optim.AdamW(self.parameters())
            return [optimizer1, optimizer2]

    _plugins = [SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")]
    model = TestModel()
    trainer = Trainer(
        accelerator="sdaa",
        fast_dev_run=True,
        default_root_dir=tmpdir,
        strategy=SDAADeepSpeedStrategy(
            parallel_devices=[torch.device("sdaa")]),
        plugins=_plugins,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(
        MisconfigurationException, match="DeepSpeed currently only supports single optimizer, single optional scheduler"
    ):
        trainer.fit(model)


@pytest.mark.standalone()
@pytest.mark.skipif(SDAAAccelerator.auto_device_count() <= 1, reason="Test requires multiple SDAA devices")
@pytest.mark.parametrize(
    "zero_config",
    [
        0,
        1,
        2,
        pytest.param(
            "infinity",
            marks=pytest.mark.skipif(
                SDAAAccelerator.auto_device_count() > 1, reason="Test will block, maybe a bug for sdaa."
            ),
        ),
    ],
)
@pytest.mark.parametrize("cpu_offload", [True, False])
@pytest.mark.parametrize(
    ("activation_checkpoints", "partition_activations",
     "contiguous_checkpointing", "checkpoint_in_cpu"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
        pytest.param(
            True, True, True, True,
            marks=pytest.mark.skipif(
                # TODO: when enable checkpoint_in_cpu, it will fail at zero1
                SDAAAccelerator.auto_device_count() > 1, reason="Test will failed, maybe a bug for sdaa."
            ),
        ),
    ],
)
def test_lightning_model(
    deepspeed_base_config,
    zero_config,
    cpu_offload,
    activation_checkpoints,
    partition_activations,
    contiguous_checkpointing,
    checkpoint_in_cpu,
    device_count,
):
    """Test that DeepSpeed works with a simple LightningModule and LightningDataModule."""
    seed_everything(42)
    config = config_generator(
        deepspeed_base_config,
        zero_config,
        cpu_offload,
        activation_checkpoints,
        partition_activations,
        contiguous_checkpointing,
        checkpoint_in_cpu,
    )

    model = SampleModel()
    _plugins = [SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")]
    _accumulate_grad_batches = config["train_micro_batch_size_per_gpu"]
    _batch_size = 2
    _parallel_sdaas = [torch.device("sdaa")] * device_count

    config["train_batch_size"] = device_count * \
        _accumulate_grad_batches * _batch_size

    trainer = Trainer(
        accelerator="sdaa",
        strategy=SDAADeepSpeedStrategy(
            config=config, parallel_devices=_parallel_sdaas),
        enable_progress_bar=False,
        fast_dev_run=10,
        plugins=_plugins,
        use_distributed_sampler=False,
        limit_train_batches=16,
        accumulate_grad_batches=_accumulate_grad_batches,
    )

    trainer.fit(model)
    expected = torch.tensor([0.0654])
    current_loss = trainer.callback_metrics["train_loss"].detach().to("cpu")
    assert torch.allclose(
        current_loss, expected, atol=2e-2
    ), f"incorrect loss value {current_loss}, expected {expected}"


@pytest.mark.standalone()
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("offload", [True, False])
def test_lightning_deepspeed_stages(device_count, zero_stage, offload):
    model = SampleModel()
    trainer = Trainer(
        accelerator="sdaa",
        devices=device_count,
        strategy=SDAADeepSpeedStrategy(
            zero_optimization=True,
            stage=zero_stage,
            offload_optimizer=offload,
            parallel_devices=[torch.device("sdaa")] * device_count,
        ),
        plugins=[SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")],
        fast_dev_run=2,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        limit_train_batches=16,
        accumulate_grad_batches=2,
    )
    trainer.fit(model)


@pytest.mark.standalone()
def test_sdaa_deepspeed_with_invalid_optimizer():
    """Test to ensure if we pass an invalid optimizer and throws an exception."""

    class DummyModel(BoringModel):
        def configure_optimizers(self):
            return None

    import logging

    model = DummyModel()
    _plugins = [SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")]
    trainer = Trainer(
        accelerator="sdaa",
        strategy=SDAADeepSpeedStrategy(
            logging_level=logging.INFO, parallel_devices=[torch.device("sdaa")]),
        max_epochs=1,
        plugins=_plugins,
        devices=1,
    )
    with pytest.raises(
        MisconfigurationException, match="You have specified an invalid optimizer to be run with deepspeed."
    ):
        trainer.fit(model)


@pytest.mark.standalone()
def test_sdaa_deepspeed_with_optimizer_and_config(deepspeed_zero_config):
    """Test the preference of optimizer when configured both from deepspeed config and LightningModule."""

    class DummyModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=0.1)

    class TestCB(Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            from deepspeed.runtime.lr_schedules import WarmupLR

            assert isinstance(trainer.optimizers[0], DeepSpeedZeroOptimizer)
            assert isinstance(
                trainer.optimizers[0].optimizer, torch.optim.AdamW)
            assert isinstance(
                trainer.lr_scheduler_configs[0].scheduler, WarmupLR)
            assert trainer.lr_scheduler_configs[0].interval == "step"

    import logging

    model = DummyModel()

    _plugins = [SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")]
    trainer = Trainer(
        accelerator="sdaa",
        strategy=SDAADeepSpeedStrategy(
            logging_level=logging.INFO, config=deepspeed_zero_config, parallel_devices=[
                torch.device("sdaa")]
        ),
        callbacks=[TestCB()],
        max_epochs=1,
        plugins=_plugins,
        devices=1,
    )
    trainer.fit(model)


@pytest.mark.standalone()
@pytest.mark.skipif(SDAAAccelerator.auto_device_count() <= 1, reason="Test requires multiple SDAA devices")
def test_deepspeed_resume_training(tmpdir, deepspeed_base_config, device_count):
    """Test to ensure with Stage 3 and single GPU that we can resume training."""
    initial_model = SampleModel()
    _plugins = [SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")]
    _zero_stage = 3
    config = config_generator(
        deepspeed_base_config,
        _zero_stage,
        True,
        True,
        False,
        False,
        True,
    )
    _accumulate_grad_batches = config["train_micro_batch_size_per_gpu"]
    _batch_size = 2
    _parallel_sdaas = [torch.device("sdaa")] * device_count

    config["train_batch_size"] = device_count * \
        _accumulate_grad_batches * _batch_size
    config_copy = config.copy()
    ck = ModelCheckpoint(monitor="train_loss", mode="max",
                         save_last=True, save_top_k=-1)
    initial_trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=SDAADeepSpeedStrategy(
            config=config, parallel_devices=_parallel_sdaas),
        accelerator="sdaa",
        accumulate_grad_batches=_accumulate_grad_batches,
        plugins=_plugins,
        callbacks=[ck],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    initial_trainer.fit(initial_model)

    class TestCallback(Callback):
        def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            original_deepspeed_strategy = initial_trainer.strategy
            current_deepspeed_strategy = trainer.strategy

            assert isinstance(original_deepspeed_strategy,
                              SDAADeepSpeedStrategy)
            assert isinstance(current_deepspeed_strategy,
                              SDAADeepSpeedStrategy)
            # assert optimizer states are the correctly loaded
            original_optimizer_dict = original_deepspeed_strategy.deepspeed_engine.optimizer.state_dict()
            current_optimizer_dict = current_deepspeed_strategy.deepspeed_engine.optimizer.state_dict()
            for orig_tensor, current_tensor in zip(
                original_optimizer_dict["fp32_flat_groups"], current_optimizer_dict["fp32_flat_groups"]
            ):
                assert torch.all(orig_tensor.eq(current_tensor))
            # assert model state is loaded correctly
            for current_param, initial_param in zip(pl_module.parameters(), initial_model.parameters()):
                assert torch.equal(current_param.cpu(), initial_param.cpu())
            # assert epoch has correctly been restored
            assert trainer.current_epoch == 1

            # assert lr-scheduler states are loaded correctly
            original_lr_scheduler = initial_trainer.lr_scheduler_configs[0].scheduler
            current_lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert original_lr_scheduler.state_dict() == current_lr_scheduler.state_dict()

    model = SampleModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=SDAADeepSpeedStrategy(
            config=config_copy, parallel_devices=_parallel_sdaas),
        accelerator="sdaa",
        accumulate_grad_batches=_accumulate_grad_batches,
        plugins=_plugins,
        callbacks=TestCallback(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, ckpt_path=ck.best_model_path)


class TestLayer(torch.nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones([data_size]))

    def forward(self, input):
        return input * torch.matmul(input, self.w)


class InferenceModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = TestLayer(2)
        self.l2 = TestLayer(2)
        self.l3 = TestLayer(2)
        self.l4 = TestLayer(2)

    def forward(self, x):
        l1_out = self.l1(x)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        return self.l4(l3_out)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def predict_dataloader(self):
        return DataLoader(SampleDataset(1, 2))


@pytest.mark.standalone()
@pytest.mark.skipif(SDAAAccelerator.auto_device_count() <= 1, reason="Test requires multiple SDAA devices")
@pytest.mark.parametrize("enable_cuda_graph", [False])
def test_lightning_deepspeed_inference_kwargs(enable_cuda_graph, device_count):
    model = InferenceModel()
    kwargs = {"dtype": torch.float}
    kwargs["tensor_parallel"] = {"tp_size": device_count}
    kwargs["enable_cuda_graph"] = enable_cuda_graph
    kwargs["replace_method"] = "auto"
    kwargs["replace_with_kernel_inject"] = False
    kwargs["injection_policy"] = {InferenceModel: ("l1")}
    _parallel_sdaas = [torch.device("sdaa")] * device_count

    trainer = Trainer(
        accelerator="sdaa",
        devices=device_count,
        strategy=SDAADeepSpeedStrategy(
            parallel_devices=_parallel_sdaas, **kwargs),
        plugins=[SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")],
        use_distributed_sampler=False,
    )
    preds = trainer.predict(model)
    expected = torch.tensor([32768.0, 32768.0])
    assert torch.allclose(
        preds[0].detach().to(torch.float), expected
    ), f"incorrect result value {preds}, expected {expected}"


@pytest.mark.standalone()
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float,
    ],
)
def test_lightning_deepspeed_inference_params(device_count, dtype):
    model = InferenceModel()
    _parallel_sdaas = [torch.device("sdaa")] * device_count

    trainer = Trainer(
        accelerator="sdaa",
        devices=device_count,
        strategy=SDAADeepSpeedStrategy(
            parallel_devices=_parallel_sdaas,
            tensor_parallel={"tp_size": device_count},
            dtype=dtype,
            replace_with_kernel_inject=False,
        ),
        plugins=[SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")],
        use_distributed_sampler=False,
    )
    preds = trainer.predict(model)
    expected = torch.tensor([32768.0, 32768.0])
    assert torch.allclose(
        preds[0].detach().to(torch.float), expected
    ), f"incorrect result value {preds}, expected {expected}"


@pytest.mark.standalone()
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float,
    ],
)
def test_lightning_deepspeed_inference_config(device_count, dtype):
    model = InferenceModel()
    _parallel_sdaas = [torch.device("sdaa")] * device_count

    _config = {
        "replace_with_kernel_inject": False,
        "tensor_parallel": {"tp_size": device_count},
        "dtype": dtype,
        "enable_cuda_graph": False,
    }

    trainer = Trainer(
        accelerator="sdaa",
        devices=device_count,
        strategy=SDAADeepSpeedStrategy(
            parallel_devices=_parallel_sdaas,
            config=_config,
        ),
        plugins=[SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed")],
        use_distributed_sampler=False,
    )
    preds = trainer.predict(model)
    expected = torch.tensor([32768.0, 32768.0])
    assert torch.allclose(
        preds[0].detach().to(torch.float), expected
    ), f"incorrect result value {preds}, expected {expected}"


@pytest.mark.standalone()
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_sdaa_deepspeed_training_accuracy(tmpdir, device_count, stage):
    """Test compare training accuracy between fp16 precision for deepspeed."""

    class TestModel(BoringModel):
        """Test model."""

        def __init__(self):
            """Init."""
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def training_step(self, batch, batch_idx):
            """Training step."""
            loss = super().training_step(batch, batch_idx)
            self.log("train_loss", loss.get("loss"),
                     prog_bar=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            """Validation step."""
            loss = super().validation_step(batch, batch_idx)
            self.log("val_loss", loss.get("x"), prog_bar=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            """Configure optimizer."""
            from torch.optim.adamw import AdamW

            return AdamW(self.parameters())

    def run_training(tmpdir, model, plugin, strategy):
        """Runs a model and returns loss."""
        trainer = Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=True,
            accelerator="sdaa",
            devices=device_count,
            strategy=strategy,
            plugins=plugin,
        )
        trainer.fit(model)
        return trainer.callback_metrics["val_loss"], trainer.callback_metrics["train_loss"]

    precision_list = [
        "32-true",
        "fp16-mixed",
        "16-mixed",
    ]

    loss_list = []

    for precision in precision_list:
        seed_everything(42)
        model = TestModel()
        _plugin = SDAADeepSpeedPrecisionPlugin(precision=precision)
        _strategy = SDAADeepSpeedStrategy(stage=stage, parallel_devices=[
            torch.device("sdaa")] * device_count)
        loss_list.append(run_training(tmpdir, model, _plugin, _strategy))

    assert torch.allclose(torch.tensor(loss_list[0][1]), torch.tensor(
        loss_list[1][1]), rtol=1e-2, atol=1e-2)


@pytest.mark.standalone()
@pytest.mark.standalone_only()
def test_sdaa_deepspeed_fp16_inference_accuracy(tmpdir, device_count):
    """Test maintain fp16 test loss used in fp8 inference accuracy test using deepspeed."""

    class TestModel(BoringModel):
        """Test model."""

        def test_step(self, batch, batch_idx):
            """Training step."""
            loss = super().test_step(batch, batch_idx)
            self.log("test_loss", loss.get("y"), prog_bar=True, sync_dist=True)
            return loss

    seed_everything(42)
    model = TestModel()
    dm = BoringDataModule()

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="sdaa",
        devices=device_count,
        strategy=SDAADeepSpeedStrategy(),
        plugins=SDAADeepSpeedPrecisionPlugin(precision="fp16-mixed"),
        fast_dev_run=True,
    )
    trainer.test(model, dm)

    fp16_test_loss = trainer.callback_metrics["test_loss"]
    fp16_loss = torch.tensor(0.6641)
    if device_count == 2:
        fp16_loss = torch.tensor(1.2734)
    assert torch.allclose(fp16_test_loss, fp16_loss, rtol=1e-4, atol=1e-4)
