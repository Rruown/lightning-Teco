# Lightning ⚡  Teco

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![PyPI Status](https://badge.fury.io/py/lightning-teco.svg)](https://badge.fury.io/py/lightning-teco)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-teco)](https://pypi.org/project/lightning-teco/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lightning-Teco)](https://pepy.tech/project/lightning-teco)
[![Deploy Docs](https://github.com/Lightning-AI/lightning-Teco/actions/workflows/docs-deploy.yml/badge.svg)](https://lightning-ai.github.io/lightning-Teco/)

[![General checks](https://github.com/Lightning-AI/lightning-teco/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-teco/actions/workflows/ci-checks.yml)
[![Build Status](https://dev.azure.com/Lightning-AI/compatibility/_apis/build/status/Lightning-AI.lightning-Teco?branchName=main)](https://dev.azure.com/Lightning-AI/compatibility/_build/latest?definitionId=45&branchName=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-Teco/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-Teco/main)

[Intel® Gaudi® AI Processor (SDAA)](https://teco.ai/) training processors are built on a heterogeneous architecture with a cluster of fully programmable Tensor Processing Cores (TPC) along with its associated development tools and libraries, and a configurable Matrix Math engine.

The TPC core is a VLIW SIMD processor with an instruction set and hardware tailored to serve training workloads efficiently.
The Gaudi memory architecture includes on-die SRAM and local memories in each TPC and,
Gaudi is the first DL training processor that has integrated RDMA over Converged Ethernet (RoCE v2) engines on-chip.

On the software side, the PyTorch Teco bridge interfaces between the framework and SynapseAI software stack to enable the execution of deep learning models on the Teco Gaudi device.

Gaudi provides a significant cost-effective benefit, allowing you to engage in more deep learning training while minimizing expenses.

For more information, check out [Gaudi Architecture](https://docs.teco.ai/en/latest/Gaudi_Overview/Gaudi_Overview.html) and [Gaudi Developer Docs](https://developer.teco.ai).

______________________________________________________________________

## Installing Lighting Teco

To install Lightning Teco, run the following command:

```bash
pip install -U lightning lightning-teco
```

______________________________________________________________________

**NOTE**

Ensure either of lightning or pytorch-lightning is used when working with the plugin.
Mixing strategies, plugins etc from both packages is not yet validated.

______________________________________________________________________

## Using PyTorch Lighting with SDAA

To enable PyTorch Lightning with SDAA accelerator, provide `accelerator=SDAAAccelerator()` parameter to the Trainer class.

```python
from lightning import Trainer
from lightning_teco.accelerator import SDAAAccelerator

# Run on one SDAA.
trainer = Trainer(accelerator=SDAAAccelerator(), devices=1)
# Run on multiple SDAAs.
trainer = Trainer(accelerator=SDAAAccelerator(), devices=8)
# Choose the number of devices automatically.
trainer = Trainer(accelerator=SDAAAccelerator(), devices="auto")
```

The `devices=1` parameter with SDAAs enables the Teco accelerator for single card training using `SingleSDAAStrategy`.

The `devices>1` parameter with SDAAs enables the Teco accelerator for distributed training. It uses `SDAADDPStrategy` which is based on DDP strategy with the integration of Habana’s collective communication library (TCCL) to support scale-up within a node and scale-out across multiple nodes.

# Support Matrix

| **SynapseAI**         | **1.18.0**                                          |
| --------------------- | --------------------------------------------------- |
| PyTorch               | 2.4.0                                               |
| (PyTorch) Lightning\* | 2.4.x                                               |
| **Lightning Teco**  | **1.7.0**                                           |
| DeepSpeed\*\*         | Forked from v0.14.4 of the official DeepSpeed repo. |

\* covers both packages [`lightning`](https://pypi.org/project/lightning/) and [`pytorch-lightning`](https://pypi.org/project/pytorch-lightning/)

For more information, check out [SDAA Support Matrix](https://docs.teco.ai/en/latest/Support_Matrix/Support_Matrix.html)
