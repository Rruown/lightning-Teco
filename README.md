# Lightning ⚡  Teco

## Installing Lighting Teco

To install Lightning Teco, run the following command:

```bash
python setup.py install
```

______________________________________________________________________

**NOTE**

Ensure pytorch-lightning is used when working with the plugin.
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
or use 'sdaa' afther imported lightning_teco
```Python
from lightning import Trainer
import lightning_teco
# Run on one SDAA with ddp (find_unused_parameters=False).
trainer = Trainer(accelerator='sdaa', devices=4, strategy='ddp_sdaa_find_unused_parameters_false')
```

The `devices=1` parameter with SDAAs enables the Teco accelerator for single card training using `SingleSDAAStrategy`.

The `devices>1` parameter with SDAAs enables the Teco accelerator for distributed training. It uses `SDAADDPStrategy` which is based on DDP strategy with the integration of Habana’s collective communication library (TCCL) to support scale-up within a node and scale-out across multiple nodes.

# Support Matrix

| **Product**         | **1.18.6**                                          |
| --------------------- | --------------------------------------------------- |
| PyTorch               | >= 2.0.0                                               |
| (PyTorch) Lightning\* | 1.8.6                                             |
| **Lightning Teco**  |  main branch                                       |

\* covers package [`pytorch-lightning`](https://pypi.org/project/pytorch-lightning/)
