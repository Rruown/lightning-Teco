# Lightning ⚡  Teco

## Installing Lighting Teco

To install Lightning Teco, run the following command:

```bash
python setup.py install

# or 

pip install .
```

______________________________________________________________________

**NOTE**

Ensure pytorch-lightning is used when working with the plugin.
Mixing strategies, plugins etc from both packages is not yet validated.

______________________________________________________________________

## Using PyTorch Lightning with SDAA

This following map is registered at pytorch lightning AcceleratorRegistry and StrategyRegistry when `import lightning_teco`:
- "sdaa": SDAAAccelerator
- "ddp_sdaa_find_unused_parameters_false": SDAADDPStrategy(find_unused_parameters=False)
- "ddp_saa": SDAADDPStrategy
- "single_sdaa": SingleSDAAStrategy
- "sdaa_deepspeed": SDAADeepSpeedStrategy
- "sdaa_fsdp": SDAAFSDPStrategy

To enable PyTorch Lightning with SDAA accelerator provide `accelerator="sdaa"` parameter to the Trainer class.
The `devices=1` parameter with SDAAs enables the Teco accelerator for single card training using `single_sdaa`.
```Python
from lightning import Trainer
import lightning_teco
trainer = Trainer(accelerator='sdaa', devices=1, strategy='single_sdaa')
```
The `devices>1` parameter with SDAAs enables the Teco accelerator for distributed training. It uses `SDAADDPStrategy` which is based on DDP strategy with the integration of Tecorigin’s collective communication library (TCCL) to support scale-up within a node and scale-out across multiple nodes.
```Python
from lightning import Trainer
import lightning_teco
# Run on one SDAA with ddp (find_unused_parameters=False).
trainer = Trainer(accelerator='sdaa', devices=4, strategy='ddp_sdaa_find_unused_parameters_false')
```

or provide `accelerator=SDAAAccelerator()` parameter to the Trainer class.

```python
from lightning import Trainer
from lightning_teco.accelerator import SDAAAccelerator, SDAADDPStrategy

# Run on one SDAA.
trainer = Trainer(accelerator=SDAAAccelerator(), devices=1)
# Run on multiple SDAAs.
trainer = Trainer(accelerator=SDAAAccelerator(), devices=8, strategy=SDAADDPStrategy(find_unused_parameters=False))
# Choose the number of devices automatically 
trainer = Trainer(accelerator=SDAAAccelerator(), devices=4, strategy=SDAADDPStrategy())
```


# Support Matrix

| **Product**         | **2.5.0**                                          |
| --------------------- | --------------------------------------------------- |
| PyTorch               | >= 2.0.0                                               |
| (PyTorch) Lightning\* | 2.5.0                                             |
| **Lightning Teco**  |  main branch                                       |

\* covers package [`pytorch-lightning`](https://pypi.org/project/pytorch-lightning/)
