## Long-tailed distribution

Run `supervised.py` with the following arguments.
```
optional arguments:
  -h, --help            show this help message and exit
  --reweighting
                        Reweighting techniques. Select one from loss.py.
  --resampling
                        Resampling techniques. Select one from [balance,
                        square, progressive].
  --two_stage
                        Deferred re-balancing by resampling (drs) and by re-weighting (drw).
  --two_stage_start_epoch
                        The epoch when we start second stage training.
```
### Reweighting with one or two stages
Specify one of the loss functions in `loss.py`.
The `loss.py` is adapted from https://github.com/zhangyongshun/BagofTricks-LT/blob/main/lib/loss/loss.py.

For those loss functions need hyperparameters (e.g., gamma in focal loss), we can only edit these hyperparameters in `supervised.py`. They will be passed as arguments in the future.

### Resampling
Three resampling techniques (balance, square, prgoressive) are implemented in `wm_811k_dataset.py`.
Specify one of them if needed.

Two stage training with resampling is not supported.

### Mixup training.
Different from the common object recognition tasks, mixup is not an appropriate data augmention for the wafer dataset.

## Utility

The file `resnet.py` is adapted from https://github.com/akamaster/pytorch_resnet_cifar10.

The files `generate_data.py` and `wm_811k_dataset.py` are how we preprocess and import the dataset.
