# FlawlessProjectTemplate
A custom deep learning project template based on PyTorch and PyTorch-Lightning.

## Requirements

- torch >= 1.8.0
- pytorch-lightning

## Usage

1. Rename the `my_package` directory and replace the package name `my_package` in `main.py` correspondingly.
2. Inherit `BaseDataset` in `my_package/datasets/base.py` and implement your Dataset class.
3. Inherit `BaseMetrics` in `my_package/metrics/base_metrics.py` and implement your custom Metrics class.
4. Implement your model and put all the components in `my_package/models`
5. Use registry to organize all the above parts.
6. Follow the example configure file `configs/example_cfg.py` to make your custom configure file.
7. Train your model by the following command:
    ```shell
   python main.py ./configs/example_cfg.py --train
    ```
8. Test your model by the following command:
    ```shell
   python main.py ./configs/example_cfg.py --test
    ```
    
## Note
1. Unit tests are really important.
2. Always make your code clean and readable-friendly.