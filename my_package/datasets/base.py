import os
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset

from .transforms import Compose


class BaseDataset(Dataset):
    """A base dataset wrapper of Dataset class in PyTorch.

    For dealing with a new dataset, we define a pipeline as follow:
    1. Preprocess the original data file and save each sample as a single '*.pt in '{processed_dir}'
       directory. (Need to implement the abstract method `self._process()`)
    2. Get the list of all data. (Need to implement the abstract method `self._get_data_list()`).
    3. Get a sample from the dataset.

    Args:
        root (str): The root directory of the dataset.
        task (str): The name of the subtask on the dataset.
        phase (str): Choose from ['train', 'val', 'test'].
        transforms (List[Callable]): a list of transform. Each transform is an lambda function with format
            'lambda data: dict(`variable name`=function(args, **args))'.
        in_memory (bool): if pre-load all the data into memory
        force_regen (bool): If force to regenerate the processed cache.
    """

    # The arguments required to be set.
    _fields = [
        "root",
        "task",
        "phase",
        "transforms",
        "in_memory",
        "force_regen",
    ]

    def __init__(self, *args, **kwargs):
        """The init function of the class.

        Each positional arguments and keyword arguments are set as an attribute of the class.
        """

        super().__init__()

        if len(args) > len(self._fields):
            raise TypeError(
                f"Expected {len(self._fields)} arguments, but got {len(args)} instead"
            )

        # Set the positional arguments
        for args_name, args_value in zip(self._fields, args):
            setattr(self, args_name, args_value)

        # Set the optional keyword arguments
        for kwargs_name, kwargs_value in kwargs.items():
            setattr(self, kwargs_name, kwargs_value)

        # Check all the required arguments are set.
        for required_attr in self._fields:
            if not hasattr(self, required_attr):
                raise TypeError(f"Excepted argument '{required_attr}' is not set!")

        # Compose transforms
        self.transforms = (
            Compose(self.transforms) if self.transforms is not None else None
        )

        # Generate the preprocessed cache
        if not os.path.exists(self.processed_dir) or self.force_regen:
            os.makedirs(self.processed_dir, exist_ok=True)
            self._process()

        if self.in_memory:
            self._data = [torch.load(file) for file in self._data_list]
        else:
            self._data = self._data_list

    @property
    def processed_dir(self):
        """The directory to save the processed cache data.

        There are two sub-directory called 'data' and 'statistics' in the processed directory, to save
        the data and statistic information respectively.
        """
        return os.path.join(self.root, "processed", f"{self.task}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item and patch statistic information to the item."""
        data = self._data[idx]

        return self.transforms(data) if self.transforms is not None else data

    def __len__(self) -> int:
        """The length of the dataset."""
        return len(self._data_list)

    def __repr__(self) -> str:
        """All the public attributes (not start with '_') are add to the __repr__."""
        repr_str = f"{self.__class__.__name__}(\n"
        repr_str += "".join(
            [f"    {k}={getattr(self, k)},\n" for k in dir(self) if k[0] != "_"]
        )
        repr_str += ")"

        return repr_str

    def _process(self) -> None:
        """Process the original files and save processed cache data.

        For each sample, the format is:
            {
                'x': (input data),
                'x_mask': (mask of input),
                'y': (label data),
                'y_mask': (mask of label),
                ...
            }
        and saved as a '*.pt' file in '{processed_dir}/data' directory.
        """
        raise NotImplementedError

    @property
    def _data_list(self) -> List[str]:
        """The list of all data."""
        raise NotImplementedError
