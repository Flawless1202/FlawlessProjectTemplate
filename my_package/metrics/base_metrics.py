from typing import Dict

from torch import Tensor


class BaseMetrics(object):
    """The base Metrics class."""

    @staticmethod
    def calculate(*args, **kwargs) -> Dict[str, Tensor]:
        """Calculate the metrics in a mini-batch.

        Returns:
            The dict of results.

        """
        raise NotImplementedError

    @staticmethod
    def summary(*args, **kwargs) -> Dict[str, Tensor]:
        """Calculate the metrics of the whole dataset.

        Args:
            outputs: The list of results of all mini-batches.

        Returns:
            The dict of results.

        """
        raise NotImplementedError
