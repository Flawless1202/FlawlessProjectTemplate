from typing import Callable, Dict, List, Union

from torch import Tensor

from ..builder import build_transforms


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[Union[Callable, Dict]]):
        self.transforms = []
        for t in transforms:
            if isinstance(t, Callable):
                self.transforms.append(t)
            elif isinstance(t, Dict):
                self.transforms.append(build_transforms(t))
            else:
                raise TypeError(
                    f"transform must be callable or dict, but got {type(t)}"
                )

    def __call__(self, data: Dict[str, Tensor]):
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        args = ["    {},".format(transform) for transform in self.transforms]
        return "{}([\n{}\n])".format(self.__class__.__name__, "\n".join(args))
