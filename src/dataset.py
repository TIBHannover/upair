import os

from typing import Dict, Callable, Any, Union, List
from pathlib import Path

from torch.utils.data import IterableDataset

from braceexpand import braceexpand
import webdataset as wds

SEP = '__SEP__'

class PatentDataset(IterableDataset):
    """PatentDataset"""

    def __init__(
        self,
        root: Path,
        split: str = 'index',
        shardshuffle: Union[bool, int] = False,
        shuffle: bool = True,
        buffer_size: int = 1000,
        figure_types: Union[str, List[str]] = 'drawing',
        reference_dirs: str = None,
        outputs: Dict[str, Callable[[bytes], Any]] = None,
    ):
        super().__init__()

        self.shardshuffle = shardshuffle

        self.shuffle = shuffle
        self.buffer_size = buffer_size

        if isinstance(figure_types, list):
            figure_types = '{' + ','.join(figure_types) + '}'

        if not reference_dirs:
            reference_dirs = '{w-ref,wo-ref}'

        if split == 'query':
            shards = 'shard-{000000..000003}.tar'
        elif split == 'train':
            shards = 'shard-{000000..000129}.tar'
        else:
            shards = 'shard-{000000..001955}.tar'

        self.shards = braceexpand(
            os.path.join(root, split, reference_dirs, figure_types, shards)
        )

        default_outputs = {
            '__key__': lambda x: x,
        }

        self.final_outputs = (
            {**default_outputs, **outputs} if outputs else default_outputs
        )

    def __iter__(self):

        for shard in self.shards:

            dataset = (
                wds.WebDataset(
                    shard,
                    shardshuffle=self.shardshuffle
                )
                .shuffle(self.buffer_size if self.shuffle else 0)
                .to_tuple(*self.final_outputs.keys())
                .map_tuple(
                    *[
                        self.final_outputs[key]
                        for key in self.final_outputs.keys()
                    ]
                )
            )

            for sample in dataset:
                yield sample
