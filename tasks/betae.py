r"""
Read data from data file formats provided by
https://github.com/snap-stanford/KGReasoning
Data can be downloaded from
http://snap.stanford.edu/betae/KG_data.zip
"""
from pathlib import Path
from typing import Dict, Any

from .base import read_pkl, read_txt_triples

# https://github.com/snap-stanford/KGReasoning/blob/ec728497f083973edbe8b9e3c4066f416fc74250/main.py#L25-L41
query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }


class BetaEDataset:
    cache: Dict[Path, Any]

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.cache = dict()

    def get_file(self, rel_path):
        abs_path = self.data_path / rel_path
        if (v := self.cache.get(abs_path)) is not None:
            return v
        name = abs_path.name
        if name.endswith('.txt'):
            data = read_txt_triples(abs_path)
        elif name.endswith('-answers.pkl'):
            data = read_pkl(abs_path)
        elif name.endswith('-queries.pkl'):
            data = read_pkl(abs_path)
            data = {query_name_dict[k]: v for k, v in data.items()}
        elif name.endswith('.pkl') and '2' in name and 'id' in name:
            data = read_pkl(abs_path)
        else:
            raise f'Unrecognized file: {rel_path}'
        self.cache[abs_path] = data
        return data

    def calldata(self):
        all_files = ['train.txt', 'valid.txt', 'test.txt']
        return tuple(self.get_file(f) for f in all_files)
