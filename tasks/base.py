from pathlib import Path


def read_txt_triples(path: Path):
    r"""
    Return format: list of 3-tuples containing (h, r, t)
    """
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = [int(token) for token in line.strip().split("\t")]
            assert len(tokens) == 3
            triples.append(tokens)
    return triples


def read_pkl(path: Path):
    with open(path, 'rb') as f:
        from pickle import load
        return load(f)
