"""Method facts for the web app, read from the source with `ast` (no PyTorch import needed).

Network classes and loss functions: name, first docstring paragraph and __init__ defaults, from
models/network.py and models/losses.py; the create_model names; the trainer's default loss.
"""
import ast
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _doc(node) -> str:
    d = ast.get_docstring(node) or ''
    return ' '.join(d.split('\n\n')[0].split())


def _defaults(fn) -> dict:
    args = fn.args.args[1:]
    defs = fn.args.defaults
    out = {}
    for a, d in zip(args[len(args) - len(defs):], defs):
        try:
            out[a.arg] = ast.literal_eval(d)
        except ValueError:
            out[a.arg] = ast.unparse(d)
    return out


def _classes(path: Path, base: str):
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(ast.unparse(b) == base for b in node.bases):
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
            out.append({'name': node.name, 'doc': _doc(node), 'defaults': _defaults(init) if init else {},
                        'line': node.lineno})
    return out, tree


@lru_cache(maxsize=1)
def method_info() -> dict:
    nets, tree = _classes(ROOT / 'models' / 'network.py', 'nn.Module')
    factory = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'create_model':
            for sub in ast.walk(node):
                if isinstance(sub, ast.If) and isinstance(sub.test, ast.Compare):
                    key = ast.literal_eval(sub.test.comparators[0])
                    call = sub.body[0].value
                    factory[key] = call.func.id
    losses, _ = _classes(ROOT / 'models' / 'losses.py', 'nn.Module')
    return {'networks': nets, 'create_model': factory, 'losses': losses,
            'trainer_loss': 'CombinedLoss (FocalLoss alpha 0.25, gamma 2 for localisation + EmbeddingLoss margin 1 '
                            'for tracking), as set in training/trainer.py and models/losses.py'}
