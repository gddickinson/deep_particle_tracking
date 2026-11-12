"""Method facts for the web app, read from the source with `ast` (no PyTorch import needed).

Network classes and loss functions: name, first docstring paragraph and __init__ defaults, from the modules
models/network.py and models/losses.py re-export (listed in their __all__ order); the create_model names; the
trainer's default loss.
"""
import ast
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _doc(node) -> str:
    d = ast.get_docstring(node) or ''
    return ' '.join(d.split('\n\n')[0].split())


def _constants(tree) -> dict:
    """Module-level NAME = <literal> assignments (e.g. DEFAULT_LAMBDA_TRACK), so defaults that name them resolve."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _defaults(fn, constants=None) -> dict:
    args = fn.args.args[1:]
    defs = fn.args.defaults
    out = {}
    for a, d in zip(args[len(args) - len(defs):], defs):
        try:
            out[a.arg] = ast.literal_eval(d)
        except ValueError:
            name = ast.unparse(d)
            out[a.arg] = (constants or {}).get(name, name)
    return out


def _classes(path: Path, base: str):
    tree = ast.parse(path.read_text())
    constants = _constants(tree)
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(ast.unparse(b) == base for b in node.bases):
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
            out.append({'name': node.name, 'doc': _doc(node), 'defaults': _defaults(init, constants) if init else {},
                        'line': node.lineno, 'file': str(path.relative_to(ROOT))})
    return out, tree


def _package_classes(path: Path, base: str):
    """Classes defined in `path` and in the models.* modules it imports from, ordered as in its __all__."""
    tree = ast.parse(path.read_text())
    files = [path]
    order = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('models.'):
            files.append(ROOT / (node.module.replace('.', '/') + '.py'))
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == '__all__' for t in node.targets):
            order = ast.literal_eval(node.value)
    out = [c for f in files for c in _classes(f, base)[0]]
    out.sort(key=lambda c: order.index(c['name']) if c['name'] in order else len(order))
    return out, tree


@lru_cache(maxsize=1)
def method_info() -> dict:
    nets, tree = _package_classes(ROOT / 'models' / 'network.py', 'nn.Module')
    factory = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'create_model':
            for sub in ast.walk(node):
                if isinstance(sub, ast.If) and isinstance(sub.test, ast.Compare):
                    key = ast.literal_eval(sub.test.comparators[0])
                    call = sub.body[0].value
                    factory[key] = call.func.id
    losses, loss_tree = _package_classes(ROOT / 'models' / 'losses.py', 'nn.Module')
    consts = _constants(loss_tree)
    return {'networks': nets, 'create_model': factory, 'losses': losses,
            'trainer_loss': 'CombinedLoss (FocalLoss alpha 0.25, gamma 2 for localisation + EmbeddingLoss margin 1 '
                            'for tracking: one embedding map per input frame, same / different ground-truth track IDs '
                            'pulled together / pushed apart across the window\'s frames, squared distances averaged '
                            f'over embedding channels; tracking weight {consts["DEFAULT_LAMBDA_TRACK"]}, ramped up from 0 '
                            f'over the first {consts["DEFAULT_TRACK_WARMUP_STEPS"]} optimiser steps), as set in training/trainer.py and models/losses.py'}
