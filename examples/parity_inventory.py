"""Compare public API names in the installed packages; presence is not parity.

References are development dependencies only. Run after installing a wheel:
python examples/parity_inventory.py --output /tmp/ferro-api-inventory.json
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import ferro
import torch
import torch_geometric
import torch_geometric.nn
import torch_geometric.transforms


def public(module, predicate):
    return {name for name in dir(module) if not name.startswith('_') and predicate(getattr(module, name))}


def compare(reference, implementation, names):
    present = sorted(name for name in names if hasattr(implementation, name))
    absent = sorted(names.difference(present))
    return {'reference': reference, 'reference_names': len(names), 'present_names': len(present),
            'absent_names': len(absent), 'present': present, 'absent': absent}


def inventory():
    from torch.nn import Module
    areas = {}
    areas['nn'] = compare('torch.nn', ferro.nn, public(torch.nn, lambda value: inspect.isclass(value) and issubclass(value, Module)))
    areas['nn.functional'] = compare('torch.nn.functional', ferro.nn.functional,
                                    public(torch.nn.functional, lambda value: callable(value) and
                                           getattr(value, '__module__', '').startswith(('torch.nn.functional', 'torch._C'))))
    areas['optim'] = compare('torch.optim', ferro.optim,
                            public(torch.optim, lambda value: inspect.isclass(value) and issubclass(value, torch.optim.Optimizer)))
    areas['optim.lr_scheduler'] = compare('torch.optim.lr_scheduler', ferro.optim.lr_scheduler,
                                         public(torch.optim.lr_scheduler, lambda value: inspect.isclass(value) and
                                                getattr(value, '__module__', '') == 'torch.optim.lr_scheduler'))
    areas['Tensor'] = compare('torch.Tensor', ferro.Tensor,
                             public(torch.Tensor, lambda value: callable(value)))
    areas['linalg'] = compare('torch.linalg', ferro.linalg, public(torch.linalg, lambda value: callable(value) and
                                                                getattr(value, '__module__', '').startswith('torch')))
    areas['graph.nn'] = compare('torch_geometric.nn', ferro.graph,
                               public(torch_geometric.nn, lambda value: callable(value) and
                                      getattr(value, '__module__', '').startswith('torch_geometric.nn')))
    areas['graph.transforms'] = compare('torch_geometric.transforms', ferro.transforms,
                                       public(torch_geometric.transforms, lambda value: inspect.isclass(value) and
                                              getattr(value, '__module__', '').startswith('torch_geometric.transforms')))
    root = Path(ferro.__file__).parent
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.rglob('*')) if path.suffix in ('.py', '.so', '.pyd')}
    return {'measurement': 'API name presence only; signatures, behavior, gradients, devices and performance are not inferred',
            'versions': {'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__},
            'ferro_installation': str(root), 'installed_files_sha256': sources, 'areas': areas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    result = inventory()
    if options.output is not None:
        options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(result['measurement'])
    for name, area in result['areas'].items():
        print(f"{name}: {area['present_names']} names present, {area['absent_names']} absent")
    if options.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
