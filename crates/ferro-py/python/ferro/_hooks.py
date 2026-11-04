"""Removable weak handles for ordered Python callback registries."""
import itertools
import weakref


class RemovableHandle:
    _ids = itertools.count()

    def __init__(self, registry, *, extra_dict=None):
        self.id = next(self._ids)
        extras = () if extra_dict is None else (extra_dict if isinstance(extra_dict, (tuple, list)) else (extra_dict,))
        self._registries = tuple(weakref.ref(value) for value in (registry, *extras))

    def remove(self):
        for reference in self._registries:
            registry = reference()
            if registry is not None:
                registry.pop(self.id, None)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.remove()

