from collections import OrderedDict
import re
from typing import Iterable, List, Optional, Set, Tuple


class Entity(dict):
    def __init__(self, id, uri, kv, order=None):
        super().__init__(kv)
        self.id = id
        self.uri = uri
        self.order = order
        if order and len(order) != len(self.keys()):
            raise ValueError("order must contain all keys")

    # make hashable
    def __hash__(self):
        return hash(self.id)


class OrderedEntity(OrderedDict):
    def __init__(
        self, id: int, uri: Optional[str], attributes: Iterable[Tuple[str, str]]
    ):
        super().__init__(attributes)
        self.id = id
        self.uri = uri

    # make hashable
    def __hash__(self) -> int:
        return hash(self.id)

    def tokens(self, include_keys=False, return_set=True):
        if include_keys:
            it = (f"{k} {v}" for (k, v) in self.items())
        else:
            it = self.values()
        vals = " ".join(it).lower()
        toks = filter(None, re.split("[\\W_]", vals))
        return set(toks) if return_set else list(toks)

    def value_string(self) -> str:
        return " ".join(str(value) for value in self.values())


def to_str(e: Entity, include_keys=False) -> str:
    order = e.order
    if order is None:
        order = sorted(e.keys())
    if include_keys:
        it = (f"{key} {e[key]}" for key in order)
    else:
        it = (e[key] for key in order)
    return " ".join(it).lower()


def tokens(
    e: Entity,
    include_keys=False,
    return_set=True,
) -> Set[str] | List[str]:
    vals = to_str(e, include_keys)
    toks = filter(None, re.split("[\\W_]", vals))
    return set(toks) if return_set else list(toks)
