from collections import OrderedDict
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


class Entity(dict):
    def __init__(self, id: int, uri: str, kv: Dict, order: Optional[List[str]] = None):
        """
        Initialize an Entity object.

        Args:
            id (int): The ID of the entity.
            uri (str): The URI of the entity.
            kv (dict): The key-value pairs of the entity.
            order (list, optional): The order of the keys in the entity. Defaults to None.

        Raises:
            ValueError: If the order list is provided and does not contain all keys.
        """
        super().__init__(kv)
        self.id = id
        self.uri = uri
        self.order = order
        if order and len(order) != len(self.keys()):
            raise ValueError("order must contain all keys")

    def __hash__(self):
        return hash(self.id)

    def to_ditto_str(self) -> str:
        order = self.order
        if order is None:
            order = sorted(self.keys())
        col_vals = (f"COL {key} VAL {self[key].lower()}" for key in order)
        return " ".join(col_vals)


def to_str(e: Entity, include_keys=False) -> str:
    """
    Convert an Entity object to a string representation.

    Args:
        e (Entity): The Entity object.
        include_keys (bool, optional): Include keys in the string representation. Defaults to False.

    Returns:
        str: String representation of the Entity.
    """
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
    """
    Extract tokens from the string representation of an Entity.

    Args:
        e (Entity): The Entity object.
        include_keys (bool, optional): Include keys in the token extraction. Defaults to False.
        return_set (bool, optional): Return a set of tokens. Defaults to True.

    Returns:
        set or list: Set or list of extracted tokens.
    """
    vals = to_str(e, include_keys)
    toks = filter(None, re.split("[\\W_]", vals))
    return set(toks) if return_set else list(toks)


class OrderedEntity(OrderedDict):
    """
    Represents an ordered entity with an ID, URI, and ordered key-value pairs.

    Args:
        id (int): The ID of the entity.
        uri (str, optional): The URI of the entity.
        attributes (Iterable[Tuple[str, str]]): Iterable of key-value pairs.

    Raises:
        TypeError: If attributes is not an iterable of tuples.
    """

    def __init__(
        self, id: int, uri: Optional[str], attributes: Iterable[Tuple[str, str]]
    ):
        super().__init__(attributes)
        self.id = id
        self.uri = uri

    def __hash__(self) -> int:
        return hash(self.id)

    def tokens(self, include_keys=False, return_set=True) -> Set[str] | List[str]:
        """
        Extract tokens from the string representation of an OrderedEntity.

        Args:
            include_keys (bool, optional): Include keys in the token extraction. Defaults to False.
            return_set (bool, optional): Return a set of tokens. Defaults to True.

        Returns:
            set or list: Set or list of extracted tokens.
        """
        if include_keys:
            it = (f"{k} {v}" for (k, v) in self.items())
        else:
            it = self.values()
        vals = " ".join(it).lower()
        toks = filter(None, re.split("[\\W_]", vals))
        return set(toks) if return_set else list(toks)

    def value_string(self) -> str:
        """
        Get a string representation of the values in the OrderedEntity.

        Returns:
            str: String representation of values.
        """
        return " ".join(str(value) for value in self.values())
