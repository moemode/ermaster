from collections import OrderedDict
import re
import random
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

    def get_order(self) -> List[str]:
        """
        Get the order of the keys in the Entity.

        Returns:
            list: The order of the keys.
        """
        order = self.order
        if order is None:
            order = sorted(self.keys())
        return order

    def to_ditto_str(self) -> str:
        order = self.get_order()
        col_vals = (f"COL {key} VAL {self[key].lower()}" for key in order)
        return " ".join(col_vals)

    def ffm_wrangle_string(self) -> str:
        """
        Get a string representation of the values according to paper
        "Can Foundation Models Wrangle Your Data?".

        Returns:
            str: String representation of values in the format attr1 : val1 . . . attr𝑚 : val𝑚
        """
        order = self.get_order()
        return " ".join(f"{name}:{self[name]}" for name in order)


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

    def ffm_wrangle_string(self, random_order=False) -> str:
        """
        Get a string representation of the values according to paper
        "Can Foundation Models Wrangle Your Data?".

        Returns:
            str: String representation of values in the format attr1 : val1 . . . attr𝑚 : val𝑚
        """
        its = list(self.items())
        if random_order:
            random.shuffle(its)
        return " ".join(f"{name}:{value}" for name, value in its)

    def embed_values(self, p_move: float, random_order: bool) -> str:
        """
        Get a string representation of the values in the OrderedEntity with a chance of data corruption leading
         to embedded values.
        For each attribute we randomly move its value to another random attribute with p_move probability.

        Args:
            p_move (float, optional): The probability of embedding an attribute value under another attribute.

        Returns:
            str: String representation of values.
        """
        corrupted_copy = OrderedEntity(self.id, self.uri, list(self.items()))
        moves = []
        for key, _ in self.items():
            if random.random() < p_move:
                # Choose a random attribute to embed the value
                allowed_target_keys = list(set(self.keys()) - {key})
                target_key = random.choice(allowed_target_keys)
                moves.append((key, target_key))
        for src, _ in moves:
            corrupted_copy[src] = ""
        for src, dst in moves:
            corrupted_copy[dst] += " " + self[src]
        # in corrupted copy remove attributes which are empty
        for key in list(corrupted_copy.keys()):
            if corrupted_copy[key] == "":
                del corrupted_copy[key]
        return corrupted_copy.ffm_wrangle_string(random_order)
