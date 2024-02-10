"""
Contains Entity and OrderedEntity classes to represent entities and serialize them into strings for use in prompts.
"""

from collections import OrderedDict
import re
import numpy as np
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

    def __hash__(self) -> int:
        """
        Calculates the hash value of the entity based on its ID.

        Returns:
            int: The hash value of the entity.
        """
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
        """
        Converts the entity object to a string representation in the format required by Ditto.

        Returns:
            str: The string representation of the entity.
        """
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
        """
        Calculate the hash value of the entity based on its ID.

        Returns:
            int: The hash value of the entity.
        """
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

    def embed_values_k(self, k: int, random_order: bool) -> str:
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
        # choose k random source attributes from keys
        if k > len(self.keys()):
            raise ValueError("k must be less than or equal to the number of attributes")
        src_keys = random.sample(list(self.keys()), k)
        for key in src_keys:
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

    def embed_values_freq(self, freq: float, random_order: bool) -> str:
        n_attr = len(self.keys())
        k = int(round(freq * n_attr))
        return self.embed_values_k(k, random_order)

    def embed_values_p(self, p_move: float, random_order: bool) -> str:
        """
        Get a string representation of the values in the OrderedEntity with a chance of data corruption leading
         to embedded values.
        For each attribute we randomly move its value to another random attribute with p_move probability.

        Args:
            p_move (float, optional): The probability of embedding an attribute value under another attribute.

        Returns:
            str: String representation of values.
        """
        n_attr = len(self.keys())
        k = np.random.binomial(n_attr, p_move)
        return self.embed_values_k(k, random_order)

    def misfield_str_freq(self, freq: float, random_order: bool) -> str:
        """
        Get a string representation of the values in the OrderedEntity with a chance of data corruption leading
         to misfielded values.
        For each attribute we randomly move its value to another random attribute with p_move probability.

        Args:
            p_move (float, optional): The probability of embedding an attribute value under another attribute.

        Returns:
            str: String representation of values.
        """
        n_attr = len(self.keys())
        k = int(round(freq * n_attr))
        misfielded = self.misfield(k)
        return ffm_wrangle_string(misfielded, random_order)

    def misfield_str(self, k: int, random_order: bool) -> str:
        """
        Get a string representation of the values in the OrderedEntity with a chance of data corruption leading
         to misfielded values.
        For each attribute we randomly move its value to another random attribute with p_move probability.

        Args:
            p_move (float, optional): The probability of embedding an attribute value under another attribute.

        Returns:
            str: String representation of values.
        """
        if k <= 1:
            raise ValueError("k must be greater than 1")
        if k > len(self.keys()):
            raise ValueError("k must be less than or equal to the number of attributes")
        misfielded = self.misfield(k)
        return ffm_wrangle_string(misfielded, random_order)

    def misfield(self, k: int) -> OrderedDict:
        original_kvs = OrderedDict(self.items())
        keys = list(original_kvs.keys())
        mapping = {}
        misfield_src_attributes = random.sample(keys, k)
        misfield_target_attributes = set(misfield_src_attributes.copy())
        for i, src in enumerate(misfield_src_attributes):
            # avoid situation in which last remaining src attribute must be mapped to itself
            if (
                len(misfield_target_attributes) == 2
                and misfield_src_attributes[i + 1] in misfield_target_attributes
            ):
                mapping[src] = misfield_src_attributes[i + 1]
                misfield_target_attributes.remove(misfield_src_attributes[i + 1])
                mapping[misfield_src_attributes[i + 1]] = (
                    misfield_target_attributes.pop()
                )
                break
            target = random.choice(list(misfield_target_attributes - {src}))
            misfield_target_attributes.remove(target)
            mapping[src] = target
        kvs = original_kvs.copy()
        for src, target in mapping.items():
            kvs[target] = original_kvs[src]
        return kvs

    def embed_values_avg_freq(self, avg_freq: float, random_order: bool) -> str:
        n_attr = len(self.keys())
        p_move = avg_freq * n_attr
        return self.embed_values_p(p_move, random_order)

    def to_ditto_str(self) -> str:
        """
        Converts the entity object to a string representation in the format required by Ditto.

        Returns:
            str: The string representation of the entity.
        """
        col_vals = (f"COL {key} VAL {self[key].lower()}" for (key, val) in self.items())
        return " ".join(col_vals)


def ffm_wrangle_string(kvs: OrderedDict, random_order: bool = False) -> str:
    """
    Get a string representation of the values according to paper
    "Can Foundation Models Wrangle Your Data?".

    Returns:
        str: String representation of values in the format attr1 : val1 . . . attr𝑚 : val𝑚
    """
    its = list(kvs.items())
    if random_order:
        random.shuffle(its)
    return " ".join(f"{name}:{value}" for name, value in kvs.items())
