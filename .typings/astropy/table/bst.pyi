from _typeshed import Incomplete

__all__ = ['BST']

class MaxValue:
    """
    Represents an infinite value for purposes
    of tuple comparison.
    """
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __repr__(self) -> str: ...
    __str__ = __repr__

class MinValue:
    """
    The opposite of MaxValue, i.e. a representation of
    negative infinity.
    """
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __repr__(self) -> str: ...
    __str__ = __repr__

class Epsilon:
    '''
    Represents the "next largest" version of a given value,
    so that for all valid comparisons we have
    x < y < Epsilon(y) < z whenever x < y < z and x, z are
    not Epsilon objects.

    Parameters
    ----------
    val : object
        Original value
    '''
    __slots__: Incomplete
    val: Incomplete
    def __init__(self, val) -> None: ...
    def __lt__(self, other): ...
    def __gt__(self, other): ...
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class Node:
    """
    An element in a binary search tree, containing
    a key, data, and references to children nodes and
    a parent node.

    Parameters
    ----------
    key : tuple
        Node key
    data : list or int
        Node data
    """
    __lt__: Incomplete
    __le__: Incomplete
    __eq__: Incomplete
    __ge__: Incomplete
    __gt__: Incomplete
    __ne__: Incomplete
    __slots__: Incomplete
    key: Incomplete
    data: Incomplete
    left: Incomplete
    right: Incomplete
    def __init__(self, key, data) -> None: ...
    def replace(self, child, new_child) -> None:
        """
        Replace this node's child with a new child.
        """
    def remove(self, child) -> None:
        """
        Remove the given child.
        """
    def set(self, other) -> None:
        """
        Copy the given node.
        """
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class BST:
    """
    A basic binary search tree in pure Python, used
    as an engine for indexing.

    Parameters
    ----------
    data : Table
        Sorted columns of the original table
    row_index : Column object
        Row numbers corresponding to data columns
    unique : bool
        Whether the values of the index must be unique.
        Defaults to False.
    """
    NodeClass = Node
    root: Incomplete
    size: int
    unique: Incomplete
    def __init__(self, data, row_index, unique: bool = False) -> None: ...
    def add(self, key, data: Incomplete | None = None) -> None:
        """
        Add a key, data pair.
        """
    def find(self, key):
        """
        Return all data values corresponding to a given key.

        Parameters
        ----------
        key : tuple
            Input key

        Returns
        -------
        data_vals : list
            List of rows corresponding to the input key
        """
    def find_node(self, key):
        """
        Find the node associated with the given key.
        """
    def shift_left(self, row) -> None:
        """
        Decrement all rows larger than the given row.
        """
    def shift_right(self, row) -> None:
        """
        Increment all rows greater than or equal to the given row.
        """
    def _find_recursive(self, key, node, parent): ...
    def traverse(self, order: str = 'inorder'):
        '''
        Return nodes of the BST in the given order.

        Parameters
        ----------
        order : str
            The order in which to recursively search the BST.
            Possible values are:
            "preorder": current node, left subtree, right subtree
            "inorder": left subtree, current node, right subtree
            "postorder": left subtree, right subtree, current node
        '''
    def items(self):
        """
        Return BST items in order as (key, data) pairs.
        """
    def sort(self) -> None:
        """
        Make row order align with key order.
        """
    def sorted_data(self):
        """
        Return BST rows sorted by key values.
        """
    def _preorder(self, node, lst): ...
    def _inorder(self, node, lst): ...
    def _postorder(self, node, lst): ...
    def _substitute(self, node, parent, new_node) -> None: ...
    def remove(self, key, data: Incomplete | None = None):
        """
        Remove data corresponding to the given key.

        Parameters
        ----------
        key : tuple
            The key to remove
        data : int or None
            If None, remove the node corresponding to the given key.
            If not None, remove only the given data value from the node.

        Returns
        -------
        successful : bool
            True if removal was successful, false otherwise
        """
    def is_valid(self):
        """
        Returns whether this is a valid BST.
        """
    def _is_valid(self, node): ...
    def range(self, lower, upper, bounds=(True, True)):
        """
        Return all nodes with keys in the given range.

        Parameters
        ----------
        lower : tuple
            Lower bound
        upper : tuple
            Upper bound
        bounds : (2,) tuple of bool
            Indicates whether the search should be inclusive or
            exclusive with respect to the endpoints. The first
            argument corresponds to an inclusive lower bound,
            and the second argument to an inclusive upper bound.
        """
    def range_nodes(self, lower, upper, bounds=(True, True)):
        """
        Return nodes in the given range.
        """
    def same_prefix(self, val):
        """
        Assuming the given value has smaller length than keys, return
        nodes whose keys have this value as a prefix.
        """
    def _range(self, lower, upper, op1, op2, node, lst): ...
    def _same_prefix(self, val, node, lst): ...
    def __repr__(self) -> str: ...
    def _print(self, node, level): ...
    @property
    def height(self):
        """
        Return the BST height.
        """
    def _height(self, node): ...
    def replace_rows(self, row_map) -> None:
        """
        Replace all rows with the values they map to in the
        given dictionary. Any rows not present as keys in
        the dictionary will have their nodes deleted.

        Parameters
        ----------
        row_map : dict
            Mapping of row numbers to new row numbers
        """
