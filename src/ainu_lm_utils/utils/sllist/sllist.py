from __future__ import annotations

from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Sllist(Generic[T]):
    head: Optional[SllistNode[T]]
    tail: Optional[SllistNode[T]]

    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def iter_nodes(self) -> SllistNodeIterator[T]:
        return SllistNodeIterator(self.head)

    def iter_values(self) -> SllistValueIterator[T]:
        return SllistValueIterator(self.head)

    def append(self, node: SllistNode[T]) -> None:
        if self.head is None or self.tail is None:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
        node.owner = self

    def remove(self, node: SllistNode[T]) -> None:
        if node.owner is not self:
            raise ValueError("Node does not belong to this list")
        if self.head is node:
            self.head = node.next
        else:
            for n in self.iter_nodes():
                if n.next is node:
                    n.next = node.next
                    break
        if self.tail is node:
            self.tail = None if self.head is node else n

    @classmethod
    def from_list(cls, values: list[T]) -> Sllist[T]:
        sllist = cls()
        for value in values:
            sllist.append(SllistNode(value))
        return sllist


class SllistNode(Generic[T]):
    value: T
    next: Optional[SllistNode[T]]
    owner: Optional[Sllist[T]]

    def __init__(self, value: T) -> None:
        self.value = value
        self.next = None

    def insert_after(self, node: SllistNode[T]) -> None:
        node.next = self.next
        self.next = node
        node.owner = self.owner

    def remove(self) -> None:
        if self.owner is None:
            raise ValueError("Node does not belong to any list")
        self.owner.remove(self)


class SllistNodeIterator(Generic[T]):
    def __init__(self, node: Optional[SllistNode[T]]) -> None:
        self.node = node

    def __iter__(self) -> SllistNodeIterator[T]:
        return self

    def __next__(self) -> SllistNode[T]:
        if self.node is None:
            raise StopIteration
        else:
            node = self.node
            self.node = self.node.next
            return node


class SllistValueIterator(Generic[T]):
    def __init__(self, node: Optional[SllistNode[T]]) -> None:
        self.node = node

    def __iter__(self) -> SllistValueIterator[T]:
        return self

    def __next__(self) -> T:
        if self.node is None:
            raise StopIteration
        else:
            value = self.node.value
            self.node = self.node.next
            return value
