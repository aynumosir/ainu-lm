from .sllist import Sllist, SllistNode


def test_create_empty_list() -> None:
    sllist = Sllist[int]()
    assert sllist.head is None
    assert sllist.tail is None


def test_create_list() -> None:
    sllist = Sllist[int]()
    n = SllistNode(42)
    sllist.append(n)

    assert sllist.head is n
    assert sllist.tail is n


def test_create_list_multiple_items() -> None:
    sllist = Sllist[int]()
    n1 = SllistNode(42)
    n2 = SllistNode(43)
    n3 = SllistNode(44)
    sllist.append(n1)
    sllist.append(n2)
    sllist.append(n3)

    assert sllist.head is n1
    assert sllist.tail is n3

    assert n1.next is n2
    assert n2.next is n3
    assert n3.next is None


def test_iterate_over_list() -> None:
    sllist = Sllist[int]()
    n1 = SllistNode(42)
    n2 = SllistNode(43)
    n3 = SllistNode(44)
    sllist.append(n1)
    sllist.append(n2)
    sllist.append(n3)

    nodes = list(sllist.iter_nodes())
    assert nodes == [n1, n2, n3]


def test_appending_list() -> None:
    sllist = Sllist[int]()
    n1 = SllistNode(42)
    n2 = SllistNode(43)
    n3 = SllistNode(44)
    sllist.append(n1)
    sllist.append(n2)
    sllist.append(n3)

    n4 = SllistNode(45)
    sllist.append(n4)

    assert sllist.head is n1
    assert sllist.tail is n4

    assert n1.next is n2
    assert n2.next is n3
    assert n3.next is n4
    assert n4.next is None


def test_removing_node() -> None:
    sllist = Sllist[int]()
    n1 = SllistNode(42)
    n2 = SllistNode(43)
    n3 = SllistNode(44)
    sllist.append(n1)
    sllist.append(n2)
    sllist.append(n3)

    n2.remove()

    assert sllist.head is n1
    assert sllist.tail is n3

    assert n1.next is n3
    assert n3.next is None
