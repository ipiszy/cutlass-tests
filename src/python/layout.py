from pycute.layout import Layout
from pycute.swizzle import Swizzle, ComposedLayout


def print_layout(layout: Layout) -> None:
    print(layout)
    shape = layout.layoutA.shape if isinstance(layout, ComposedLayout) else layout.shape
    assert len(shape) == 2
    width = 5

    print(" " * width, end="")
    for j in range(shape[1]):
        print(f" {j:>{width}}", end="")
    print()
    print("=" * width * shape[1])
    for i in range(shape[0]):
        print(f"{i:>{width-1}}|", end="")
        for j in range(shape[1]):
            print(f" {layout(i, j):>{width}}", end="")
        print()
    print()


layout0 = Layout((8, 64), (64, 1))
print_layout(layout0)

layout1 = ComposedLayout(Swizzle(0, 4, 3), 0, layout0)
print_layout(layout1)

layout2 = ComposedLayout(Swizzle(1, 4, 3), 0, layout0)
print_layout(layout2)

layout3 = ComposedLayout(Swizzle(2, 4, 3), 0, layout0)
print_layout(layout3)
