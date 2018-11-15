class Setup:
    def __init__(self, down, mid, up, desc):
        self.down = down
        self.mid = mid
        self.up = up
        self.desc = desc

basic = Setup(
    down=[(3, ),(2, 5, 2), (5, 7, 5)],
    mid=(10, 14, 10),
    up=[(5, 7, 5), (2, 5, 2), (1, )],
    desc='basic benchmark'
)

many_low_order = Setup(
    down=[(3, ),(5, 2, 2), (5, 7, 5)],
    mid=(10, 14, 10),
    up=[(5, 7, 5), (2, 5, 2), (1, )],
    desc='more low order features in first layer'
)

many_high_order = Setup(
    down=[(3, ),(2, 3, 4), (5, 7, 5)],
    mid=(10, 14, 10),
    up=[(5, 7, 5), (2, 5, 2), (1, )],
    desc='more high order features in first layer'
)

mid_block_to_low_order = Setup(
    down=[(3, ),(7, 12, 7), (5, 7, 5)],
    mid=(5, 7, 5),
    up=[(5, 7, 5), (2, 5, 2), (1, )],
    desc=('we save half the parameters from `mid` and move them '
          'to low levels of the pyramid')
)

up_heavy = Setup(
    down=[(3, ),(2, 5, 2), (5, 7, 5)],
    mid=(5, 7, 5),
    up=[(7, 10, 7), (5, 9, 5), (1, )],
    desc=('we save half the parameters from `mid` and move them '
          'to the up path')
)

down_heavy = Setup(
    down=[(3, ), (5, 9, 5), (7, 10, 7)],
    mid=(5, 7, 5),
    up=[(5, 7, 5), (2, 5, 2), (1, )],
    desc=('we save half the parameters from `mid` and move them '
          'to the down path')
)

setups = [many_low_order, many_high_order, mid_block_to_low_order, up_heavy, down_heavy]
