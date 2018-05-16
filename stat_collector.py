from collections import namedtuple, defaultdict, OrderedDict

Scalar = namedtuple("Scalar", field_names=["tick", "value"])

class StatCollector():

    def __init__(self):
        self._tick = 0
        self._ranges = defaultdict(lambda : [])
        self._scalars = defaultdict(OrderedDict)

    def tick(self):
        self._tick += 1

    def scalar(self, name, value):
        scalars = self._scalars[name]
        # s = Scalar(self._tick, value)
        scalars[self._tick] = value
        self._scalars[name] = scalars

        r = self._ranges[name]
        if len(r) == 0:
            r = [value, value]
        else:
            r[0] = min(r[0], value)
            r[1] = max(r[1], value)
        self._ranges[name] = r

    def get_history(self, name):
        scalars = self._scalars[name]
        scalars = scalars.values()
        return scalars

    def get_range(self, name):
        return self._ranges[name]
