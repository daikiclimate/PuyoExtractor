import numpy as np


def return_transform():
    return Composer([Flip(), PuyoColorExchange()])


class Composer(object):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, x):
        for t in self._transforms:
            x = t(x)
        return x


class Flip(object):
    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, fields):
        p = np.random.rand()
        if p > self._p:
            for i, f in enumerate(fields):
                f = np.fliplr(f).copy()
                fields[i] = f
            return fields

        else:
            return fields


class PuyoColorExchange(object):
    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, fields):
        p = np.random.rand()
        puyo_list = np.random.permutation([0, 1, 2, 3, 4])
        if p > self._p:
            for i, f in enumerate(fields):
                f = f.reshape(-1)
                src_field = f.copy()
                src_field = src_field.reshape(-1)
                for p_index, x in enumerate(puyo_list):
                    f[src_field == p_index] = x
                f = f.reshape(12, 6)

            return fields

        else:
            return fields


if __name__ == "__main__":
    sample_img = [np.random.randint(0, 7, (12, 6)) for _ in range(3)]
    tranform = Composer([Flip(), PuyoColorExchange()])
    x = tranform(sample_img)
    print(x)
