import cv2
import numpy as np


def return_transform():
    train_transform = Composer(
        [
            Flip(),
            PuyoColorExchange(),
            # PuyoCounter(),
            PositionEncoder(),
        ]
    )
    test_transform = Composer(
        [
            # PuyoCounter(),
            PositionEncoder(),
        ]
    )
    return train_transform, test_transform


class Composer(object):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, x):
        for t in self._transforms:
            x = t(x)
        return x


class PuyoCounter(object):
    def __init__(self, kernel_size=3, padding=1):
        self._kernel_size = kernel_size
        self._padding = padding

    def __call__(self, fields):
        for i, f in enumerate(fields):
            pad_f = self.padding(f)
            count_field = self.count(pad_f.copy()).astype(float) / 10
            count_field25 = self.count2(f.copy()).astype(float) / 10
            f = np.concatenate([f, count_field])
            f = np.concatenate([f, count_field25])

            fields[i] = f
        return fields

    def padding(self, field, fill=9):
        z = np.full((1, 1, 6), fill)
        field = np.concatenate([z, field, z], axis=1)
        z = np.full((1, 14, 1), fill)
        field = np.concatenate([z, field, z], axis=2)
        return field

    def count(self, field, fill=9):
        count_field = np.zeros((1, 12, 6), dtype=np.uint8)
        for i in range(12):
            for j in range(6):
                xi, xj = i + 1, j + 1
                center = field[0, xi, xj]
                if center == 6:
                    count = 0
                else:
                    patch = field[0, xi - 1 : xi + 2, xj - 1 : xj + 2].reshape(-1)
                    count = sum(patch == center)
                count_field[0, i, j] = count
        return count_field

    def count2(self, field, fill=9):
        z = np.full((1, 1, 6), fill)
        field = np.concatenate([z, z, field, z, z], axis=1)
        z = np.full((1, 16, 1), fill)
        field = np.concatenate([z, z, field, z, z], axis=2)

        count_field = np.zeros((1, 12, 6), dtype=np.uint8)
        for i in range(12):
            for j in range(6):
                xi, xj = i + 2, j + 2
                center = field[0, xi, xj]
                if center == 6:
                    count = 0
                else:
                    patch = field[0, xi - 2 : xi + 3, xj - 2 : xj + 3].reshape(-1)
                    count = sum(patch == center)
                count_field[0, i, j] = count
        return count_field


class PositionEncoder(object):
    def __init__(self):
        self._column_position = np.array([[list(range(6)) for i in range(12)]]) / 5
        self._index_position = (
            np.array([[[i for _ in range(6)] for i in range(12)[::-1]]]) / 11
        )
        self._all_position = np.array([list(range(72))]).reshape(1, 12, 6) / 72

    def __call__(self, fields):
        for i, f in enumerate(fields):
            f = np.concatenate([f, self._column_position])
            f = np.concatenate([f, self._index_position])
            f = np.concatenate([f, self._all_position])
            fields[i] = f
        return fields


class Flip(object):
    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, fields):
        p = np.random.rand()
        if p > self._p:
            for i, f in enumerate(fields):
                f = np.flip(f, axis=2).copy()
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
                f = f.reshape(1, 12, 6)

            return fields

        else:
            return fields


if __name__ == "__main__":
    sample_img = [np.random.randint(0, 7, (12, 6)) for _ in range(3)]
    tranform = Composer([Flip(), PuyoColorExchange()])
    x = tranform(sample_img)
    print(x)
