import os

import cv2
import numpy as np


def get_field_info(img):
    H = 490
    W = 255
    h_start = 107
    player1_field = img[h_start : h_start + H, 190 - 2 : 190 - 2 + W]
    player2_field = img[h_start : h_start + H, 840 : 840 + W]
    fields = []
    for field in [player1_field, player2_field]:
        init_field = np.zeros((12, 6), dtype=np.uint8)

        h_unit = H // 12
        w_unit = W // 6
        for h in range(0, H, h_unit)[:-1]:
            for w in range(0, W, w_unit)[:-1]:
                grid = field[h : h + h_unit, w : w + w_unit]
                puyo = classifier.predict(grid, template_type="field")
                init_field[h // h_unit, w // w_unit] = puyo

        init_field = field_edit(init_field)
        fields.append(init_field)
    return fields


def get_puyo_judge(img):
    judge_img = img[200:300, 0:100].reshape(-1)
    template_judge_img = cv2.imread("images/judge_img.jpg").reshape(-1)
    diff = np.mean(judge_img - template_judge_img)
    return judge_img


def get_next_puyo_info(img):
    player1_next = img[100 + 5 + 3 : 200 - 5 - 6, 470 + 10 : 520 + 5 - 3]
    h, w, c = player1_next.shape
    player1_next = player1_next[: h // 2], player1_next[h // 2 :]
    player1_next = [classifier.predict(i, template_type="p1") for i in player1_next]

    player1_next_next = img[200 - 4 : 250 + 11, 500 + 10 : 550 - 5]
    h, w, c = player1_next_next.shape
    player1_next_next = player1_next_next[: h // 2], player1_next_next[h // 2 :]
    player1_next_next = [
        classifier.predict(i, template_type="p1") for i in player1_next_next
    ]

    player2_next = img[100 + 5 + 3 : 200 - 5 - 6, 750 + 2 + 5 : 800 + 2 - 3]
    h, w, c = player2_next.shape
    player2_next = player2_next[: h // 2], player2_next[h // 2 :]
    player2_next = [classifier.predict(i, template_type="p2") for i in player2_next]

    player2_next_next = img[200 - 4 : 250 + 11, 725 + 2 : 765 - 3]
    h, w, c = player2_next_next.shape
    player2_next_next = player2_next_next[: h // 2], player2_next_next[h // 2 :]
    player2_next_next = [
        classifier.predict(i, template_type="p2") for i in player2_next_next
    ]

    output = [player1_next, player1_next_next, player2_next, player2_next_next]
    return output


class puyo_classifier(object):
    def __init__(self, puyo_types):
        self._puyo_types = puyo_types
        self._field_template = {}
        for name in self._puyo_types:
            img = cv2.resize(cv2.imread(f"images/field/{name}.jpg"), (40, 40))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._field_template[name] = img

        self._p1_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p1/p1_{name}.jpg"), (40, 40))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p1_template[name] = img

        self._p2_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p2/p2_{name}.jpg"), (40, 40))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p2_template[name] = img

    def predict(self, img, template_type="field"):
        img = cv2.resize(img, (40, 40))
        differences = []

        channel = 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])

        for name in self._puyo_types[:-2]:
            if template_type == "field":
                template_img = self._field_template[name]
            elif template_type == "p1":
                template_img = self._p1_template[name]
            elif template_type == "p2":
                template_img = self._p2_template[name]
            template_img_hist = cv2.calcHist(
                [template_img], [channel], None, [256], [0, 256]
            )

            diff = cv2.compareHist(template_img_hist, img_hist, 0)
            differences.append(diff)

        if template_type == "field":
            channel = 0
            img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])

            for name in ["ojama", "back"]:
                satu = np.mean(img[:, :, 1])
                if satu < 110 and name == "back":
                    diff = 1.00
                else:
                    template_img = self._field_template[name]
                    template_img_hist = cv2.calcHist(
                        [template_img], [channel], None, [256], [0, 256]
                    )
                    diff = cv2.compareHist(template_img_hist, img_hist, 0)

                differences.append(diff)
        puyo_type = differences.index(max(differences))
        return puyo_type


class FieldConstructor(object):
    def __init__(self, puyo_types):
        self._puyo_types = puyo_types
        self._field_template = {}
        for name in self._puyo_types:
            img = cv2.resize(cv2.imread(f"images/field/{name}.jpg"), (40, 40))
            self._field_template[name] = img

    def make_field_construct(self, field):
        init_img = np.zeros((480, 240, 3), dtype=np.uint8)
        for h in range(12):
            for w in range(6):
                puyo = field[h, w]
                puyo = self._puyo_types[int(puyo)]
                template_img = self._field_template[puyo]
                grid_h = h * 40
                grid_w = w * 40
                init_img[grid_h : grid_h + 40, grid_w : grid_w + 40] = template_img
        return init_img


def field_edit(field):
    if field[1, 2] == 6:
        field[0, 2] = 6
    return field


puyo_types = ["aka", "ao", "kiiro", "midori", "murasaki", "ojama", "back"]
classifier = puyo_classifier(puyo_types)


def main():
    img = cv2.imread("sample.jpg")
    field_puyos = get_field_info(img)
    next_puyos = get_next_puyo_info(img)
    print(next_puyos)

    fc = FieldConstructor(puyo_types)
    player1_img = fc.make_field_construct(field_puyos[0])
    cv2.imwrite("player1_img.jpg", player1_img)
    player2_img = fc.make_field_construct(field_puyos[1])
    cv2.imwrite("player2_img.jpg", player2_img)


if __name__ == "__main__":
    main()
