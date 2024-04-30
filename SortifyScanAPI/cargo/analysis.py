import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cargo.constants as c
from shapely.geometry import LinearRing, LineString

from cargo import constants


class CargoAnalysis:
    @staticmethod
    def calculate_perspective_coef():
        def distance(xs, ys):
            return np.sqrt((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2)

        return distance(c.ARR_LENT_DOWN[:, 0], -c.ARR_LENT_DOWN[:, 1]) / \
            distance(c.ARR_LENT_UP[:, 0], -c.ARR_LENT_UP[:, 1])

    @staticmethod
    def draw_point_cloud_edges(xy: list, size=None):
        if not constants.DEBUG: return
        if size is None:
            size = [640, 640]

        plt.figure()
        ax = plt.axes()

        plt.xlim([0, size[0]])
        plt.ylim([-size[1], 10])  # -640, 0  , 10потому что не видно врезнюю границу
        print(len(xy))
        x1, y1, x2, y2, x3, y3 = xy

        # точки граней
        ax.scatter(x1, y1, 0.1)
        ax.scatter(x2, y2, 0.1)
        ax.scatter(x3, y3, 0.1)

        # # контуры граней
        # plt.plot(x1, y1)
        # plt.plot(x2, y2)
        # plt.plot(x3, y3)

        # границы ленты
        plt.plot(c.ARR_LENT_LEFT[:, 0], -c.ARR_LENT_LEFT[:, 1])
        plt.plot(c.ARR_LENT_RIGHT[:, 0], -c.ARR_LENT_RIGHT[:, 1])
        plt.plot(c.ARR_LENT_DOWN[:, 0], -c.ARR_LENT_DOWN[:, 1])
        plt.plot(c.ARR_LENT_UP[:, 0], -c.ARR_LENT_UP[:, 1])

        plt.show()

    @staticmethod
    def get_xy_edges_SAM(result, bbox=None):
        # for r in result:
        #   print(len(r.masks.xy[0]))
        #   print(len(r.masks.xy[1]))
        #   print(len(r.masks.xy[2]))
        #   print(len(r.masks.xy[3]))
        if bbox is None:
            bbox = [0, 0]
        r = result[0]

        bbx1, bby1 = bbox[0], bbox[1]
        # TODO: Нужно определить нужные индексы координат для граней, возможно ошибочное выделение
        x1, y1 = bbx1 + np.array(r.masks.xy[-3])[:, 0], -(np.array(r.masks.xy[-3])[:, 1] + bby1)
        x2, y2 = bbx1 + np.array(r.masks.xy[-2])[:, 0], -(np.array(r.masks.xy[-2])[:, 1] + bby1)
        x3, y3 = bbx1 + np.array(r.masks.xy[-1])[:, 0], -(np.array(r.masks.xy[-1])[:, 1] + bby1)

        point_cloud = [x1, y1, x2, y2, x3, y3]

        print(len(x1), len(y1))
        print(len(x2), len(y2))
        print(len(x3), len(y3))

        return point_cloud

    @staticmethod
    def get_xy_edges_OpenCV(result):
        point_cloud = []

        for contour in result:
            x_coords = contour[:, 0, 0]
            y_coords = -contour[:, 0, 1]
            print(len(x_coords), len(y_coords))
            point_cloud.append(x_coords)
            point_cloud.append(y_coords)

        return point_cloud

    @staticmethod
    def approximate_point_cloud(point_cloud: list):
        x1, y1, x2, y2, x3, y3 = point_cloud

        line_strings_all = [CargoAnalysis.simplify_polygon(np.c_[x1, y1], epsilon=5),
                            CargoAnalysis.simplify_polygon(np.c_[x2, y2], epsilon=5),
                            CargoAnalysis.simplify_polygon(np.c_[x3, y3], epsilon=5)]

        return line_strings_all

    @staticmethod
    def draw_edges(line_strings_all: list, image):
        if not constants.DEBUG: return

        def draw_line_strings(lines):
            x_line, y_line = [], []
            for line in lines:
                x_line += line.xy[0]
                y_line += line.xy[1]
            y_line[:] = [-i for i in y_line]
            plt.plot(x_line, y_line, 'o-')

        plt.figure()
        plt.axes()
        # plt.xlim([0, size[0]])
        # plt.ylim([-size[1], 10])#-640, 0  , 10потому что не видно врезнюю границу

        for line_strings in line_strings_all:
            draw_line_strings(line_strings)

        # границы ленты
        plt.plot(c.ARR_LENT_LEFT[:, 0], c.ARR_LENT_LEFT[:, 1])
        plt.plot(c.ARR_LENT_RIGHT[:, 0], c.ARR_LENT_RIGHT[:, 1])
        plt.plot(c.ARR_LENT_DOWN[:, 0], c.ARR_LENT_DOWN[:, 1])
        plt.plot(c.ARR_LENT_UP[:, 0], c.ARR_LENT_UP[:, 1])
        # Открываем изображение с помощью Pillow
        # image = Image.open(c.PATH_BEGIN_IMAGE)
        # Наложение графика на изображение
        plt.imshow(image)

        plt.show()

    # находим четыре самых больших отрезка и ищем где они последовательное соединены мелочью, затем дропаем мелочь и
    # ищем где пересекаются большие
    @staticmethod
    def simplify_polygon(points, epsilon):
        """ Упрощение многоугольника с помощью метода Рамера-Дугласа-Пекера сводит к 4тырем сторонам"""
        # Упрощение линии с помощью метода Рамера-Дугласа-Пекера
        ring = LinearRing(points).simplify(epsilon)

        # Вершины упрощенного многоугольника -1 так как последняя координата повторяется
        xy = np.c_[np.array(ring.coords)]

        line_strings = CargoAnalysis.drop_small_line(xy)
        line_strings = CargoAnalysis.prolongation_segments(line_strings)
        line_strings = CargoAnalysis.cut_at_intersection(line_strings)

        return line_strings

    @staticmethod
    def drop_small_line(xy):
        """ Возвращает набор 4-рех ребер LineString (сведение до четырехугольника)"""
        # вынесем в отдельный список набор LineString
        xy_line_lenght = []
        for i in range(len(xy) - 1):
            line = LineString([xy[i], xy[i + 1]])
            xy_line_lenght.append([line.length, line])

        # найдем 4 самых больших ребра, дропнем мелочь
        while len(xy_line_lenght) != 4:
            xy_line_lenght.remove(min(xy_line_lenght))

        # пересоздадим массив без длинны ребер
        xy_line_lenght = np.array(xy_line_lenght)
        xy_line_lenght = xy_line_lenght[:, 1]

        return xy_line_lenght

    @staticmethod
    def prolongation_segments(line_strings):
        """ Возвращает масштабированный набор ребер в формате LineString"""
        # интреполируем отрезки с обоих сторон что бы потом обрезать в пересечениях условно на своюэе длинну
        for i in range(len(line_strings)):
            f = list(line_strings[i].coords)

            p1 = [f[0][0] * 2 - f[1][0], f[0][1] * 2 - f[1][1]]
            p2 = [f[1][0] * 2 - f[0][0], f[1][1] * 2 - f[0][1]]
            line_strings[i] = LineString([p1, p2])
            # x_line, y_line = line_strings[i].xy

        return line_strings

    @staticmethod
    def cut_at_intersection(line_strings):
        """ Обрезает прямые по ресечениям до четырехугольника"""
        lines = []
        # print(line_strings)
        # print(len(line_strings))
        for i in range(len(line_strings)):
            if i == len(line_strings) - 1:
                points = line_strings[i].intersection([line_strings[i - 1], line_strings[0]])
            else:
                points = line_strings[i].intersection([line_strings[i - 1], line_strings[i + 1]])
            lines.append(LineString([points[0].coords[0], points[1].coords[0]]))
            # если тут дропает значит не правильно сегментирует стороны
        #     вариант дропать лишнее логическим вычитанием из большего меньшее

        return lines
