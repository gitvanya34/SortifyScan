import numpy as np
import matplotlib.pyplot as plt

import sortifyscan.cargo.constants as c
from shapely.geometry import LinearRing, LineString

from sortifyscan.cargo import constants
from sortifyscan.export import ExportMedia


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
        # print(len(xy))
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

        # print(len(x1), len(y1))
        # print(len(x2), len(y2))
        # print(len(x3), len(y3))

        return point_cloud

    @staticmethod
    def get_xy_edges_OpenCV(result):
        point_cloud = []

        for contour in result:
            x_coords = contour[:, 0, 0]
            y_coords = -contour[:, 0, 1]
            # print(len(x_coords), len(y_coords))
            point_cloud.append([x_coords, y_coords])

        return point_cloud

    @staticmethod
    def approximate_point_cloud(point_cloud: list):
        # на вход [[[x],[y]],[..]]

        # TODO сделать объединения пересекаемых полигонов после апроксимации
        x1, y1, = point_cloud[0]
        x2, y2 = point_cloud[1]
        x3, y3 = point_cloud[2]

        edge1 = np.c_[x1, y1]
        edge2 = np.c_[x2, y2]
        edge3 = np.c_[x3, y3]
        line_strings_all = [CargoAnalysis.simplify_polygon(edge1, epsilon=5),
                            CargoAnalysis.simplify_polygon(edge2, epsilon=5),
                            CargoAnalysis.simplify_polygon(edge3, epsilon=5)]

        return line_strings_all

    @staticmethod
    def draw_edges(line_strings_all: list, image, n_shot, path):
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
        plt.plot(c.ARR_LENT_LEFT[:, 0], c.ARR_LENT_LEFT[:, 1], color="red")
        plt.plot(c.ARR_LENT_RIGHT[:, 0], c.ARR_LENT_RIGHT[:, 1], color="red")
        plt.plot(c.ARR_LENT_DOWN[:, 0], c.ARR_LENT_DOWN[:, 1], color="red")
        plt.plot(c.ARR_LENT_UP[:, 0], c.ARR_LENT_UP[:, 1], color="red")
        # Открываем изображение с помощью Pillow
        # image = Image.open(c.PATH_BEGIN_IMAGE)
        # Наложение графика на изображение

        plt.imshow(image)
        ExportMedia.export_plt(n_shot=n_shot, plt=plt, path=path)

        if constants.DEBUG:
            plt.axis('off')
            plt.show()
        plt.close()

    # находим четыре самых больших отрезка и ищем где они последовательное соединены мелочью, затем дропаем мелочь и
    # ищем где пересекаются большие
    @staticmethod
    def simplify_polygon(points, epsilon):

        """ Упрощение многоугольника с помощью метода Рамера-Дугласа-Пекера сводит к 4тырем сторонам"""

        def xy_to_line_strings(xy):
            line_strings = []
            # print(xy)
            for i in range(len(xy) - 1):
                line_strings.append(LineString([xy[i], xy[i + 1]]))
            return line_strings

        # Упрощение линии с помощью метода Рамера-Дугласа-Пекера
        ring = LinearRing(points)

        # Вершины упрощенного многоугольника -1 так как последняя координата повторяется
        xy = np.c_[np.array(ring.convex_hull.simplify(epsilon).minimum_rotated_rectangle.exterior.coords)]
        xy_rect = np.c_[np.array(ring.convex_hull.simplify(epsilon).exterior.coords)]
        xy_rect2 = np.c_[np.array(ring.simplify(epsilon).minimum_rotated_rectangle.exterior.coords)]

        # CargoAnalysis.minimize(xy, xy_rect)

        def draw_xy(xy, color):
            x = [point[0] for point in xy]
            y = [point[1] for point in xy]
            plt.plot(x, y, color=color, marker="o")

        if c.DEBUG:
            draw_xy(xy, "red")
            draw_xy(xy_rect, "blue")
            draw_xy(xy_rect2, "green")
            # plt.show()

        line_strings = xy_to_line_strings(xy_rect)
        line_strings = CargoAnalysis.drop_small_line(line_strings)
        # line_strings = CargoAnalysis.prolongation_segments(line_strings)
        # line_strings = CargoAnalysis.cut_at_intersection(line_strings)

        return line_strings

    @staticmethod
    def minimize(ring, rot_rect):
        # TODO минимизация четырехугольника на выпуклом описывающем многоугольнике
        # # Набор точек в формате LineString
        # points = ring
        #
        # def loss_function(vertices):
        #     approximated_polygon = Polygon(vertices.reshape(-1, 2))
        #     return -approximated_polygon.area  # Минимизируем площадь, поэтому возвращаем отрицательную площадь
        #
        # def constraints(vertices):
        #     return vertices[0] - vertices[2], vertices[1] - vertices[3]
        #
        # initial_guess = rot_rect
        # print(initial_guess)
        # # Оптимизация
        # result = minimize(loss_function, initial_guess, constraints={'type': 'eq', 'fun': constraints})
        #
        # # Получение аппроксимированного многоугольника
        # approximated_polygon = Polygon(result.x.reshape(-1, 2))
        #
        #
        # print("Аппроксимированный четырехугольник:", approximated_polygon)
        return

    @staticmethod
    def drop_small_line(lines):
        """ Возвращает набор 4-рех ребер LineString (сведение до четырехугольника)"""
        # вынесем в отдельный список набор LineString
        xy_line_lenght = []
        for line in lines:
            xy_line_lenght.append([line.length, line])

        num_edges_to_keep = 4

        def drop_line(left, midl, right):
            midl_p1, midl_p2 = midl[1].coords[0], midl[1].coords[-1]
            left_p1, left_p2 = left[1].coords[0], left[1].coords[-1]
            right_p1, right_p2 = right[1].coords[0], right[1].coords[-1]

            long_left, long_right = CargoAnalysis.prolongation_segments([left[1], right[1]])
            new_public = long_right.intersection(long_left)

            if new_public.is_empty:
                print("\nЛинии не пересекаются.\n")
                midl[0] = 999.0  # коэффициент неприкосновенности)
                return left, midl, right

            # print(left[1], midl[1], right[1])
            # print(left[1].intersection(midl[1]))
            #
            # print(left[1].intersection(midl[1]).coords[0] == left_p1,
            #       left[1].intersection(midl[1]).coords[0] == left_p2)
            # print(right[1].intersection(right[1]).coords[0] == right_p1,
            #       right[1].intersection(midl[1]).coords[0] == right_p2)
            #
            # print(right[1].intersection(midl[1]))
            left_notpublic = left_p1 if left[1].intersection(midl[1]).coords[0] != left_p1 else left_p2
            right_notpublic = right_p1 if right[1].intersection(midl[1]).coords[0] != right_p1 else right_p2

            new_public = new_public.coords[0]

            # print(new_public.coords.xy.tolist())
            new_left = LineString([left_notpublic, new_public])
            new_right = LineString([new_public, right_notpublic])

            new_left = [new_left.length, new_left]
            new_right = [new_right.length, new_right]
            # print(left, midl, right)
            # print(new_left, new_right)
            # print("Итог", new_left, new_right)
            return new_left, 0, new_right

        while len(xy_line_lenght) > num_edges_to_keep:
            i_midl = min(range(len(xy_line_lenght)), key=lambda i: xy_line_lenght[i][0])

            i_left = i_midl - 1
            i_right = (i_midl + 1) if i_midl != len(xy_line_lenght) - 1 else 0

            xy_line_lenght[i_left], xy_line_lenght[i_midl], xy_line_lenght[i_right] = drop_line(xy_line_lenght[i_left],
                                                                                                xy_line_lenght[i_midl],
                                                                                                xy_line_lenght[i_right])
            if xy_line_lenght[i_midl] == 0:
                xy_line_lenght.pop(i_midl)

        xy_line_lenght = np.array(xy_line_lenght)
        xy_line_lenght = xy_line_lenght[:, 1]

        if c.DEBUG:
            for line in xy_line_lenght:
                x, y = line.xy
                plt.plot(x, y, color="orange", marker="o")
            # Показываем график
            plt.show()
        return xy_line_lenght

    @staticmethod
    def prolongation_segments(line_strings):
        """ Возвращает масштабированный набор ребер в формате LineString"""
        # интерполируем отрезки с обеих сторон, что бы потом обрезать в пересечениях условно на своюэе длинну
        k = 2  # коэффициент пролонгации

        for i in range(len(line_strings)):
            p1 = list(line_strings[i].coords)[0]
            p2 = list(line_strings[i].coords)[1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            extended_p1 = [p1[0] - dx * k, p1[1] - dy * k]
            extended_p2 = [p2[0] + dx * k, p2[1] + dy * k]

            if c.DEBUG:
                plt.plot([extended_p1[0], extended_p2[0]], [extended_p1[1], extended_p2[1]])
                plt.show()

            line_strings[i] = LineString([extended_p1, extended_p2])
            # x_line, y_line = line_strings[i].xy
        return line_strings

    @staticmethod
    def cut_at_intersection(line_strings):
        """ Обрезает прямые по ресечениям до четырехугольника"""
        lines = []
        for i in range(len(line_strings)):

            if i == len(line_strings) - 1:
                points = line_strings[i].intersection([line_strings[i - 1], line_strings[0]])
            else:
                points = line_strings[i].intersection([line_strings[i - 1], line_strings[i + 1]])
            lines.append(LineString([points[0].coords[0], points[1].coords[0]]))
            # если тут дропает значит не правильно сегментирует стороны
        #     вариант дропать лишнее логическим вычитанием из большего меньшее

        return lines
