import matplotlib.pyplot as plt
import numpy as np
import cv2

from sortifyscan.cargo import constants
from sortifyscan.export import ExportMedia


class Boxwgh:
    # TODO перенести из джсона координаты
    # TODO сделать первые измерения

    def __str__(self):
        return f"""{{
'up_side': {self.up_side}
'down_side': {self.down_side}
'left_side': {self.left_side}
'right_side': {self.right_side}
'front_side': {self.front_side}
'back_side': {self.back_side}
'length_x': {self.length_x}
'length_y': {self.length_y}
'length_z': {self.length_z}
}}"""

    # TODO: Распределение в сторон в переменные контруктор фронт и верхний
    def __init__(self, json_data=None):
        self.front_side = Side()
        self.up_side = Side()
        self.down_side = Side()
        self.left_side = Side()
        self.right_side = Side()
        self.back_side = Side()
        self.length_x = None
        self.length_y = None
        self.length_z = None

        if json_data is not None:
            front_side_down_edge, front_side_up_edge, front_side_left_edge, front_side_right_edge \
                = Boxwgh.search_front_side(json_data)
            self.front_side.down_edge = Edge(front_side_down_edge)
            self.front_side.up_edge = Edge(front_side_up_edge)
            self.front_side.left_edge = Edge(front_side_left_edge)
            self.front_side.right_edge = Edge(front_side_right_edge)

            up_side_down_edge, up_side_up_edge, up_side_left_edge, up_side_right_edge \
                = Boxwgh.search_up_side(json_data)
            self.up_side.down_edge = Edge(up_side_down_edge)
            self.up_side.up_edge = Edge(up_side_up_edge)
            self.up_side.left_edge = Edge(up_side_left_edge)
            self.up_side.right_edge = Edge(up_side_right_edge)

    @staticmethod
    def search_front_side(sd):
        lowest_centroids = sd['sides'][0]['edges'][0]['line']['centoroid']['xy'][1]
        lowest_side = sd['sides'][0]['edges']
        for side in sd['sides']:
            for edge in side['edges']:
                if lowest_centroids < edge['line']['centoroid']['xy'][1]:
                    lowest_centroids = edge['line']['centoroid']['xy'][1]
                    lowest_side = side['edges']

        down_edge, up_edge, left_edge, right_edge = Boxwgh.distribution_edge(lowest_side)
        return down_edge, up_edge, left_edge, right_edge

    @staticmethod
    def search_up_side(sd):
        uppers_centroids = sd['sides'][0]['edges'][0]['line']['centoroid']['xy'][1]
        upper_side = sd['sides'][0]['edges']
        for side in sd['sides']:
            for edge in side['edges']:
                if uppers_centroids > edge['line']['centoroid']['xy'][1]:
                    uppers_centroids = edge['line']['centoroid']['xy'][1]
                    upper_side = side['edges']

        down_edge, up_edge, left_edge, right_edge = Boxwgh.distribution_edge(upper_side)
        return down_edge, up_edge, left_edge, right_edge

    @staticmethod
    def distribution_edge(edges):
        down = up = edges[0]['line']['centoroid']['xy'][1]
        left = right = edges[0]['line']['centoroid']['xy'][0]
        down_edge = up_edge = left_edge = right_edge = edges[0]
        for edge in edges:
            if down < edge['line']['centoroid']['xy'][1]:
                down = edge['line']['centoroid']['xy'][1]
                down_edge = edge
            if up > edge['line']['centoroid']['xy'][1]:
                up = edge['line']['centoroid']['xy'][1]
                up_edge = edge
            if left > edge['line']['centoroid']['xy'][0]:
                left = edge['line']['centoroid']['xy'][0]
                left_edge = edge
            if right < edge['line']['centoroid']['xy'][0]:
                right = edge['line']['centoroid']['xy'][0]
                right_edge = edge

        return down_edge, up_edge, left_edge, right_edge

    def equality_edges_length(self):

        first_dimension = [self.front_side.down_edge, self.front_side.up_edge,
                           self.back_side.down_edge, self.back_side.up_edge,
                           self.down_side.up_edge, self.down_side.down_edge,
                           self.up_side.up_edge, self.up_side.down_edge,
                           self.length_x]

        second_dimension = [self.left_side.down_edge, self.left_side.up_edge,
                            self.right_side.down_edge, self.right_side.up_edge,
                            self.down_side.left_edge, self.down_side.right_edge,
                            self.up_side.left_edge, self.up_side.right_edge,
                            self.length_z]

        third_dimension = [self.front_side.left_edge, self.front_side.right_edge,
                           self.back_side.left_edge, self.back_side.right_edge,
                           self.left_side.left_edge, self.left_side.right_edge,
                           self.right_side.left_edge, self.right_side.right_edge,
                           self.length_y]

        for dimension in [first_dimension, second_dimension, third_dimension]:
            for edge in dimension:
                if edge.length_m is not None:
                    for e in third_dimension:
                        e.length_m = edge.length_m
                    break


class Side:
    def __str__(self):
        return f"""{{
    'up_edge':{self.up_edge} 
    'down_edge': {self.down_edge} 
    'left_edge': {self.left_edge}
    'right_edge': {self.right_edge}
}}"""

    # TODO: Распределение в ребер в переменные контруктор нуден верних нижний и боковой на верхней
    def __init__(self, json_data: dict = None):
        self.up_edge = Edge()
        self.down_edge = Edge()
        self.left_edge = Edge()
        self.right_edge = Edge()


class Point:
    def __str__(self):
        return f"""{{'x':{self.x}, 'y':{self.y}}}"""

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def get_xy(self):
        return [self.x, self.y]

    @property
    def is_xy(self):
        return self.x is not None and self.y is not None


class Edge:
    def __str__(self):
        return f"""
    {{
        'point_1': {self.point_1},
        'point_2': {self.point_2},
        'centroid': {self.centroid}, 
        'alpha': {self.alpha}, 
        'beta': {self.beta}, 
        'length_m': {self.length_m},
        'length_p': {self.length_p},
    }}"""

    def __init__(self, json_data: dict = None, p1: Point = None, p2: Point = None):
        self.point_1 = p1
        self.point_2 = p2
        self.centroid = Point()
        self.length_m = None
        self.length_p = None
        self.alpha = None
        self.beta = None

        if json_data is not None:
            js = json_data['line']['xy']
            self.point_1 = Point(js[0][0], js[0][1])
            self.point_2 = Point(js[1][0], js[1][1])
            self.length_p = self.distance(js[0][0], js[0][1], js[1][0], js[1][1])
            if 'length' in json_data['line']:
                self.length_m = json_data['line']['length']

            if 'centoroid' in json_data['line']:
                jc = json_data['line']['centoroid']['xy']
                self.centroid = Point(jc[0], jc[1])

            # self.alpha, self.beta = self.linear_equation(self.point_1.x, self.point_1.y,
            #                                              self.point_2.x, self.point_2.y)

    def get_edge(self):
        return [self.point_1.get_xy(), self.point_2.get_xy()]

    @property
    def is_xy(self):
        return self.point_1.is_xy and self.point_2.is_xy

    @property
    def get_len(self):
        if self.is_xy:
            p1 = self.point_1.get_xy()
            p2 = self.point_2.get_xy()
            # print(p1, p2)
            return self.distance(p1[0], p1[1], p2[0], p2[1])
        return None

    # @staticmethod
    # def linear_equation(x1, y1, x2, y2):
    #     a = (y2 - y1) / (x2 - x1)
    #     b = y1 - a * x1
    #     # print(f"Уравнение прямой: y = {a}x + {b}")
    #     return a, b

    @staticmethod
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Borders:
    def __str__(self):
        return f"""{{
'up':{self.up} 
'down': {self.down} 
'left': {self.left}
'right': {self.right}
'proportion_left_right': {self.proportion_p_left_right}

    }}"""

    def __init__(self, d):
        for k in d:
            d[k]['line']['xy'][0][1] = d[k]['line']['xy'][0][1]
            d[k]['line']['xy'][1][1] = d[k]['line']['xy'][1][1]

        self.up = Edge(d['up'])
        self.down = Edge(d['down'])
        self.left = Edge(d['left'])
        self.right = Edge(d['right'])
        self.scale = 500
        self.proportion_p_left_right = self.left.length_p / self.right.length_p

        image_coords = np.array([[self.up.point_1.x, self.up.point_1.y],
                                 [self.up.point_2.x, self.up.point_2.y],
                                 [self.down.point_2.x, self.down.point_2.y],
                                 [self.down.point_1.x, self.down.point_1.y]],
                                dtype=np.float32)  # Координаты на изображении
        world_coords = np.array([[0, 0],
                                 [self.up.length_m * self.scale, 0],
                                 [self.down.length_m * self.scale, self.right.length_m * self.scale],
                                 [0, self.left.length_m * self.scale]], dtype=np.float32)  # Новые координаты
        # Вычисление матрицы преобразования перспективы
        self.perspective_matrix = cv2.getPerspectiveTransform(image_coords, world_coords)
        # Инвертирование матрицы преобразования
        self.inverse_perspective_matrix = np.linalg.inv(self.perspective_matrix)

    def parallel_mesh(self, min_limit=-1, max_limit=-639, step=-50):
        mesh = []
        for dyl in range(min_limit, max_limit, step):
            dxl = int((dyl - self.left.beta) / self.left.alpha)
            dyr = int(dyl * self.proportion_p_left_right)
            dxr = int((dyl - self.right.beta) / self.right.alpha)
            mesh.append([[dxl, -dyl], [dxr, -dyr]])
        return mesh

    def get_orto_coords(self, edge: Edge):
        return cv2.perspectiveTransform(np.array(edge.get_edge(), dtype=np.float32)
                                        .reshape(-1, 1, 2), self.perspective_matrix)

    def draw_gabarity(self, box: Boxwgh, image, show=False, save_dir_path=None, name_img="Name"):
        def draw(edge, lenght):
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            plt.plot(x, y, linewidth=3, linestyle='-')
            middle_x = (x[0] + x[-1]) / 2
            middle_y = (y[0] + y[-1]) / 2
            # plt.text(middle_x, middle_y, round(lenght, 3), color='black', fontsize=20, ha='center', va='center')
            # plt.text(middle_x, middle_y, round(lenght, 3), color='purple', fontsize=20, ha='center', va='center')
            plt.text(middle_x, middle_y, f"{round(lenght, 3)}м", fontsize=13, color='black', ha='center', va='center',
                     bbox=dict(facecolor='red', alpha=0.5))

        edges = [box.front_side.down_edge, box.front_side.left_edge, box.up_side.left_edge]
        for edge in edges:
            draw(edge.get_edge(), edge.length_m)

        plt.imshow(image)
        # Путь для сохранения изображения

        if save_dir_path is not None:
            ExportMedia.export_plt(name_img, plt, save_dir_path)

        if show:
            plt.show()
        plt.close()

    # TODO проверить после контрукторов
    def get_gabarity(self, box: Boxwgh):
        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        orto_front_down = self.get_orto_coords(box.front_side.down_edge)
        orto_front_up = self.get_orto_coords(box.front_side.up_edge)
        orto_front_left = self.get_orto_coords(box.front_side.left_edge)

        orto_up_down = self.get_orto_coords(box.up_side.down_edge)
        orto_up_up = self.get_orto_coords(box.up_side.up_edge)
        orto_up_left = self.get_orto_coords(box.up_side.left_edge)

        box.front_side.down_edge.length_p = np.linalg.norm(orto_front_down[0] - orto_front_down[1]) / self.scale
        box.front_side.down_edge.length_m = box.front_side.down_edge.length_p
        box.length_x = box.front_side.down_edge.length_m

        box.front_side.up_edge.length_p = np.linalg.norm(orto_front_up[0] - orto_front_up[1]) / self.scale
        box.front_side.up_edge.length_m = box.front_side.down_edge.length_m

        box.front_side.left_edge.length_p = np.linalg.norm(orto_front_left[0] - orto_front_left[1]) / self.scale
        box.front_side.left_edge.length_m = box.front_side.left_edge.length_p * \
                                            (box.front_side.down_edge.length_p / box.front_side.up_edge.length_p)

        box.up_side.down_edge.length_p = np.linalg.norm(orto_up_down[0] - orto_up_down[1]) / self.scale
        box.up_side.down_edge.length_m = box.front_side.down_edge.length_m

        box.up_side.up_edge.length_m = box.front_side.down_edge.length_m

        box.up_side.left_edge.length_p = np.linalg.norm(orto_up_left[0] - orto_up_left[1]) / self.scale
        box.up_side.up_edge.length_p = np.linalg.norm(orto_up_up[0] - orto_up_up[1]) / self.scale
        box.up_side.left_edge.length_m = box.up_side.left_edge.length_p \
 \
                                         * (box.front_side.down_edge.length_p / box.front_side.up_edge.length_p)

        box.length_x = box.front_side.down_edge.length_m
        box.length_y = box.front_side.left_edge.length_m
        box.length_z = box.up_side.left_edge.length_m

        width = 0.516
        lenght = 0.569
        height = 0.381
        print(
            f"Длина (0.569м): {box.length_z}м; delta {(lenght - box.length_z) * 100}см, %{(lenght - box.length_z) * 100 / box.length_z}")
        print(
            f"Ширина (0.516м): {box.length_x}м; delta {(width - box.length_x) * 100}см, %{(width - box.length_x) * 100 / box.length_x}")
        print(
            f"Высота (0.381м): {box.length_y}м; delta {(height - box.length_y) * 100}см, %{(height - box.length_y) * 100 / box.length_y}")

        return

    def draw_image_orto(self, image, save_dir_path, name_img):
        perspective_image = cv2.warpPerspective(image, self.perspective_matrix,
                                                (int(max(self.up.length_m * self.scale, self.down.length_m)) + 200,
                                                 int(max(self.left.length_m * self.scale, self.right.length_m))))

        plt.imshow(perspective_image)
        ExportMedia.export_plt(plt=plt, path=save_dir_path, n_shot=name_img)
        if constants.DEBUG:
            plt.show()

    # def test(self, front_side, up_side, path=PATH_PRIVATE_IMAGE):
    #     # Исходное изображение
    #     image = cv2.imread(path)
    #     # Применение преобразования перспективы к изображению
    #     perspective_image = cv2.warpPerspective(image, self.perspective_matrix,
    #                                             (int(max(self.up.length_m * self.scale, self.down.length_m)) + 200,
    #                                              int(max(self.left.length_m * self.scale, self.right.length_m))))
    #     plt.imshow(perspective_image)
    #     plt.show()
    #
    #     # Инвертирование матрицы преобразования
    #     # inverse_perspective_matrix = np.linalg.inv(perspective_matrix)
    #
    #     # Известные координаты точек на изображении
    #     image_coords_unknown = np.array(
    #         [[277, 338], [228, 218], [487, 302], [267, 189], [515, 155], [217, 80], [414, 59]], dtype=np.float32)
    #
    #     # Преобразование координат изображения в координаты реального мира
    #     world_coords_unknown = cv2.perspectiveTransform(image_coords_unknown.reshape(-1, 1, 2),
    #                                                     self.perspective_matrix)
    #
    #     print()
    #     print(world_coords_unknown[0][0])
    #     print(world_coords_unknown[1][0])
    #     print(world_coords_unknown[2][0])
    #     print()
    #
    #     def draw(p1, p2):
    #         x_values = [p1[0][0], p2[0][0]]
    #         y_values = [p1[0][1], p2[0][1]]
    #         plt.plot(x_values, y_values, color='red', linewidth=0.5, linestyle='-')
    #
    #     # draw(world_coords_unknown[0], world_coords_unknown[1])
    #     draw(world_coords_unknown[0], world_coords_unknown[2])
    #     draw(world_coords_unknown[0], world_coords_unknown[3])
    #     draw(world_coords_unknown[3], world_coords_unknown[4])
    #     draw(world_coords_unknown[5], world_coords_unknown[6])
    #
    #     plt.imshow(perspective_image)
    #     plt.show()
    #
    #     left_down = np.linalg.norm(world_coords_unknown[0] - world_coords_unknown[1]) / self.scale
    #     front_down = np.linalg.norm(world_coords_unknown[0] - world_coords_unknown[2]) / self.scale
    #     front_left = np.linalg.norm(world_coords_unknown[0] - world_coords_unknown[3]) / self.scale
    #     front_up = np.linalg.norm(world_coords_unknown[3] - world_coords_unknown[4]) / self.scale
    #     up_up = np.linalg.norm(world_coords_unknown[5] - world_coords_unknown[6]) / self.scale
    #     up_left = np.linalg.norm(world_coords_unknown[3] - world_coords_unknown[5]) / self.scale
    #
    #     height = front_left / (front_up / front_down)
    #     width = front_down
    #     length = up_left / (up_up / front_up) / (front_up / front_down)
    #
    #     print(f"Длина (0.569м): {length}м; delta {(0.569 - length) * 100}см")
    #     print(f"Ширина (0.516м): {width}м; delta {(0.516 - width) * 100}см")
    #     print(f"Высота (0.381м): {height}м; delta {(0.381 - height) * 100}см")
