from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from privateconstants import PATH_PRIVATE_IMAGE


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

    def __init__(self, json_data=None):
        self.up_side = Side()
        self.down_side = Side()
        self.left_side = Side()
        self.right_side = Side()
        self.front_side = Side()
        self.back_side = Side()
        self.length_x = None
        self.length_y = None
        self.length_z = None

    # def equality_edges(self):
    #     if self.front_side.down_edge.is_xy:
    #
    #     print(self.front_side.down_edge.is_xy)
    #     print(self.front_side.up_edge.is_xy)

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
                print(edge)
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

    def __init__(self):
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

            self.alpha, self.beta = self.linear_equation(self.point_1.x, self.point_1.y,
                                                         self.point_2.x, self.point_2.y)

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
            print(p1, p2)
            return self.distance(p1[0], p1[1], p2[0], p2[1])
        return None

    @staticmethod
    def linear_equation(x1, y1, x2, y2):
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        print(f"Уравнение прямой: y = {a}x + {b}")
        return a, b

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
            d[k]['line']['xy'][0][1] = -d[k]['line']['xy'][0][1]
            d[k]['line']['xy'][1][1] = -d[k]['line']['xy'][1][1]

        self.up = Edge(d['up'])
        self.down = Edge(d['down'])
        self.left = Edge(d['left'])
        self.right = Edge(d['right'])
        self.scale = 500
        self.proportion_p_left_right = self.left.length_p / self.right.length_p

    def parallel_mesh(self, min_limit=-1, max_limit=-639, step=-50):
        mesh = []
        for dyl in range(min_limit, max_limit, step):
            dxl = int((dyl - self.left.beta) / self.left.alpha)
            dyr = int(dyl * self.proportion_p_left_right)
            dxr = int((dyl - self.right.beta) / self.right.alpha)
            mesh.append([[dxl, -dyl], [dxr, -dyr]])
        return mesh

    def draw_mesh(self, mesh):
        borders_line = [self.up.get_edge(),
                        self.down.get_edge(),
                        self.left.get_edge(),
                        self.right.get_edge()]

        print("borders_line", borders_line)
        print("mesh", mesh)
        for m in borders_line:
            print(m)
            x_values = [m[0][0], m[1][0]]
            y_values = [-m[0][1], -m[1][1]]
            plt.plot(x_values, y_values, color='red', linewidth=0.5, linestyle='-')

        for m in mesh:
            print(m)
            x_values = [m[0][0], m[1][0]]
            y_values = [m[0][1], m[1][1]]
            plt.plot(x_values, y_values, color='blue', linewidth=0.5, linestyle='-')

        image = Image.open(PATH_PRIVATE_IMAGE)
        plt.imshow(image)
        plt.show()

    def get_perspective_transform(self, side: Side = None):

        # Исходное изображение
        image = cv2.imread(PATH_PRIVATE_IMAGE)

        # Заданные начальные и конечные точки преобразования
        image_coords = np.array([[self.up.point_1.x, -self.up.point_1.y],
                                 [self.up.point_2.x, -self.up.point_2.y],
                                 [self.down.point_2.x, -self.down.point_2.y],
                                 [self.down.point_1.x, -self.down.point_1.y]],
                                dtype=np.float32)  # Координаты на изображении
        world_coords = np.array([[0, 0],
                                 [self.up.length_m * self.scale, 0],
                                 [self.down.length_m * self.scale, self.right.length_m * self.scale],
                                 [0, self.left.length_m * self.scale]], dtype=np.float32)  # Новые координаты

        # Вычисление матрицы преобразования перспективы
        perspective_matrix = cv2.getPerspectiveTransform(image_coords, world_coords)

        # Применение преобразования перспективы к изображению
        perspective_image = cv2.warpPerspective(image, perspective_matrix,
                                                (int(max(self.up.length_m * self.scale, self.down.length_m)) + 200,
                                                 int(max(self.left.length_m * self.scale, self.right.length_m))))
        plt.imshow(perspective_image)
        plt.show()

        # Инвертирование матрицы преобразования
        # inverse_perspective_matrix = np.linalg.inv(perspective_matrix)

        # Известные координаты точек на изображении
        image_coords_unknown = np.array(
            [[277, 338], [228, 218], [487, 302], [267, 189], [515, 155], [217, 80], [414, 59]], dtype=np.float32)

        # Преобразование координат изображения в координаты реального мира
        world_coords_unknown = cv2.perspectiveTransform(image_coords_unknown.reshape(-1, 1, 2),
                                                        perspective_matrix)

        print()
        print(world_coords_unknown[0][0])
        print(world_coords_unknown[1][0])
        print(world_coords_unknown[2][0])
        print()

        def draw(p1, p2):
            x_values = [p1[0][0], p2[0][0]]
            y_values = [p1[0][1], p2[0][1]]
            plt.plot(x_values, y_values, color='red', linewidth=0.5, linestyle='-')

        # draw(world_coords_unknown[0], world_coords_unknown[1])
        draw(world_coords_unknown[0], world_coords_unknown[2])
        draw(world_coords_unknown[0], world_coords_unknown[3])
        draw(world_coords_unknown[3], world_coords_unknown[4])
        draw(world_coords_unknown[5], world_coords_unknown[6])

        plt.imshow(perspective_image)
        plt.show()

        left_down = np.linalg.norm(world_coords_unknown[0] - world_coords_unknown[1]) / self.scale
        front_down = np.linalg.norm(world_coords_unknown[0] - world_coords_unknown[2]) / self.scale
        front_left = np.linalg.norm(world_coords_unknown[0] - world_coords_unknown[3]) / self.scale
        front_up = np.linalg.norm(world_coords_unknown[3] - world_coords_unknown[4]) / self.scale
        up_up = np.linalg.norm(world_coords_unknown[5] - world_coords_unknown[6]) / self.scale

        up_left = np.linalg.norm(world_coords_unknown[3] - world_coords_unknown[5]) / self.scale

        print(front_left / (front_up / front_down))
        height = front_left / (front_up / front_down)
        width = front_down
        print(front_left)
        print(front_up)
        print(up_up)
        print(up_left / (up_up / front_up) / (front_up / front_down))

        length = up_left / (up_up / front_up) / (front_up / front_down)
        print(f"Длина (0.569м): {length}м; delta {(0.569 - length) *100}см")
        print(f"Ширина (0.516м): {width}м; delta {(0.516 - width) *100}см")
        print(f"Высота (0.381м): {height}м; delta {(0.381 - height)*100}см")
