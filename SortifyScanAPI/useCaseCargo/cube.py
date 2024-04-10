class Cube:
    # TODO перенести из джсона координаты
    # TODO сделать первые измерения
    def __init__(self, json_data):
        self.up_side = Side()
        self.down_side = Side()
        self.left_side = Side()
        self.right_side = Side()
        self.front_side = Side()
        self.back_side = Side()
        self.length_x = None
        self.length_y = None
        self.length_z = None

    def equality_edges(self):
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
                if edge.length is not None:
                    for e in third_dimension:
                        e.length = edge.length
                    break


class Side:
    def __init__(self):
        self.up_edge = Edge()
        self.down_edge = Edge()
        self.left_edge = Edge()
        self.right_edge = Edge()


class Edge:
    def __init__(self):
        self.point_1 = Point()
        self.point_2 = Point()
        self.centroid = Point()
        self.length = None


class Point:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
