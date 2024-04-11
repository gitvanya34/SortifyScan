import json
import cargo
import boxwgh
# def case1():
#     result = cargo.CargoDetection.detect_cargo()
#     show_image_after_ultralytics(result)
#
#     clean_memory_for_gpu()
#
#     bbox = get_bbox_from_result(result_to_json(result))
#     result = segmentation_bboxes(result[0].orig_img, bbox)
#     show_image_after_ultralytics(result)
#
#     result_side = segmentation_of_the_side(result, crop = True, bgcolor = 'black')
#     show_image_after_ultralytics(result_side)
#
#     point_cloud = get_xy_edges(result_side, bbox)#TODO bbox если кроп передавать нормаьный bbox сделать проверку
#     draw_point_cloud_edges(point_cloud)# TODO: передать размеры изображения аргументом
#
#
#     line_strings_all = approximate_point_cloud(point_cloud, size = [640,640])
#     draw_edges(line_strings_all, size = [640,640])
#
#     print(line_strings_all)
#     # Формируем словарь граней c координатами
#     sides_dict = {"sides":[]}
#     for i, side in enumerate(line_strings_all):
#       edge_dict = {"edges":[]}
#       for j, edge in enumerate(side):
#         edge_dict[f"edges"].append({f"line": {"xy": [[edge.xy[0][0], edge.xy[1][0]], [edge.xy[0][1], edge.xy[1][1]]],\
#                                               "centoroid": {"xy" : [edge.centroid.xy[0][0],edge.centroid.xy[1][0]]}}})
#       sides_dict[f"sides"].append(edge_dict)
#
#     # Формируем словарь граней c нормальными
#
#
#     # добавляем на будущее словарь центрроидов грани
#     return sides_dict
import numpy as np

from privateconstants import PATH_PRIVATE_IMAGE


def distance(xs, ys):
    return np.sqrt((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2)


def search_lowest_centroid_edges(sd):
    centroids = []

    for i in sd['sides']:
        c = []
        for j in i['edges']:
            c.append(j['line']['centoroid']['xy'])
        centroids.append(c)
    print(centroids)
    min_centroids = sd['sides'][0]['edges'][0]['line']['centoroid']['xy'][1]

    return min_centroids


def search_lowest_edge(sd):
    lowest_edges = []
    min_centroids = search_lowest_centroid_edges(sd)
    for i in sd['sides']:
        c = []
        for j in i['edges']:
            if j['line']['centoroid']['xy'][1] < min_centroids:
                min_centroids = j['line']['centoroid']['xy'][1]
                lowest_edges = j

    # coef_position = (lowest_edges['line']['centoroid']['xy'][1] + 640) / 640
    # # print(coef_position * perspective_coef * p_dist)

    return lowest_edges


def proportion_work_area():
    left_border_dist = distance(cargo.ARR_LENT_LEFT[0], cargo.ARR_LENT_LEFT[1])
    right_border_dist = distance(cargo.ARR_LENT_RIGHT[0], cargo.ARR_LENT_RIGHT[1])
    print(left_border_dist, right_border_dist, left_border_dist/right_border_dist)


if __name__ == "__main__":
    sides_dict = cargo.MOCK_SIDES_DICT
    sides_dict = json.loads(sides_dict.replace("\'", "\""))
    print(sides_dict['sides'][0]['edges'][0])

    print(search_lowest_edge(sides_dict))
    box = boxwgh.Boxwgh()
    box.front_side.down_edge = boxwgh.Edge(search_lowest_edge(sides_dict))
    print(box.front_side.down_edge)

    proportion_work_area()
    print(cargo.JSON_BORDERS)
    borders = boxwgh.Borders(cargo.JSON_BORDERS)
    print(borders.draw_mesh(borders.parallel_mesh()))
    print(borders)

    import cv2
    import numpy as np
    # Исходное изображение
    image = cv2.imread(PATH_PRIVATE_IMAGE)
    # Известные мировые координаты прямоугольной области
    # world_coords = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
    #
    # # Известные проекции мировых координат на изображение
    # image_coords = np.array([[188, 577], [638, 509], [165, 18], [332, 9]], dtype=np.float32)

    # Заданные начальные и конечные точки преобразования
    src_points = np.array([[165, 18], [332, 9], [638, 509], [188, 577]], dtype=np.float32)  # Координаты на изображении
    dst_points = np.array([[0, 0], [160, 0], [160, 600], [0, 600]], dtype=np.float32)  # Новые координаты

    # Вычисление матрицы преобразования перспективы
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Применение преобразования перспективы к изображению
    perspective_image = cv2.warpPerspective(image, perspective_matrix, (640, 640))

    # Отображение ортогонального изображения
    cv2.imshow('Orthogonal Image', perspective_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
