import gc
import json
import torch
# from cargo import CargoDetection, CargoAnalysis, CargoProcessing
from cargo import *
import boxwgh


def clean_memory_for_gpu():
    torch.cuda.empty_cache()
    gc.collect()


def case1():
    result = CargoDetection.detect_cargo()
    CargoProcessing.show_image_after_ultralytics(result)

    clean_memory_for_gpu()

    bbox = CargoProcessing.get_bbox_from_result(CargoProcessing.result_to_json(result))
    result = CargoDetection.segment_cargo_bboxes(result[0].orig_img, bbox)
    CargoProcessing.show_image_after_ultralytics(result)

    result_side = CargoDetection.segmentation_of_the_side(result, crop=True, bgcolor='black')
    CargoProcessing.show_image_after_ultralytics(result_side)

    point_cloud = CargoAnalysis.get_xy_edges(result_side,
                                             bbox)  # TODO bbox если кроп передавать нормаьный bbox сделать проверку
    CargoAnalysis.draw_point_cloud_edges(point_cloud)  # TODO: передать размеры изображения аргументом

    line_strings_all = CargoAnalysis.approximate_point_cloud(point_cloud)
    CargoAnalysis.draw_edges(line_strings_all)

    print(line_strings_all)
    # Формируем словарь граней c координатами
    sides_dict = {"sides": []}
    for i, side in enumerate(line_strings_all):
        edge_dict = {"edges": []}
        for j, edge in enumerate(side):
            edge_dict[f"edges"].append(
                {f"line": {"xy": [[edge.xy[0][0], edge.xy[1][0]], [edge.xy[0][1], edge.xy[1][1]]], \
                           "centoroid": {"xy": [edge.centroid.xy[0][0], edge.centroid.xy[1][0]]}}})
        sides_dict[f"sides"].append(edge_dict)

    # Формируем словарь граней c нормальными

    # добавляем на будущее словарь центрроидов грани
    return sides_dict


import numpy as np

def distance(xs, ys):
    return np.sqrt((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2)


# def search_lowest_centroid_edges(sd):
#     centroids = []
#
#     for i in sd['sides']:
#         c = []
#         for j in i['edges']:
#             c.append(j['line']['centoroid']['xy'])
#         centroids.append(c)
#     print(centroids)
#     min_centroids = sd['sides'][0]['edges'][0]['line']['centoroid']['xy'][1]
#
#     return min_centroids
#
#
# def search_down_edge(sd):
#     lowest_edges = []
#     min_centroids = search_lowest_centroid_edges(sd)
#     for i in sd['sides']:
#         for j in i['edges']:
#             if j['line']['centoroid']['xy'][1] < min_centroids:
#                 min_centroids = j['line']['centoroid']['xy'][1]
#                 lowest_edges = j
#     return lowest_edges


def proportion_work_area():
    left_border_dist = distance(ARR_LENT_LEFT[0], ARR_LENT_LEFT[1])
    right_border_dist = distance(ARR_LENT_RIGHT[0], ARR_LENT_RIGHT[1])
    print(left_border_dist, right_border_dist, left_border_dist / right_border_dist)


if __name__ == "__main__":
    sides_dict = MOCK_SIDES_DICT
    sides_dict = json.loads(sides_dict.replace("\'", "\""))
    # print(sides_dict['sides'][0]['edges'][0])
    # TODO:  Распределение в сторон в переменные контруктор
    # print(sides_dict)
    # print(search_down_edge(sides_dict))
    box = boxwgh.Boxwgh(sides_dict)
    print(box)
    # box.front_side.down_edge = boxwgh.Edge(search_down_edge(sides_dict))
    # print(box.front_side.down_edge)
    #
    proportion_work_area()
    # # print(box.equality_edges())
    # print(box)
    # # print(.get_perspective_transform)
    #
    # print(cargo.JSON_BORDERS)
    borders = boxwgh.Borders(JSON_BORDERS)
    print(borders.draw_mesh(borders.parallel_mesh()))

    print(borders.test("bo", " "))
    print(borders.get_gabarity(box))
    borders.draw_gabarity(box)
    # print(box)

    # borders.get_perspective_transform()
# Длина (0.569м): 0.5537314142752592м; delta 1.5268585724740769см
# Ширина (0.516м): 0.5097783508300782м; delta 0.622164916992185см
# Высота (0.381м): 0.4294147491048885м; delta -4.841474910488852см
