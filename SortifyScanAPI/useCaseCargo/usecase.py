import json
import cargo


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

sides_dict = cargo.MOCK_SIDES_DICT
sides_dict = json.loads(sides_dict.replace("\'", "\""))
print(sides_dict['sides'][0]['edges'][0])
centroids = []

for i in sides_dict['sides']:
    c = []
    for j in i['edges']:
        c.append(j['line']['centoroid']['xy'])
    centroids.append(c)
print(centroids)

min_centroids = sides_dict['sides'][0]['edges'][0]['line']['centoroid']['xy'][1]
min_line = []
for i in sides_dict['sides']:
    c = []
    for j in i['edges']:
        if j['line']['centoroid']['xy'][1] < min_centroids:
            min_centroids = j['line']['centoroid']['xy'][1]
            min_line = j

print(min_line['line']['xy'][1])

import numpy as np


def distance(xs, ys):
    return np.sqrt((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2)


print(min_line['line']['centoroid']['xy'])

p_dist = distance(min_line['line']['xy'][0], min_line['line']['xy'][1])

print(p_dist)
coef_position = (min_line['line']['centoroid']['xy'][1] + 640) / 640
# print(coef_position * perspective_coef * p_dist)

centroids_sort = []
for i in centroids:
    centroids_sort.append(sorted(i, key=lambda x: x[1]))

centroids_sort = sorted(centroids_sort, key=lambda x: x[0])

print(min_line)

