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
    detection = CargoDetection(PATH_WEIGHTS_YOLO, PATH_WEIGHTS_SAM, PATH_BEGIN_IMAGE)

    result_det = detection.detect_cargo()
    CargoProcessing.show_image_after_ultralytics(result_det)

    # clean_memory_for_gpu()

    bbox = CargoProcessing.get_bbox_from_result(CargoProcessing.result_to_json(result_det))
    result_seg = detection.segment_cargo_bboxes(result_det[0].orig_img, bbox)
    CargoProcessing.show_image_after_ultralytics(result_seg)

    result_side = detection.segmentation_of_the_side(result_seg, result_det, crop=True, bgcolor='black')
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


if __name__ == "__main__":
    detection = CargoDetection(PATH_WEIGHTS_YOLO, PATH_WEIGHTS_SAM, PATH_BEGIN_IMAGE)

    result_det = detection.detect_cargo()
    CargoProcessing.show_image_after_ultralytics(result_det)

    # clean_memory_for_gpu()

    bbox = CargoProcessing.get_bbox_from_result(CargoProcessing.result_to_json(result_det))
    result_seg = detection.segment_cargo_bboxes(result_det[0].orig_img, bbox)
    CargoProcessing.show_image_after_ultralytics(result_seg)

    result_side = detection.segmentation_of_the_side(result_seg, result_det, crop=True, bgcolor='black')
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
                {f"line": {"xy": [[edge.xy[0][0], -edge.xy[1][0]], [edge.xy[0][1], -edge.xy[1][1]]],
                           "centoroid": {"xy": [edge.centroid.xy[0][0], -edge.centroid.xy[1][0]]}}})
        sides_dict[f"sides"].append(edge_dict)

    # sides_dict = MOCK_SIDES_DICT
    # sides_dict = json.loads(sides_dict.replace("\'", "\""))

    box = boxwgh.Boxwgh(sides_dict)
    # print(box)

    borders = boxwgh.Borders(JSON_BORDERS)
    # print(borders.draw_mesh(borders.parallel_mesh()))

    # print(borders.test("bo", " "))
    print(borders.get_gabarity(box))
    borders.draw_gabarity(box)

# Длина (0.569м): 0.5537314142752592м; delta 1.5268585724740769см
# Ширина (0.516м): 0.5097783508300782м; delta 0.622164916992185см
# Высота (0.381м): 0.4294147491048885м; delta -4.841474910488852см
