import gc
import json
import os

import cv2
import torch
from matplotlib import pyplot as plt

# from cargo import CargoDetection, CargoAnalysis, CargoProcessing
from cargo import *
import boxwgh
from export import ExportMedia


def clean_memory_for_gpu():
    torch.cuda.empty_cache()
    gc.collect()


def case2():
    # # Применение кластеризации методом k-средних
    # pixels = image.reshape((-1, 3))
    # pixels = np.float32(pixels)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1)
    # k = 4
    # _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # centers = np.uint8(centers)
    # segmented_image = centers[labels.flatten()]
    # segmented_image = segmented_image.reshape(image.shape)

    image = cv2.imread(
        'D:\\StudentData\\Project\\Program\\SortifyScan\\ConveyorEmulator\\render\\untitled — копия (2).png')  # Загрузка в оттенках серого
    # Применение пороговой сегментации
    # for i in range(0,255):
    #     print(i)

    # cv2.imwrite(f'D:\\StudentData\\Project\\Program\\SortifyScan\\runs\\TESTCV\\{i}.png',
    #            cv2.threshold(image, i, 255, cv2.THRESH_BINARY)[1])


# Загрузка изображения
def check_obj_in_area(bbox):
    return min(ARR_LENT_UP[:, 1]) < bbox[1] < max(ARR_LENT_DOWN[:, 1]) and min(ARR_LENT_UP[:, 1]) < bbox[1] < max(ARR_LENT_DOWN[:, 1])



def case1():
    detection = CargoDetection(PATH_WEIGHTS_YOLO, PATH_WEIGHTS_SAM, PATH_WEIGHTS_SEGYOLO9, PATH_BEGIN_IMAGE)

    results_det = detection.detect_cargo()
    export = ExportMedia()
    for n_shot, result_det in enumerate(results_det):
        print(f"\nОбработка кадра {n_shot}\n")
        # try:
        #     if len(result_det.boxes.xyxy) == 0: raise Exception(f"No detections frame {n_shot}")

        # CargoProcessing.show_image_after_ultralytics(result_det,
        #                                              n_shot=n_shot,
        #                                              save_dir_path=export.folder_name_yolo)
        ExportMedia.export_images(n_shot=n_shot, img=cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                  path=export.folder_name_root)
        if len(result_det.boxes.xyxy) == 0:
            continue
        json_result_detection = CargoProcessing.result_to_json(result_det)
        print(json_result_detection)
        bbox = CargoProcessing.get_bbox_from_result(json_result_detection)
        check_obj_in_area(bbox)
        CargoProcessing.show_image_detection(result_det, n_shot, export.folder_name_yolo)
        result_seg = detection.segment_cargo_bboxes_SAM(result_det.orig_img, bbox)[0]

        clean_memory_for_gpu()
        CargoProcessing.show_image_after_ultralytics(result_seg,
                                                     n_shot=n_shot,
                                                     save_dir_path=export.folder_name_sam)

        result_side = detection.segmentation_of_the_side(result_seg,
                                                         result_det,
                                                         False,
                                                         "white",
                                                         n_shot=n_shot,
                                                         path_dir_segment=export.folder_name_segmentation,
                                                         path_dir_points=export.folder_name_points,
                                                         )

        if not check_obj_in_area(bbox):
            print(f"Объект не в рабочей зоне {n_shot}")
            continue

        point_cloud = CargoAnalysis.get_xy_edges_OpenCV(result_side)

        CargoAnalysis.draw_point_cloud_edges(point_cloud)  # TODO: передать размеры изображения аргументом

        line_strings_all = CargoAnalysis.approximate_point_cloud(point_cloud)

        CargoAnalysis.draw_edges(line_strings_all,
                                 cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                 n_shot,
                                 export.folder_name_bbox, )

        # Формируем словарь граней c координатами
        sides_dict = {"sides": []}
        for i, side in enumerate(line_strings_all):
            edge_dict = {"edges": []}
            for j, edge in enumerate(side):
                edge_dict[f"edges"].append(
                    {f"line": {"xy": [[edge.xy[0][0], -edge.xy[1][0]], [edge.xy[0][1], -edge.xy[1][1]]],
                               "centoroid": {"xy": [edge.centroid.xy[0][0], -edge.centroid.xy[1][0]]}}})
            sides_dict[f"sides"].append(edge_dict)
        box = boxwgh.Boxwgh(sides_dict)
        borders = boxwgh.Borders(JSON_BORDERS)
        print(borders.get_gabarity(box,
                                   image=cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                   show=False,
                                   save_dir_path=export.folder_name_result,
                                   name_img=n_shot))
        # except Exception as e:
        #     print(e.args)
        #     plt.close()
        #     continue


if __name__ == "__main__":
    case1()

# Длина (0.569м): 0.5537314142752592м; delta 1.5268585724740769см
# Ширина (0.516м): 0.5097783508300782м; delta 0.622164916992185см
# Высота (0.381м): 0.4294147491048885м; delta -4.841474910488852см
# Длина (0.34): 0.33259536851587707м;
# Ширина (0.563): 0.5482442626953125м;
# Высота (0.221): 0.3088412796035742м;
