import gc
import json
import os

import cv2
import torch
from matplotlib import pyplot as plt

# from cargo import CargoDetection, CargoAnalysis, CargoProcessing
from cargo import *
import boxwgh


def clean_memory_for_gpu():
    torch.cuda.empty_cache()
    gc.collect()


def make_dir():
    import os
    from datetime import datetime
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{time_str}"
    os.makedirs(folder_name)
    print(f"Создана папка с именем: {folder_name}")
    return folder_name


if __name__ == "__main__":
    detection = CargoDetection(PATH_WEIGHTS_YOLO, PATH_WEIGHTS_SAM, PATH_BEGIN_IMAGE)

    results_det = detection.detect_cargo()
    folder_name = make_dir()
    for n_shot, result_det in enumerate(results_det):
        try:
            CargoProcessing.show_image_after_ultralytics(result_det)

            # clean_memory_for_gpu()
            print(f'{result_det.boxes.xyxy}')
            print(f'{len(result_det.boxes.xyxy)}')
            if len(result_det.boxes.xyxy) == 0:
                continue

            bbox = CargoProcessing.get_bbox_from_result(CargoProcessing.result_to_json(result_det))
            result_seg = detection.segment_cargo_bboxes(cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB), bbox)
            CargoProcessing.show_image_after_ultralytics(result_seg[0])

            configurations = [
                {'crop': False, 'bgcolor': 'white', 'expected_length': 5},
                {'crop': False, 'bgcolor': 'black', 'expected_length': 5},
                {'crop': True, 'bgcolor': 'white', 'expected_length': 4},
                {'crop': True, 'bgcolor': 'black', 'expected_length': 4},

            ]

            for config in configurations:
                result_side = detection.segmentation_of_the_side(result_seg, result_det, crop=config['crop'],
                                                                 bgcolor=config['bgcolor'])

                CargoProcessing.show_image_after_ultralytics(result_side[0])
                if len(result_side[0].names) == config['expected_length']:
                    point_cloud = CargoAnalysis.get_xy_edges(result_side, bbox) if config['crop'] \
                        else CargoAnalysis.get_xy_edges(result_side)
                    break
            else:
                print("Нет конфигураций")
                continue


            # print(point_cloud)
            CargoAnalysis.draw_point_cloud_edges(point_cloud)  # TODO: передать размеры изображения аргументом

            line_strings_all = CargoAnalysis.approximate_point_cloud(point_cloud)
            CargoAnalysis.draw_edges(line_strings_all, cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB))

            # print(line_strings_all)
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
            print(borders.get_gabarity(box,
                                       image=cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                       show=False,
                                       save_dir_path=folder_name,
                                       name_img=n_shot))
        except Exception:
            plt.imshow(cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB))
            image_path = os.path.join(folder_name, f'{n_shot}.png')
            plt.savefig(image_path)
            print(f"Изображение {n_shot}.png успешно сохранено в папке: {image_path}")
            plt.close()
            continue

# Длина (0.569м): 0.5537314142752592м; delta 1.5268585724740769см
# Ширина (0.516м): 0.5097783508300782м; delta 0.622164916992185см
# Высота (0.381м): 0.4294147491048885м; delta -4.841474910488852см
# Длина (0.34): 0.33259536851587707м;
# Ширина (0.563): 0.5482442626953125м;
# Высота (0.221): 0.3088412796035742м;
