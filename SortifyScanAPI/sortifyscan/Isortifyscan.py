import gc
import time
import traceback

import cv2
import torch
from matplotlib import pyplot as plt

from sortifyscan import ExportMedia, JSON_BORDERS, CargoProcessing, CargoAnalysis, CargoDetection, PATH_WEIGHTS_YOLO, \
    PATH_WEIGHTS_SAM, PATH_WEIGHTS_SEGYOLO9
from sortifyscan.cargo import boxwgh


class ISortifyScan:
    @staticmethod
    def clean_memory_for_gpu():
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def sortify_scan_make_video(path):
        export = ExportMedia(name_root=path)
        export.make_video(export.folder_name_collage)

    @staticmethod
    def sortify_scan(path_images):
        detection = CargoDetection(PATH_WEIGHTS_YOLO, PATH_WEIGHTS_SAM, PATH_WEIGHTS_SEGYOLO9, path_images)

        results_det = detection.detect_cargo()
        export = ExportMedia()
        for n_shot, result_det in enumerate(results_det):
            start_time = time.time()

            # if n_shot < 73: continue
            print(f"\nОбработка кадра {n_shot}\n")
            borders = boxwgh.Borders(JSON_BORDERS)
            borders.draw_image_orto(image=cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                    save_dir_path=export.folder_name_orto,
                                    name_img=n_shot)
            ExportMedia.export_images(n_shot=n_shot, img=cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                      path=export.folder_name_orig)

            try:

                if len(result_det.boxes.xyxy) == 0: raise Exception(f"No detections frame {n_shot}")

                # CargoProcessing.show_image_after_ultralytics(result_det,
                #                                              n_shot=n_shot,
                #                                              save_dir_path=export.folder_name_yolo)

                if len(result_det.boxes.xyxy) == 0:
                    continue
                json_result_detection = CargoProcessing.result_to_json(result_det)
                # print(json_result_detection)
                bbox = CargoProcessing.get_bbox_from_result(json_result_detection)
                CargoProcessing.show_image_detection(result_det, n_shot, export.folder_name_detection_bbox)
                result_seg = detection.segment_cargo_bboxes_SAM(result_det.orig_img, bbox)[0]

                ISortifyScan.clean_memory_for_gpu()
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

                if not CargoDetection.check_obj_in_area(bbox):
                    print(f"Объект не в рабочей зоне {n_shot}")
                    continue

                point_cloud = CargoAnalysis.get_xy_edges_OpenCV(result_side)

                CargoAnalysis.draw_point_cloud_edges(point_cloud)  # TODO: передать размеры изображения аргументом

                line_strings_all = CargoAnalysis.approximate_point_cloud(point_cloud)

                CargoAnalysis.draw_edges(line_strings_all,
                                         cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                         n_shot,
                                         path=export.folder_name_skeleton)

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

                print(borders.get_gabarity(box))

                borders.draw_gabarity(box,
                                      image=cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                      show=False,
                                      save_dir_path=export.folder_name_result,
                                      name_img=n_shot)

            except Exception as e:
                print(e.args)
                traceback.print_exc()
            finally:
                print("\nИтерация", n_shot, "заняла", time.time() - start_time, "секунд\n")
                export.make_collage(n_shot)
                plt.close()

        export.make_video(export.folder_name_collage)
        return box

