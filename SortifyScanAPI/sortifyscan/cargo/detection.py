import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

from sortifyscan.export import ExportMedia
from . import constants
from .processing import CargoProcessing


class CargoDetection:
    def __init__(self, weights_yolo, weights_sam, weights_segyolo, path_begin_image):
        """Конструктор детекции упаковки
                 Parameters
                 ----------
                 weights_yolo: str = PATH_WEIGHTS_YOLO
                 path_begin_image: str = PATH_BEGIN_IMAGE
                 weights_sam: str = PATH_WEIGHTS_SAM
                 """
        self.path_weights_yolo = weights_yolo
        self.path_weights_sam = weights_sam
        self.path_weights_segyolo = weights_segyolo
        self.path_begin_image = path_begin_image

    def detect_cargo(self, image=None, conf=0.75):
        """Метод детекции упаковки
          Parameters
          ----------
          conf: float = 0.75
          """
        if image is None:
            image = self.path_begin_image
        model = YOLO(self.path_weights_yolo)
        result = model.predict(source=image,
                               device='cpu',
                               stream=True,
                               save=True,
                               conf=conf)
        return result

    # TODO: доделать параметры вынести модель в константу
    def segment_cargo_bboxes_SAM(self, image, bboxes):
        """Метод сегментации упаковки по bbox
                Parameters
                ----------
                # image - np.array image (result[0].orig_img)
                """
        overrides = dict(conf=1,
                         # save_crop=True,
                         task='segment',
                         mode='predict',
                         imgsz=640,
                         model=self.path_weights_sam,
                         device='cpu',
                         )
        predictor = SAMPredictor(overrides=overrides, )
        predictor.set_image(image)

        result = predictor(
            bboxes=bboxes,
            # points=[[293, 323], [404, 70]],
            # labels = [[0],[1],[0]],
            # multimask_output=True,
            # masks =

        )

        return result

    def segment_cargo_SAM(self, image):
        """Метод сегментации
          Parameters
          ----------
          # image - np.array image (result[0].orig_img)
          """
        overrides = dict(conf=1,
                         task='segment',
                         mode='predict',
                         imgsz=640,
                         model=self.path_weights_sam,
                         save_crop=True,
                         device='cpu'
                         )
        predictor = SAMPredictor(overrides=overrides)
        predictor.set_image(image)
        result = predictor()

        # model = SAM(model=self.path_weights_sam)
        # result = model(image, conf=0.99, task='segment', mode='predict', imgsz=640 )
        return result

    def segment_cargo_OpenCV(self, image, n_shot, path_segment, path_points):
        start_time_findcounturs = time.time()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sorted_contours = []

        path_segment_n = os.path.join(path_segment, f"{n_shot}")
        os.makedirs(path_segment_n)

        path_points_n = os.path.join(path_points, f"{n_shot}")
        os.makedirs(path_points_n)
        # Применение фильтра Собеля для обнаружения границ TOP
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        for thresh in range(254, 255, 20):

            _, segmented_image = cv2.threshold(gradient_magnitude, thresh, 255, cv2.THRESH_BINARY)

            ExportMedia.export_images(n_shot=thresh, img=segmented_image, path=path_segment_n)
            ExportMedia.export_images(n_shot=n_shot, img=segmented_image, path=path_segment)

            contours, hierarchy = cv2.findContours(segmented_image.astype('uint8'), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)
            contour_image = np.zeros_like(segmented_image)
            cv2.drawContours(contour_image, contours, -1, color=255, thickness=1)
            white_pixels = np.where(contour_image == 255)
            # # plt.scatter(white_pixels[1], white_pixels[0], c='black', s=1)
            plt.gca().invert_yaxis()
            # sorted_contours = sorted(contours, key=lambda x: -len(x))
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # print(
            #     f"\nНайдено {len(sorted_contours)} контуров, Количество точек в контурах: {[len(contour) for contour in sorted_contours]}\n")
            # TODO написать критерии возврата в работу

            print("\n Поиск контуров занял ", time.time() - start_time_findcounturs, " секунд\n")
            for c in sorted_contours:
                if len(c) > 150:
                    plt.plot(c[:, 0, 0], c[:, 0, 1], linewidth=1)
            #  elif len(sorted_contours) == 3:
            #     plt.plot(sorted_contours[1][:, 0, 0], sorted_contours[1][:, 0, 1], linewidth=1, c='blue')
            #     plt.plot(sorted_contours[2][:, 0, 0], sorted_contours[2][:, 0, 1], linewidth=1, c='red')
            #     plt.plot(sorted_contours[0][:, 0, 0], sorted_contours[0][:, 0, 1], linewidth=1, c='green')
            # else:
            #     for contour in sorted_contours:
            #         plt.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=1, c='red')

            ExportMedia.export_plt(n_shot=thresh, plt=plt, path=path_points_n)
            ExportMedia.export_plt(n_shot=n_shot, plt=plt, path=path_points)

            if constants.DEBUG:
                plt.axis('off')
                plt.imshow(image)
                plt.show()

            plt.close()

        # plt.plot(sorted_contours[0][:, 0, 0], sorted_contours[0][:, 0, 1], linewidth=1, color="red")
        # plt.plot(sorted_contours[1][:, 0, 0], sorted_contours[1][:, 0, 1], linewidth=1, color="green")
        # plt.plot(sorted_contours[2][:, 0, 0], sorted_contours[2][:, 0, 1], linewidth=1, color="blue")
        # plt.plot(sorted_contours[3][:, 0, 0], sorted_contours[3][:, 0, 1], linewidth=1, color="orange")
        #
        # plt.show()
        #
        return sorted_contours[1:] # исключили первый элемент
        # if len(sorted_contours) > 3:
        #     return [sorted_contours[1], sorted_contours[2], sorted_contours[3]]
        # elif len(sorted_contours) == 3:
        #     return [sorted_contours[1], sorted_contours[2], sorted_contours[0]]
        # else:
        #     return [[], [], []]

    def segmentation_of_the_side(self,
                                 result_seg,
                                 result_det,
                                 crop: bool = False,
                                 bgcolor: str = "white",
                                 n_shot=None,
                                 path_dir_segment=None,
                                 path_dir_points=None,
                                 ):
        start_time_segmentation = time.time()
        img = CargoProcessing.preparing_for_detailed_segmentation(result_seg, result_det, crop, bgcolor)
        print("\n подготовка к детальной сегментации заняла", time.time() - start_time_segmentation, " секунд\n")

        CargoProcessing.show_image(img)

        # result = self.segment_cargo_SAM(img)
        result = self.segment_cargo_OpenCV(img,
                                           n_shot,
                                           path_dir_segment,
                                           path_dir_points)
        return result

    @staticmethod
    def check_obj_in_area(bbox):
        return max(constants.ARR_LENT_UP[:, 1]) < bbox[1] < min(constants.ARR_LENT_DOWN[:, 1]) and max(
            constants.ARR_LENT_UP[:, 1]) < bbox[3] < min(
            constants.ARR_LENT_DOWN[:, 1])


