import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import platform

from matplotlib import patches

from sortifyscan.cargo import constants
from sortifyscan.export import ExportMedia


class CargoProcessing:
    @staticmethod
    def show_image(img):
        """Метод вывода изображения np
        """
        if not constants.DEBUG: return
        plt.imshow(img)
        plt.show()

    @staticmethod
    def show_image_after_ultralytics(result: list, save_dir_path , n_shot):
        """Метод вывода изображения после применения модели
        Parameters
        ----------
        result: list
        результат после применения модели
        """
        slash = {'Windows': '\\', 'Linux': '/'}
        image = Image.open(result.save_dir + slash[platform.system()] + \
                           result.path.split(slash[platform.system()])[-1])
        plt.imshow(image)

        if save_dir_path is not None:
            ExportMedia.export_plt(n_shot, plt, save_dir_path)

        if not constants.DEBUG:
            plt.close()
            return
        plt.show()

    @staticmethod
    def show_points_after_ultralytics(result: list):
        """Метод вывода изображения после применения модели показать точки
        Parameters
        ----------
        result: list
        результат после применения модели
        """
        if not constants.DEBUG: return
        plt.figure()
        ax = plt.axes()

        plt.imshow(result.orig_img)
        print(result)
        print(result.masks)
        for xy in result.masks.xy:
            x1, y1 = np.array(xy)[:, 0], np.array(xy)[:, 1]
            ax.scatter(x1, y1, 1)
            print(np.array(xy))
        plt.show()

        # Предположим, что у вас есть набор точек объекта в формате xy

    @staticmethod
    def result_to_json(result):
        """Экспорт данных из result в json
          Parameters
          ----------
          result: list
          результат после применения модели
          ----------
          """
        # print(result)
        # for r in result:
        #     print(r.tojson())
        return json.loads(json.dumps(result[0].tojson()))

    @staticmethod
    def get_bbox_from_result(json_data):
        """Получить данные bbox из json в формате [x1,y1,x2,y2]"""
        box = json.loads(json_data)[0]['box']
        # print(list(box.values()))
        return list(box.values())

    @staticmethod
    def preparing_for_detailed_segmentation(result_seg, result_det, crop: bool = False, bgcolor: str = "white"):
        """Подготовка для сегментации боковых граней(уменьшение размерности, удаление лишних элементов)
        Возвращает изображение в формате np
        Параметр:
        """
        img = np.copy(result_seg.orig_img)
        for ci, c in enumerate(result_seg):
            # Create contour mask
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)

            # OPTION-3: черный фон
            if bgcolor == "black":
                b_mask = np.zeros(img.shape[:2], np.uint8)
                cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, img)

            # OPTION-2: прозрачный фон # Белый фон эффективнее из-за внутренних алгоритмов
            if bgcolor == "white":
                white_bg = np.ones_like(img) * 255
                cv2.drawContours(white_bg, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
                isolated = cv2.bitwise_or(white_bg, img)

            # OPTION-3: прозрачный фон на стадии распознования убирает альфа канал и изображение остается неизменным
            bbox = CargoProcessing.get_bbox_from_result(CargoProcessing.result_to_json(result_det))
            bbx1, bby1, bbx2, bby2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            iso_crop = isolated[bby1:bby2, bbx1:bbx2]

            if (crop):
                return iso_crop
            else:
                return isolated

    @staticmethod
    def show_image_detection(result_det, n_shot, save_dir_path):
        data = json.loads(CargoProcessing.result_to_json(result_det))

        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB))

        for d in data:
            print(d)
            x1, y1 = d["box"]["x1"], d["box"]["y1"]
            x2, y2 = d["box"]["x2"], d["box"]["y2"]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{d['name']} {d['class']}: {d['confidence']:.2f}", color='r', fontsize=8)
        plt.axis('off')
        ExportMedia.export_images(n_shot=n_shot,
                                  img=cv2.cvtColor(result_det.orig_img, cv2.COLOR_BGR2RGB),
                                  path=save_dir_path)
        if constants.DEBUG:
            plt.show()
        plt.close()

