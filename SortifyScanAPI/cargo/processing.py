import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
import json
import platform


class CargoProcessing:
    @staticmethod
    def show_image(img):
        """Метод вывода изображения np
        """
        plt.imshow(img)
        plt.show()

    @staticmethod
    def show_image_after_ultralytics(result: list):
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
        plt.show()

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
        for r in result:
            print(r.tojson())
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
        for r in result_seg:
            img = np.copy(cv2.cvtColor(r.orig_img, cv2.COLOR_GRAY2BGR))
            img_name = Path(r.path).stem

            # iterate each object contour
            for ci, c in enumerate(r):
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
                # isolated = np.dstack([img, b_mask])

                bbox = CargoProcessing.get_bbox_from_result(CargoProcessing.result_to_json(result_det))
                bbx1, bby1, bbx2, bby2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                iso_crop = isolated[bby1:bby2, bbx1:bbx2]
                # _ = cv2.imwrite(f'segcrop{ci}.png', isolated)
                # print(f'segcrop{ci}.png сохранено на диск')

                if (crop):
                    return iso_crop
                else:
                    return isolated
