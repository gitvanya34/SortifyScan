from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
from .processing import CargoProcessing


class CargoDetection:
    def __init__(self, weights_yolo, weights_sam, path_begin_image):
        """Конструктор детекции упаковки
                 Parameters
                 ----------
                 weights_yolo: str = PATH_WEIGHTS_YOLO
                 path_begin_image: str = PATH_BEGIN_IMAGE
                 weights_sam: str = PATH_WEIGHTS_SAM
                 """
        self.path_weights_yolo = weights_yolo
        self.path_weights_sam = weights_sam
        self.path_begin_image = path_begin_image

    def detect_cargo(self, conf=0.75):
        """Метод детекции упаковки
          Parameters
          ----------
          conf: float = 0.75
          """
        model = YOLO(self.path_weights_yolo)
        result = model.predict(source=self.path_begin_image, save=True, conf=conf)
        return result

    # TODO: доделать параметры вынести модель в константу
    def segment_cargo_bboxes(self, image, bboxes):
        """Метод сегментации упаковки по bbox
                Parameters
                ----------
                # image - np.array image (result[0].orig_img)
                """
        overrides = dict(conf=0.99, task='segment', mode='predict', model=self.path_weights_sam)
        predictor = SAMPredictor(overrides=overrides)
        predictor.set_image(image)
        # bboxes = list(box.values())
        result = predictor(bboxes=bboxes)
        return result

    def segment_cargo(self, image):
        """Метод сегментации
          Parameters
          ----------
          # image - np.array image (result[0].orig_img)
          """
        overrides = dict(conf=0.99, task='segment', mode='predict', model=self.path_weights_sam, save_crop=True)
        predictor = SAMPredictor(overrides=overrides)
        predictor.set_image(image)
        result = predictor()
        return result

    def segmentation_of_the_side(self, result, crop: bool = False, bgcolor: str = "white"):
        img = CargoProcessing.preparing_for_detailed_segmentation(result, crop, bgcolor)
        CargoProcessing.show_image(img)
        result = self.segment_cargo(img)
        return result
