import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import os
from datetime import datetime


class ExportMedia:

    def __init__(self, name_root=None):
        self.path_output = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "Output")

        if name_root is None:
            self.path_folder_root, self.name_root = self.make_dirs()
        else:
            self.name_root = name_root
            self.path_folder_root = os.path.join(self.path_output, name_root)

        self.folder_name_detection_bbox = os.path.join(self.path_folder_root, "detection_bbox")
        self.folder_name_orig = os.path.join(self.path_folder_root, "orig")
        self.folder_name_sam = os.path.join(self.path_folder_root, "sam")
        self.folder_name_segmentation = os.path.join(self.path_folder_root, "segmentation")
        self.folder_name_points = os.path.join(self.path_folder_root, "points")
        self.folder_name_skeleton = os.path.join(self.path_folder_root, "skeleton")
        self.folder_name_orto = os.path.join(self.path_folder_root, "orto")
        self.folder_name_result = os.path.join(self.path_folder_root, "result")
        self.folder_name_collage = os.path.join(self.path_folder_root, "collage")

        if name_root is None:
            os.makedirs(self.folder_name_orig)
            os.makedirs(self.folder_name_detection_bbox)
            os.makedirs(self.folder_name_sam)
            os.makedirs(self.folder_name_segmentation)
            os.makedirs(self.folder_name_points)
            os.makedirs(self.folder_name_skeleton)
            os.makedirs(self.folder_name_orto)
            os.makedirs(self.folder_name_result)
            os.makedirs(self.folder_name_collage)

        self.folders = {
            self.folder_name_orig: None,
            self.folder_name_detection_bbox: None,
            self.folder_name_sam: None,
            self.folder_name_segmentation: None,
            self.folder_name_points: None,
            self.folder_name_skeleton: None,
            self.folder_name_orto: None,
            self.folder_name_result: None}

    def make_dirs(self):
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name_root = f"{time_str}"

        os.makedirs(os.path.join(self.path_output, folder_name_root))
        print(f"Создана папка с именем: {folder_name_root}")
        return os.path.join(self.path_output, folder_name_root), folder_name_root

    def make_collage(self, n_shot):

        images = []

        for folder in self.folders.keys():
            p = os.path.join(folder, f"{n_shot}.png")
            if os.path.exists(p):
                images.append(Image.open(p))
                self.folders[folder] = p
            else:
                if self.folders[folder] is not None:
                    images.append(Image.open(self.folders[folder]))
                else:
                    p = os.path.join(self.folder_name_orig, f"{n_shot}.png")
                    if os.path.exists(p):
                        images.append(Image.open(p))
                    else:
                        return

        fig, axes = plt.subplots(2, 4, figsize=(15, 15))

        for i, image in enumerate(images):
            row = i // 4  # Номер строки
            col = i % 4  # Номер столбца
            axes[row, col].imshow(image)
            axes[row, col].axis('on')

        plt.tight_layout()
        # plt.axis('auto')
        self.export_plt(n_shot, plt, self.folder_name_collage)
        plt.close()

    def make_video(self, path, fps=5, target_resolution=(2205, 1826)):
        # Путь к папке с изображениями
        images_folder = path

        # Получаем список файлов в папке
        image_files = sorted(os.listdir(images_folder), key=lambda x: int(x.split('.')[0]))
        # print(images_folder)
        # print(image_files)
        # Определяем размеры видео по первому изображению
        first_image = cv2.imread(os.path.join(images_folder, image_files[0]))
        height, width, _ = first_image.shape

        filename = os.path.join(self.path_folder_root, f"{self.path_folder_root}.mp4")
        # Инициализируем объект VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Выбираем кодек для сохранения видео
        out = cv2.VideoWriter(filename, fourcc, fps,
                              target_resolution)

        # Проходимся по каждому изображению, изменяем его размер и добавляем в видео
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            frame = cv2.imread(image_path)
            resized_frame = cv2.resize(frame, target_resolution)
            out.write(resized_frame)

        # Закрываем объект VideoWriter
        out.release()
        print(f"Сохранено видео {filename}")

    @staticmethod
    def export_plt(n_shot, plt, path):
        """Метод вывода изображения np
        """
        image_path = os.path.join(path, f'{n_shot}.png')
        plt.axis('off')
        plt.savefig(image_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"Изображение {n_shot}.png успешно сохранено в папке: {image_path}")

    @staticmethod
    def export_images(n_shot, img, path):
        """Метод вывода изображения np
        """

        plt.imshow(img)
        ExportMedia.export_plt(n_shot, plt, path)
        plt.close()
