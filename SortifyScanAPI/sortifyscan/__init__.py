import gc
import time
import traceback

import cv2
import torch
from matplotlib import pyplot as plt

from sortifyscan.cargo import *
from sortifyscan.export import ExportMedia
from sortifyscan.Isortifyscan import ISortifyScan

if __name__ == '__main__':
    ISortifyScan.sortify_scan("D:\\StudentData\\Project\\Program\\sortifyscan\\ConveyorEmulator\\render\\0000-0105.avi")
    ISortifyScan.sortify_scan_make_video("2024-05-04_23-29-28")

# Длина (0.569м): 0.5537314142752592м; delta 1.5268585724740769см
# Ширина (0.516м): 0.5097783508300782м; delta 0.622164916992185см
# Высота (0.381м): 0.4294147491048885м; delta -4.841474910488852см
# Длина (0.34): 0.33259536851587707м;
# Ширина (0.563): 0.5482442626953125м;
# Высота (0.221): 0.3088412796035742м;