import os
import numpy as np
import privateconstants

DEBUG = False

# PATH_BEGIN_IMAGE = privateconstants.PATH_PRIVATE_BEGIN_IMAGE
PATH_WEIGHTS_YOLO = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "SortifyScanAPI", "sortifyscan",
                                 "weights", "YOLO8best.pt")
# "/content/SortifyScan/SortifyScanAPI/sortifyscan/weights/YOLO8best.pt"
# os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "sortifyscan", "weights", "YOLO8best.pt"))
# privateconstants.PATH_PRIVATE_WEIGHTS_YOLO
PATH_WEIGHTS_SAM = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "SortifyScanAPI", "sortifyscan",
                                "weights", "mobile_sam.pt")
# "/content/SortifyScan/SortifyScanAPI/sortifyscan/weights/mobile_sam.pt"
# os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "sortifyscan", "weights", "mobile_sam.pt"))
#privateconstants.PATH_PRIVATE_WEIGHTS_SAM
# print(PATH_WEIGHTS_YOLO)
# print(PATH_WEIGHTS_SAM)

PATH_WEIGHTS_FASTSAM = "weights/FastSAM-s.pt"  #privateconstants.PATH_PRIVATE_WEIGHTS_FASTSAM
PATH_WEIGHTS_SEGYOLO9 = "weights/yolov9c-seg.pt"  #privateconstants.PATH_PRIVATE_WEIGHTS_SEGYOLO9
# PATH_BEGIN_IMAGE = "/content/drive/MyDrive/images/render1.png"
# PATH_WEIGHTS_YOLO = '/content/drive/MyDrive/yolov8l.pt 2.0 /runs4 best 0.751 /detect/train4/weights/YOLO9best.pt'
# PATH_WEIGHTS_SAM = '/content/drive/MyDrive/SAM/sam_b.pt'

# Размеры ленты калиборованные реальные в метрах
LENT_REAL_LEFT = 2.71
LENT_REAL_RIGHT = 2.71
LENT_REAL_UP = 0.8
LENT_REAL_DOWN = 0.8
LENT_REAL_WIGHT = 0.8
LENT_REAL_CAMERA_LEN = 1.10854

# Откалиброванные границы ленты пиксели ленты, легко калибруются с клиента
# ARR_LENT_LEFT = np.array([[165, 18], [188, 577]])
# ARR_LENT_RIGHT = np.array([[332, 9], [638, 509]])
# ARR_LENT_UP = np.array([[165, 18], [332, 9]])
# ARR_LENT_DOWN = np.array([[188, 577], [638, 509]])
ARR_LENT_LEFT = np.array([[166, 30], [187, 557]])
ARR_LENT_RIGHT = np.array([[343, 21], [638, 491]])
ARR_LENT_UP = np.array([[166, 30], [343, 21]])
ARR_LENT_DOWN = np.array([[187, 557], [638, 491]])

JSON_BORDERS = {'left': {'line': {'xy': ARR_LENT_LEFT, 'length': LENT_REAL_LEFT}},
                'right': {'line': {'xy': ARR_LENT_RIGHT, 'length': LENT_REAL_RIGHT}},
                'up': {'line': {'xy': ARR_LENT_UP, 'length': LENT_REAL_UP}},
                'down': {'line': {'xy': ARR_LENT_DOWN, 'length': LENT_REAL_DOWN}},
                }

MOCK_SIDES_DICT = """{'sides': [{'edges': [{'line': {'xy': [[411.40599614753864,
        59.52388654678937],
       [212.08042401421156, 79.90945642406146]],
      'centoroid': {'xy': [311.7432100808751, 69.71667148542542]}}},
    {'line': {'xy': [[212.08042401421156, 79.90945642406146],
       [269.0914306640625, 187.09014892578125]],
      'centoroid': {'xy': [240.58592733913704, 133.49980267492134]}}},
    {'line': {'xy': [[269.0914306640625, 187.09014892578125],
       [512.0914306640625, 153.09014892578125]],
      'centoroid': {'xy': [390.5914306640625, 170.09014892578125]}}},
    {'line': {'xy': [[512.0914306640625, 153.09014892578125],
       [411.40599614753864, 59.52388654678937]],
      'centoroid': {'xy': [461.7487134058006, 106.3070177362853]}}}]},
  {'edges': [{'line': {'xy': [[217.33673057453697, 81.43750792309548],
       [229.0914306640625, 213.09014892578125]],
      'centoroid': {'xy': [223.21408061929975, 147.26382842443837]}}},
    {'line': {'xy': [[229.0914306640625, 213.09014892578125],
       [275.0914306640625, 334.09014892578125]],
      'centoroid': {'xy': [252.0914306640625, 273.59014892578125]}}},
    {'line': {'xy': [[275.0914306640625, 334.09014892578125],
       [268.0914306640625, 187.09014892578125]],
      'centoroid': {'xy': [271.5914306640625, 260.59014892578125]}}},
    {'line': {'xy': [[268.0914306640625, 187.09014892578125],
       [217.33673057453697, 81.43750792309548]],
      'centoroid': {'xy': [242.71408061929975, 134.26382842443837]}}}]},
  {'edges': [{'line': {'xy': [[507.6168803387659, 155.59508577997184],
       [269.0914306640625, 189.09014892578125]],
      'centoroid': {'xy': [388.3541555014142, 172.34261735287654]}}},
    {'line': {'xy': [[269.0914306640625, 189.09014892578125],
       [276.0914306640625, 336.09014892578125]],
      'centoroid': {'xy': [272.5914306640625, 262.59014892578125]}}},
    {'line': {'xy': [[276.0914306640625, 336.09014892578125],
       [482.0914306640625, 301.09014892578125]],
      'centoroid': {'xy': [379.0914306640625, 318.59014892578125]}}},
    {'line': {'xy': [[482.0914306640625, 301.09014892578125],
       [507.6168803387659, 155.59508577997184]],
      'centoroid': {'xy': [494.8541555014142, 228.34261735287652]}}}]}]}"""
MOCK_SIDES_DICT_MINUS = """{'sides': [{'edges': [{'line': {'xy': [[411.40599614753864,
        -59.52388654678937],
       [212.08042401421156, -79.90945642406146]],
      'centoroid': {'xy': [311.7432100808751, -69.71667148542542]}}},
    {'line': {'xy': [[212.08042401421156, -79.90945642406146],
       [269.0914306640625, -187.09014892578125]],
      'centoroid': {'xy': [240.58592733913704, -133.49980267492134]}}},
    {'line': {'xy': [[269.0914306640625, -187.09014892578125],
       [512.0914306640625, -153.09014892578125]],
      'centoroid': {'xy': [390.5914306640625, -170.09014892578125]}}},
    {'line': {'xy': [[512.0914306640625, -153.09014892578125],
       [411.40599614753864, -59.52388654678937]],
      'centoroid': {'xy': [461.7487134058006, -106.3070177362853]}}}]},
  {'edges': [{'line': {'xy': [[217.33673057453697, -81.43750792309548],
       [229.0914306640625, -213.09014892578125]],
      'centoroid': {'xy': [223.21408061929975, -147.26382842443837]}}},
    {'line': {'xy': [[229.0914306640625, -213.09014892578125],
       [275.0914306640625, -334.09014892578125]],
      'centoroid': {'xy': [252.0914306640625, -273.59014892578125]}}},
    {'line': {'xy': [[275.0914306640625, -334.09014892578125],
       [268.0914306640625, -187.09014892578125]],
      'centoroid': {'xy': [271.5914306640625, -260.59014892578125]}}},
    {'line': {'xy': [[268.0914306640625, -187.09014892578125],
       [217.33673057453697, -81.43750792309548]],
      'centoroid': {'xy': [242.71408061929975, -134.26382842443837]}}}]},
  {'edges': [{'line': {'xy': [[507.6168803387659, -155.59508577997184],
       [269.0914306640625, -189.09014892578125]],
      'centoroid': {'xy': [388.3541555014142, -172.34261735287654]}}},
    {'line': {'xy': [[269.0914306640625, -189.09014892578125],
       [276.0914306640625, -336.09014892578125]],
      'centoroid': {'xy': [272.5914306640625, -262.59014892578125]}}},
    {'line': {'xy': [[276.0914306640625, -336.09014892578125],
       [482.0914306640625, -301.09014892578125]],
      'centoroid': {'xy': [379.0914306640625, -318.59014892578125]}}},
    {'line': {'xy': [[482.0914306640625, -301.09014892578125],
       [507.6168803387659, -155.59508577997184]],
      'centoroid': {'xy': [494.8541555014142, -228.34261735287652]}}}]}]}"""
