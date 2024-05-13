import json

import cv2

from sortifyscan import cargo
from privateconstants import PATH_PRIVATE_IMAGE
from sortifyscan.cargo import boxwgh


def test_search_upper_centroid_edges():
    assert False


def test_search_lowest_centroid_edges():
    sides_dict = cargo.MOCK_SIDES_DICT
    sides_dict = json.loads(sides_dict.replace("\'", "\""))

    box = boxwgh.Boxwgh(sides_dict)
    borders = boxwgh.Borders(cargo.JSON_BORDERS)
    borders.get_gabarity(box)
    # print(box)

    image = cv2.imread(PATH_PRIVATE_IMAGE)
    borders.draw_image_orto(image)
    # Длина (0.569м): 0.5764242231926985м; delta -0.7424223192698531см
    # Ширина (0.516м): 0.51229052734375м; delta 0.370947265624999см
    # Высота (0.381м): 0.44726728054781617м; delta -6.6267280547816165см

    # Длина (0.569м): 0.6021872559228533м; delta -3.318725592285332см
    # Ширина (0.516м): 0.5030335693359375м; delta 1.2966430664062556см
    # Высота (0.381м): 0.4510009440599383м; delta -7.000094405993829см
    assert False


def test_search_down_edge():
    assert False


def test_equality_edges_length():
    assert False


def test_distribution_edge():
    assert False
