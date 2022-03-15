#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# File    :   homework..py
# Time    :   2022/03/14 20:58:33
# Author  :   Pu Yanheng
'''

# here put the import lib

import cv2
import numpy as np


def read_img(path, invertcolor=False, show=False):
    """
    ### Description
    Read image.

    ### Parameters
    - `path`: path of image
    - `invertcolor`: whether to invert the color of image
    - `show`: whether to show the image

    ### Returns
    Array of image.
    """

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if invertcolor:
        img = invert_color(img)
    if show:
        show_img(img)

    return img


def read_templates(invertcolor=False, show=False):
    """
    ### Description
    Read templates' images.

    ### Returns
    Array of all the templates.
    """

    temp = []
    for i in range(10):
        temp.append(
            read_img('Template matching/Train/' + str(i) + '_.jpg',
                     invertcolor=invertcolor,
                     show=show))

    return np.array(temp)


def read_tests():
    """
    ### Description
    Read test images.

    ### Returns
    Array of all the test images.
    """

    tests = []
    for i in range(10):
        tests.append(read_img('Template matching/Test/' + str(i) + '.jpg'))

    return np.array(tests)


def resize_template(temp, enlarge=True):
    """
    ### Description
    Zoom in or out the template by one pixel.

    ### Parameters
    - `temp`: the array of template image
    - `enlarge`: if enlarge is True, zoom in the picture, vice versa

    ### Returns
    Array of resized template image.
    """

    x, y = temp.shape[0:2]
    if x > y:
        ratio = x / y
        if enlarge:
            temp = cv2.resize(temp, (y + 1, int(x + ratio)))
        if not enlarge:
            temp = cv2.resize(temp, (y - 1, int(x - ratio)))
    else:
        ratio = y / x
        if enlarge:
            temp = cv2.resize(temp, (int(y + ratio), x + 1))
        if not enlarge:
            temp = cv2.resize(temp, (int(y - ratio), x - 1))

    return temp


def show_img(img, name='img'):
    """
    ### Description
    Show image from array.

    ### Parameters
    - `img`: array of image
    - `name`: name of cv2 window
    """

    cv2.imshow(name, img)
    d = cv2.waitKey(0)
    if d == ord('q'):
        cv2.destroyAllWindows()


def matching(template,
             test,
             threshold,
             template_name,
             path,
             show=False,
             save=True):
    """
    ### Description
    Sliding window and search for matching area.

    ### Parameters
    - `template`: template for matching
    - `test`: test image
    - `threshold`: threshold used to determine whether a match is made
    - `template_name`: name of the template
    - `test_name`: name of the tested image
    - `show`: whether to show image while matching
    - `save`: whether to save the result
    - `path`: where to save the result
    """

    template = np.uint32(template)
    test = np.uint32(test)
    filename = 'Template ' + template_name

    temp_x, temp_y = template.shape[0:2]
    test_x, test_y = test.shape[0:2]

    result_array = np.empty((test_x - temp_x + 1, test_y - temp_y + 1))
    possible = []

    for m in range(test_x - temp_x + 1):
        for n in range(test_y - temp_y + 1):

            window = test[m:m + temp_x, n:n + temp_y]
            product = template * window
            summ = np.sum(product)
            factor = (np.sqrt(np.sum(template * template)) *
                      np.sqrt(np.sum(window * window)))
            result = summ / factor

            result_array[m, n] = result

            if result > threshold:
                near = False
                if possible != []:
                    for index in possible:
                        if abs(m - index[0]) < 5 and abs(n - index[1]) < 5:
                            near = True
                if not near:
                    if show:
                        show_img(np.uint8(window), 'possible')
                    possible.append((m, n))

    if len(possible) != 0:

        for i, index in enumerate(possible):
            if show:
                show_img(
                    np.uint8(test[index[0]:index[0] + temp_x,
                                  index[1]:index[1] + temp_y]),
                    'Found ' + str(i))
            if save:
                save_result(np.uint8(test),
                            index[0],
                            index[1],
                            temp_x,
                            temp_y,
                            filename + ' Found ' + str(i) + '.jpg',
                            path=path)

    else:
        max_result = np.max(result_array)
        max_m, max_n = np.where(result_array == max_result)[0:2]
        max_m, max_n = int(max_m), int(max_n)
        if show:
            show_img(np.uint8(test[max_m:max_m + temp_x, max_n:max_n + temp_y]),
                     'Most likely')
        if save:
            save_result(np.uint8(test),
                        max_m,
                        max_n,
                        temp_x,
                        temp_y,
                        filename + ' Most likely.jpg',
                        path=path)


def save_result(test,
                m,
                n,
                temp_x,
                temp_y,
                filename,
                path='Template matching/Result/'):
    """
    ### Description
    Box out the matched area, and save the result.

    ### Parameters
    - `test`: array of the test image
    - `m`: begining x index of matched area
    - `n`: begining y index of matched area
    - `temp_x`: size of the template in axis x
    - `temp_y`: size of the template in axis y
    - `filename`: name of the saved file
    - `path`: directory of the file
    """

    test[m:m + temp_x - 1, n] = [255, 0, 0]
    test[m:m + temp_x - 1, n + temp_y - 1] = [255, 0, 0]
    test[m, n:n + temp_y - 1] = [255, 0, 0]
    test[m + temp_x - 1, n:n + temp_y - 1] = [255, 0, 0]

    cv2.imwrite(path + filename, test)


def match_all(templates,
              test,
              threshold,
              path,
              invertcolor=False,
              zoom=True,
              zoom_times=5):
    """
    ### Description
    Run matching process for every template on every test images.

    ### Parameters
    - `templates`: array of every template array under Train/
    - `test`: file name of the image to be matched
    - `threshold`: threshold used to determine whether a match is made
    - `zoom`: whether to enlarge size of template for matching
    - `zoom_times`: how many times the templates been enlarged
    """

    test = read_img(test, invertcolor)
    for i, template in enumerate(templates):
        if zoom:
            for j in range(zoom_times):
                matching(template, test, threshold,
                         str(i) + 'zoom ' + str(j), path)
                template = resize_template(template)
        if not zoom:
            matching(template, test, threshold, str(i), path)


def invert_color(img):
    """
    ### Description
    Invert the color of the picture.

    ### Parameters
    - `img`: array of the image

    ### Returns
    Inverted image.
    """

    return 255 - img


if __name__ == '__main__':
    templates = read_templates(invertcolor=False,show=False)
    test_img = 'Template matching/Test/5.jpg'
    result_path = 'Template matching/Result/5/'
    threshold = 0.9

    match_all(templates,
              test_img,
              threshold,
              result_path,
              invertcolor=False,
              zoom=False)
