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


def keep_black_pixels(image, background, front='black', threshold=50):
    """
    ### Description
    Keep all the black pixels in one image and turn others to other color.

    ### Parameters
    - `image`: array of input image
    - `background`: background color, you can choose 'white', 'green', and 'red'
    - `threshold`: threshold to confirm black

    ### Returns
    Image after process.
    """

    black = np.where(np.average(image, axis=2) <= threshold)
    not_black = np.where(np.average(image, axis=2) > threshold)
    x, y = not_black[0:2]
    for i in range(len(x)):
        if background == 'white':
            image[x[i], y[i]] = np.array([255, 255, 255])
        if background == 'green':
            image[x[i], y[i]] = np.array([0, 255, 0])
        if background == 'red':
            image[x[i], y[i]] = np.array([0, 0, 255])

    x_b, y_b = black[0:2]
    for i in range(len(x_b)):
        if front == 'black':
            image[x_b[i], y_b[i]] = np.array([0, 0, 0])
        if front == 'white':
            image[x_b[i], y_b[i]] = np.array([255, 255, 255])

    return np.uint8(image)


def matching(template,
             test,
             threshold,
             template_name,
             path,
             by_black_pixel=False,
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
    - `path`: where to save the result
    - `by_black_pixel`: whether to match by black pixels
    - `show`: whether to show image while matching
    - `save`: whether to save the result
    """

    if by_black_pixel:
        template = keep_black_pixels(template, 'green', front='white')
        show_img(template)
        # test = keep_black_pixels(test, 'green')
    template = np.uint32(template)
    parts, test, color = find_plate(test)
    test = np.uint32(test)

    filename = 'Template ' + template_name

    temp_x, temp_y = template.shape[0:2]
    test_x, test_y = test.shape[0:2]

    if temp_x > test_x:
        template = np.uint32(cv2.resize(np.uint8(template), (temp_y, test_x)))

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


def matching_by_features(template,
                         test,
                         threshold,
                         template_name,
                         path,
                         show=False,
                         save=True):
    """
    ### Description
    Sliding window and search for matching area, using divided features.

    ### Parameters
    - `template`: template for matching
    - `test`: test image
    - `threshold`: threshold used to determine whether a match is made
    - `template_name`: name of the template
    - `path`: where to save the result
    - `show`: whether to show image while matching
    - `save`: whether to save the result
    """

    template = np.uint32(template)
    test = np.uint32(test)

    filename = 'Template ' + template_name

    temp_x, temp_y = template.shape[0:2]
    test_x, test_y = test.shape[0:2]

    result_row_array = np.empty((test_x - temp_x + 1, test_y - temp_y + 1))
    result_col_array = np.empty((test_x - temp_x + 1, test_y - temp_y + 1))
    possible = []

    template_black = np.average(template, 2)
    template_black_row_avg = np.average(template_black, 1)
    template_black_col_avg = np.average(template_black, 0)

    for m in range(test_x - temp_x + 1):
        for n in range(test_y - temp_y + 1):

            window = test[m:m + temp_x, n:n + temp_y]
            window_black = np.average(window, 2)
            window_black_row_avg = np.average(window_black, 1)
            window_black_col_avg = np.average(window_black, 0)

            product_row = template_black_row_avg * window_black_row_avg
            summ_row = np.sum(product_row)
            factor_row = (
                np.sqrt(np.sum(template_black_row_avg * template_black_row_avg))
                * np.sqrt(np.sum(window_black_row_avg * window_black_row_avg)))
            result_row = summ_row / factor_row

            product_col = template_black_col_avg * window_black_col_avg
            summ_col = np.sum(product_col)
            factor_col = (
                np.sqrt(np.sum(template_black_col_avg * template_black_col_avg))
                * np.sqrt(np.sum(window_black_row_avg * window_black_row_avg)))
            result_col = summ_col / factor_col

            result_row_array[m, n] = result_row
            result_col_array[m, n] = result_col

            if result_row > threshold and result_col > threshold:
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
        max_result = np.max(result_row_array + result_col_array)
        max_m, max_n = np.where(result_row_array +
                                result_col_array == max_result)[0:2]
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
              by_feature=False,
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
                if not by_feature:
                    matching(template, test, threshold,
                             str(i) + 'zoom ' + str(j), path)
                else:
                    matching_by_features(template, test, threshold,
                                         str(i) + 'zoom ' + str(j), path)
                template = resize_template(template)
        if not zoom:
            if not by_feature:
                matching(template, test, threshold, str(i), path)
            else:
                matching_by_features(template, test, threshold, str(i), path)


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


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    #col_num_limit = self.cfg["col_num_limit"]
    row_num_limit = 21
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  #绿色有渐变
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and S > 34 and V > 46:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and S > 34 and V > 46:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


def find_waves(threshold, histogram):
    up_point = -1  #上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


def find_plate(img, resize_rate=1):

    pic_hight, pic_width = img.shape[:2]
    if pic_width > 1000:
        pic_rate = 1000 / pic_width
        img = cv2.resize(img, (1000, int(pic_hight * pic_rate)),
                         interpolation=cv2.INTER_LANCZOS4)

    if resize_rate != 1:
        img = cv2.resize(
            img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
            interpolation=cv2.INTER_LANCZOS4)
        pic_hight, pic_width = img.shape[:2]

    print("h,w:", pic_hight, pic_width)
    blur = 3
    #高斯去噪
    if blur > 0:
        img = cv2.GaussianBlur(img, (blur, blur), 0)  #图片分辨率调整
    oldimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #equ = cv2.equalizeHist(img)
    #img = np.hstack((img, equ))
    #去掉图像中不会是车牌的区域
    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)

    #找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    #使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((4, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    #查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    try:
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
    print('len(contours)', len(contours))
    #一一排除不是车牌的矩形区域
    car_contours = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        #print(wh_ratio)
        #要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if wh_ratio > 2 and wh_ratio < 5.5:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
            #cv2.imshow("edge4", oldimg)
            #cv2.waitKey(0)

    print(len(car_contours))

    print("精确定位")
    card_imgs = []
    #矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    for rect in car_contours:
        if rect[2] > -1 and rect[2] < 1:  #创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle
                )  #扩大范围，避免车牌边缘被排除

        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  #正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point,
                               new_right_point])  #字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]),
                           int(left_point[0]):int(new_right_point[0])]
            card_imgs.append(card_img)
            #cv2.imshow("card", card_img)
            #cv2.waitKey(0)
        elif left_point[1] > right_point[1]:  #负角度

            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point,
                               right_point])  #字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]),
                           int(new_left_point[0]):int(right_point[0])]
            card_imgs.append(card_img)
            #cv2.imshow("card", card_img)
            #cv2.waitKey(0)
    #开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        #有转换失败的可能，原因来自于上面矫正矩形出错
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:  #图片分辨率调整
                    yello += 1
                elif 35 < H <= 99 and S > 34:  #图片分辨率调整
                    green += 1
                elif 99 < H <= 124 and S > 34:  #图片分辨率调整
                    blue += 1

                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"

        limit1 = limit2 = 0
        if yello * 2 >= card_img_count:
            color = "yello"
            limit1 = 11
            limit2 = 34  #有的图片有色偏偏绿
        elif green * 2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 2 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124  #有的图片有色偏偏紫
        elif black + white >= card_img_count * 0.7:  #TODO
            color = "bw"
        print(color)
        colors.append(color)
        print(blue, green, yello, black, white, card_img_count)
        #cv2.imshow("color", card_img)
        #cv2.waitKey(0)
        if limit1 == 0:
            continue
        #以上为确定车牌颜色
        #以下为根据车牌颜色再定位，缩小边缘非车牌边界
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        card_imgs[card_index] = card_img[
            yl:yh,
            xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                yl - (yh - yl) // 4:yh, xl:xr]
        if need_accurate:  #可能x或y方向未缩小，需要再试一次
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = card_img[
            yl:yh,
            xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                yl - (yh - yl) // 4:yh, xl:xr]
    #以上为车牌定位

    #以下为识别车牌中的字符
    predict_result = []
    roi = None
    card_color = None
    for i, color in enumerate(colors):
        if color in ("blue", "yello", "green"):
            card_img = card_imgs[i]
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            #黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
            if color == "green" or color == "yello":
                gray_img = cv2.bitwise_not(gray_img)
            ret, gray_img = cv2.threshold(gray_img, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #查找水平直方图波峰
            x_histogram = np.sum(gray_img, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2
            wave_peaks = find_waves(x_threshold, x_histogram)
            if len(wave_peaks) == 0:
                print("peak less 0:")
                continue
            #认为水平方向，最大的波峰为车牌区域
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            gray_img = gray_img[wave[0]:wave[1]]
            #查找垂直直方图波峰
            row_num, col_num = gray_img.shape[:2]
            #去掉车牌上下边缘1个像素，避免白边影响阈值判断
            gray_img = gray_img[1:row_num - 1]
            y_histogram = np.sum(gray_img, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 5  #U和0要求阈值偏小，否则U和0会被分成两半

            wave_peaks = find_waves(y_threshold, y_histogram)

            #for wave in wave_peaks:
            #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
            #车牌字符数应大于6
            if len(wave_peaks) <= 6:
                print("peak less 1:", len(wave_peaks))
                continue

            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            #判断是否是左侧车牌边缘
            if wave_peaks[0][1] - wave_peaks[0][
                    0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)

            #组合分离汉字
            cur_dis = 0
            for i, wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i + 1:]
                wave_peaks.insert(0, wave)

            #去除车牌上的分隔点
            point = wave_peaks[2]
            if point[1] - point[0] < max_wave_dis / 3:
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

            if len(wave_peaks) <= 6:
                print("peak less 2:", len(wave_peaks))
                continue
            part_cards = seperate_card(gray_img, wave_peaks)
            for i, part_card in enumerate(part_cards):
                #可能是固定车牌的铆钉
                if np.mean(part_card) < 255 / 5:
                    print("a point")
                    continue
                part_card_old = part_card
                #w = abs(part_card.shape[1] - SZ)//2
                w = part_card.shape[1] // 3
                part_card = cv2.copyMakeBorder(part_card,
                                               0,
                                               0,
                                               w,
                                               w,
                                               cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
                #show_img(part_card_old, 'old part card')
                #show_img(part_card, 'new part card')

            roi = card_img
            show_img(roi, 'car plate')
            card_color = color
            break

    return part_cards, roi, card_color  #识别到的字符、定位的车牌图像、车牌颜色


if __name__ == '__main__':
    templates = read_templates(invertcolor=False, show=False)
    test_img = 'Template matching/Test/1.jpg'
    result_path = 'Template matching/Result/1/'
    threshold = 0.95

    match_all(templates,
              test_img,
              threshold,
              result_path,
              by_feature=False,
              invertcolor=False,
              zoom=False)