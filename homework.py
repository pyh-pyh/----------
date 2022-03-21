import cv2
import numpy as np


def read_img(path):

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    return img


def read_templates():

    temp = []
    for i in range(10):
        temp.append(read_img('Template matching/Train/' + str(i) + '_.jpg'))

    return np.array(temp)


def show_img(img, name='img'):

    cv2.imshow(name, img)
    d = cv2.waitKey(0)
    if d == ord('q'):
        cv2.destroyAllWindows()


def save_result(test,
                m,
                n,
                temp_x,
                temp_y,
                filename,
                path='Template matching/Result/'):

    test_rec = cv2.rectangle(test, (n, m), (n + temp_y - 1, m + temp_x - 1),
                             (0, 0, 255))

    cv2.imwrite(path + filename, test_rec)


def matching(template, test, threshold, template_name, path, invert=False):

    card = test

    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    if invert:
        card = 255 - card

    template = template / np.max(template) * 255
    card = card / np.max(card) * 255

    template = np.uint32(template)
    card = np.uint32(card)

    filename = 'Template ' + template_name

    temp_x, temp_y = template.shape[0:2]
    test_x, test_y = card.shape[0:2]

    result_array = np.empty((test_x - temp_x + 1, test_y - temp_y + 1))
    possible = []

    for m in range(test_x - temp_x + 1):
        for n in range(test_y - temp_y + 1):

            window = card[m:m + temp_x, n:n + temp_y]

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
                    possible.append((m, n))

    card = cv2.cvtColor(np.uint8(card), cv2.COLOR_GRAY2BGR)

    if len(possible) != 0:
        for i, index in enumerate(possible):
            test_save = np.uint8(card)
            save_result(test_save,
                        index[0],
                        index[1],
                        temp_x,
                        temp_y,
                        filename + ' Found ' + str(i) + '.jpg',
                        path=path)

    else:
        max_result = np.max(result_array)
        max_m, max_n = np.where(result_array == max_result)[0:2]
        max_m, max_n = int(max_m[0]), int(max_n[0])
        save_result(np.uint8(card),
                    max_m,
                    max_n,
                    temp_x,
                    temp_y,
                    filename + ' Most likely.jpg',
                    path=path)


if __name__ == '__main__':
    templates = read_templates()
    test_img = read_img('Template matching/Test/1.jpg')
    result_path = 'Template matching/testt/'
    threshold = 0.99

    for i, template in enumerate(templates):
        matching(template,
                 test_img,
                 threshold,
                 str(i),
                 result_path,
                 invert=False)
