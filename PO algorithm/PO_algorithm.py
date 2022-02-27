# -*- coding: utf-8 -*-
import scipy.misc
import numpy as np
import cv2
import copy
import imageio

"""update the fix_position; add the between near into this predict to cover more situation"""

imsave = imageio.imsave
imread = imageio.imread


class Graph_Optimize:
    def __init__(self, light_img, obj_p, conv_strides=(1, 1), rotate_step=15, shrink=1.0, pool=False, dichotomy=False):
        self.light_img = light_img
        self.obj_p = obj_p
        self.rotate_step = rotate_step
        self.conv_strides = conv_strides
        self.shrink = shrink
        # shrink the picked object to add distance after placing
        self.obj_shrink = cv2.resize(obj_p, (int(obj_p.shape[1] * shrink), int(obj_p.shape[0] * shrink)))

        self.conv_ans, self.angle_history, self.pooled, self.kernel = [[] for _ in range(4)]

        self.pool = pool
        self.dichotomy = dichotomy
        pass

    def conv_with_obj(self):
        """
        add convolution in image with object
        can use conv_value to realise the speed up
        """
        best_position = [0, 0]
        if self.dichotomy:
            # ------ add rotating ------
            stop = False
            while not stop:
                self.kernel, angle, stop, seq = rotating_d(self.obj_shrink, self.conv_ans, self.angle_history)
                # ---------
                best_position, conv_value = self.conv(pool=self.pool)
                if seq[0] == 1:
                    self.angle_history.append(angle)
                    self.conv_ans.append(conv_value)
                else:
                    self.angle_history.insert(seq[1] + 1, angle)
                    self.conv_ans.insert(seq[1] + 1, conv_value)
            # ------ finish rotating --------
            placed_image = self.light_img
            best_kernel = rotating(self.obj_p, angle)
            best_angle = angle
        else:
            max_angle, best_conv_value, best_angle = 0, 0, 0
            while max_angle < 360:
                self.kernel = rotating(self.obj_shrink, angle=max_angle)
                position, conv_value = self.conv(pool=self.pool)
                if best_conv_value < conv_value:
                    best_position = position
                    best_angle = max_angle
                max_angle += self.rotate_step
            # ------ finish rotating --------
            placed_image = self.light_img
            best_kernel = rotating(self.obj_p, best_angle)

        self.save_v_place(best_position, best_kernel)  # save the visual placement image
        print('The best place placement is:', best_position)
        print('The best place orientation is:', best_angle, 'degree')
        return best_position, best_angle, placed_image

    def pooling(self):
        """
        pooling the lighting_image to subscribe calculation
        using conv kernel as pooling kernel
        """
        image = copy.deepcopy(self.light_img)

        imsave('result/before_pool.jpg', image)
        kernel = self.kernel
        pool_size = kernel.shape
        strides = [int(pool_size[0] / 2.), int(pool_size[1] / 2.)]
        pooled = []
        h, x = 0, 0
        while x + pool_size[0] < image.shape[0]:
            y = 0
            while y + pool_size[1] < image.shape[1]:
                pooling = np.sum(np.multiply(image[x:x + pool_size[0], y:y + pool_size[1]], kernel))
                pooled.append(pooling)
                y += strides[1]
            x += strides[0]
            h += 1
        pooled = np.reshape(pooled, (h, int(len(pooled) / h)))
        self.pooled = pooled
        imsave('result/pools.jpg', pooled)

    def conv(self, pool=True):
        """using conv to get the best place to put"""
        if pool:
            self.pooling()
            kernel = self.kernel
            pool_img = self.pooled
            pool_size = kernel.shape
            pool_strides = [int(pool_size[0] / 2.), int(pool_size[1] / 2.)]

            conv_h = kernel.shape[0]
            conv_w = kernel.shape[1]
            # --------------------------------------------
            pooled = pool_img
            # ----------- create conv area ---------------
            search_p = np.where(pooled == np.max(pooled))
            area = [max(((search_p[0][0] * pool_strides[0]) - conv_h), 0),
                    min(((search_p[0][0] * pool_strides[0] + pool_size[0]) + conv_h), self.light_img.shape[0]),
                    max(((search_p[1][0] * pool_strides[1]) - conv_w), 0),
                    min(((search_p[1][0] * pool_strides[1] + pool_size[1]) + conv_w), self.light_img.shape[0])]
            search_area = self.light_img[area[0]:area[1], area[2]:area[3]]
            new_h = int((search_area.shape[0] - conv_h) / self.conv_strides[0] + 1)
            new_w = int((search_area.shape[1] - conv_w) / self.conv_strides[1] + 1)
            conv_imagine = np.zeros((new_h, new_w))
            for i in range(new_h):
                for j in range(new_w):
                    conv_imagine[i, j] = np.sum(np.multiply(search_area[i:(i + conv_h), j:(j + conv_w)], kernel))
            # ------------ find the best position -----------
            best_conv = np.where(conv_imagine == np.max(conv_imagine))
            best_heart_position = [area[0] + best_conv[0][0] * self.conv_strides[0],
                                   area[2] + best_conv[1][0] * self.conv_strides[1]]
        else:
            kernel = self.kernel
            conv_h = kernel.shape[0]
            conv_w = kernel.shape[1]

            search_area = self.light_img[0:, 0:]
            new_h = int((search_area.shape[0] - conv_h) / self.conv_strides[0] + 1)
            new_w = int((search_area.shape[1] - conv_w) / self.conv_strides[1] + 1)
            conv_imagine = np.zeros((new_h, new_w))
            for i in range(new_h):
                for j in range(new_w):
                    conv_imagine[i, j] = np.sum(np.multiply(search_area[i:(i + conv_h), j:(j + conv_w)], kernel))
            # ------------ find the best position -----------
            best_conv = np.where(conv_imagine == np.max(conv_imagine))
            best_heart_position = [best_conv[0][0] * self.conv_strides[0],
                                   best_conv[1][0] * self.conv_strides[1]]
        return best_heart_position, np.max(conv_imagine)

    def save_v_place(self, best_position, best_kernel):
        """save the visual placement image"""
        best_position = [best_position[0] + int(best_kernel.shape[0] * (self.shrink - 1) / 2.0),
                         best_position[1] + int(best_kernel.shape[1] * (self.shrink - 1) / 2.0)]
        self.light_img[best_position[0]:int(best_position[0] + best_kernel.shape[0]),
        best_position[1]:int(best_position[1] + best_kernel.shape[1])] = \
            np.add(self.light_img[best_position[0]:int(best_position[0] + best_kernel.shape[0]),
                   best_position[1]:int(best_position[1] + best_kernel.shape[1])], best_kernel * 100)
        imsave('result/v_place.jpg', self.light_img)


def rotating_d(img, conv_ans, history):
    """
    to find the best rotating angle
    """
    # ------ find angle -------
    stop = False
    seq = [1, 0]
    if len(history) == 0:
        angle = 0
    elif 1 <= len(history) < 4:
        angle = history[-1] + 90
    else:
        best_conv = np.where(conv_ans == np.max(conv_ans))[0][0]
        b1 = best_conv - 1
        b2 = best_conv + 1
        if best_conv == len(conv_ans) - 1:
            b2 = 0
        if best_conv == 0:
            b1 = len(conv_ans) - 1
        if conv_ans[b1] > conv_ans[b2]:
            second_conv = b1
        else:
            second_conv = b2
        seq[0] = 0
        seq[1] = min(best_conv, second_conv)
        angle = int((history[best_conv] + history[second_conv]) / 2)
        if abs(history[best_conv] - history[second_conv]) > 100:
            angle = int((history[best_conv] + history[second_conv]) / 2) + 180
            seq = [1, 0]
    if len(history) > 5:
        if angle in history:
            stop = True
    # ------ rotate obj_p -------
    return rotating(img, angle), angle, stop, seq


def rotating(img, angle):
    """rotating image to build kernel cluster"""
    print('testing angle:', angle, 'degree')
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH))


def lighting_img(img, xy):
    """create lighting point and add it into img"""
    cir = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    for a in range(cir.shape[0]):
        for b in range(cir.shape[1]):
            A = abs(int(cir.shape[0] / 2) - a + 1)
            B = abs(int(cir.shape[1] / 2) - b + 1)
            cir[a][b] = (((cir.shape[0] + cir.shape[1]) / 2 - (A ** 2 + B ** 2) ** 0.5) * 0.05) ** 2
            b += 1
        a += 1
    return cir[(img.shape[0] - xy[0]):(2 * img.shape[0] - xy[0]),
           (img.shape[1] - xy[1]):(2 * img.shape[1] - xy[1])] * img


def change_img(v_grey, relation):
    """
    lighting and padding image, extract the obj_p
    """
    change_grey = v_grey
    # ------- padding top grey -------
    size = change_grey.shape
    pad_x, pad_y = int(0.2 * size[0]), int(0.2 * size[1])
    img_p = np.ones((size[0] + pad_x, size[1] + pad_y)) * -100
    img_p[int(pad_x / 2):int(pad_x / 2) + size[0], int(pad_y / 2):int(pad_y / 2) + size[1]] = change_grey
    # ------- padding top view -------

    # ------- extract the obj_p --------
    x, y = np.where(img_p == 10)
    t_size = [np.min(x), np.max(x), np.min(y), np.max(y)]
    obj_p = copy.deepcopy(img_p[(t_size[0]):(t_size[1] + 1), (t_size[2]):(t_size[3] + 1)])
    obj_p[obj_p != 10] = 0  # avoid different color
    imsave('result/obj_p.jpg', obj_p)
    img_p[img_p == 10] = 1
    # ------- extract the obj_p --------

    # ------- get the lighting point -------
    x, y = np.where(img_p == 7)
    lx, ly = int((np.min(x) + np.max(x)) / 2), int((np.min(y) + np.max(y)) / 2)
    lx = max(lx - 2 + int((relation[2] - relation[3]) * (np.max(x) - np.min(x)) * 0.7), 0)
    ly = max(ly - 2 + int((relation[1] - relation[0]) * (np.max(y) - np.min(y)) * 0.7), 0)
    lx, ly = int(min(lx, img_p.shape[0])), int(min(ly, img_p.shape[1]))
    img_p[img_p == 7] = -100
    # ------- get the lighting point -------
    return lighting_img(img_p, [lx, ly]), lighting_img(obj_p,
                        [int(obj_p.shape[0] / 2) - 1, int(obj_p.shape[1] / 2) - 1])


def optimise_pp(v_grey, relation, shrink=1.0, pool=False, dichotomy=False):
    # ------ fix color ------
    # nan = 1; anchor = 7; pick = 10
    v_grey = np.array(v_grey, dtype=np.int16)
    v_grey[v_grey <= 40] = 1
    v_grey[v_grey > 210] = 10
    v_grey[(110 < v_grey) & (v_grey < 190)] = 7
    v_grey[(40 < v_grey) & (v_grey < 90)] = -100
    v_grey[v_grey > 11] = 1
    imsave('result/v_grey.jpg', v_grey)
    # -----------------------
    v_light, obj_p = change_img(v_grey, relation)
    imsave('result/v_light.jpg', v_light)
    imsave('result/obj_p.jpg', obj_p)
    graph_optimize = Graph_Optimize(v_light, obj_p, shrink=shrink, pool=pool, dichotomy=dichotomy)
    graph_optimize.conv_with_obj()
    print('Optimize Finished')


if __name__ == '__main__':
    """
    Load example picture and optimise the position & posture
    In the 'examples' directory, we provide five images as demo
    You can set value from different classes and create composite relation like left-front
        relation: [left, right, front, behind, beside, between, Nan] and the value in each class is from 0 to 1
        shrink: You can set the shrink value to adjust the minimum distance between the placed object and other objects,
                The recommended value of shrink is 1 to 1.8
        pool: Use pooling to speed up convolution if you like
        dichotomy: Use dichotomy to speed up the search of best posture if you like
    """
    imagine = imread('examples/ex_2.jpg')
    optimise_pp(imagine, relation=[0, 1, 0, 1, 0, 0, 0], shrink=1.2, pool=True, dichotomy=True)
