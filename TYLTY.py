import cv2  
import os
import numpy as np

def get_split_line(img, projection_row):
    split_line_list = []
    flag = False
    start = 0
    end = 0
    for i in range(0, len(projection_row)):
        if flag == False and projection_row[i] > 0:
            flag = True
            start = i
        elif flag and (projection_row[i] == 0 or i == len(projection_row) - 1):
            flag = False
            end = i
            if end - start < 15:  # need specify or rewrite
                flag = True
                continue
            else:
                split_line_list.append((start, end))
    return split_line_list


def get_contours(img):      
    contour_list = []
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours[0])):
        x, y, w, h = cv2.boundingRect(contours[0][i])
        contour_list.append((x, y, w, h))
        # cv2.rectangle(img_input, (x,y), (x+w, y+h), (0,0,255))
    return contour_list


def sort_merge(contour_row):
    contour_row = sorted(contour_row, key=lambda x: x[0])  # sort by x
    # print(contour_row)
    i = 0
    for _ in contour_row:    # 这部分的合并规则用的是刘成林paper中的方法
        if i == len(contour_row) - 1 or contour_row[i][0] == -1:
            break
        # print(contour_row[i])
        rectR = contour_row[i + 1]
        rectL = contour_row[i]
        ovlp = rectL[0] + rectL[2] - rectR[0]
        dist = abs((rectR[0] + rectR[2] / 2) - (rectL[0] - rectL[2] / 2))
        w_L = rectL[0] + rectL[2]
        w_R = rectR[0] + rectR[2]
        span = (w_R if w_R > w_L else w_L) - rectL[0]
        nmovlp = (ovlp / rectL[2] + ovlp / rectR[2]) / 2 - dist / span / 8
        if nmovlp > 0:
            x = rectL[0]
            y = (rectL[1] if rectL[1] < rectR[1] else rectR[1])
            w_L = rectL[0] + rectL[2]
            w_R = rectR[0] + rectR[2]
            w = (w_R if w_R > w_L else w_L) - x
            h_L = rectL[1] + rectL[3]
            h_R = rectR[1] + rectR[3]
            h = (h_R if h_R > h_L else h_L) - y
            contour_row[i] = (x, y, w, h)
            contour_row.pop(i + 1)  # after pop , index at i
            contour_row.append((-1, -1, -1, -1))  # add to fix bug(the better way is use iterator)
            i -= 1
        i += 1
    # print(contour_row)
    return contour_row


def combine_verticalLine(contour_row):
    i = 0
    pop_num = 0
    for _ in contour_row:
        rect = contour_row[i]
        if rect[0] == -1:
            break

        if rect[2] == 0:
            i += 1
            continue


        if rect[3] * 1.0 / rect[2] > 4:
            if i != 0 and i != len(contour_row) - 1:
                rect_left = contour_row[i - 1]
                rect_right = contour_row[i + 1]
                left_dis = rect[0] - rect_left[0] - rect_left[2]
                right_dis = rect_right[0] - rect[0] - rect[2]
                if left_dis <= right_dis and rect_left[2] < rect_right[2]:
                    x = rect_left[0]
                    y = (rect_left[1] if rect_left[1] < rect[1] else rect[1])
                    w = rect[0] + rect[2] - rect_left[0]
                    h_1 = rect_left[1] + rect_left[3]
                    h_2 = rect[1] + rect[3]
                    h_ = (h_1 if h_1 > h_2 else h_2)
                    h = h_ - y
                    contour_row[i - 1] = (x, y, w, h)
                    contour_row.pop(i)
                    contour_row.append((-1, -1, -1, -1))
                    pop_num += 1
                else:
                    x = rect[0]
                    y = (rect[1] if rect[1] < rect_right[1] else rect_right[1])
                    w = rect_right[0] + rect_right[2] - rect[0]
                    h_1 = rect_right[1] + rect_right[3]
                    h_2 = rect[1] + rect[3]
                    h_ = (h_1 if h_1 > h_2 else h_2)
                    h = h_ - y
                    contour_row[i] = (x, y, w, h)
                    contour_row.pop(i + 1)
                    contour_row.append((-1, -1, -1, -1))
                    pop_num += 1
        i += 1
    for i in range(0, pop_num):
        contour_row.pop()
    return contour_row


def split_oversizeWidth(contour_row):
    i = 0
    for _ in contour_row:
        rect = contour_row[i]
        if rect[2] * 1.0 / rect[3] > 1.2:  # height/width>1.2 -> split
            x_new = int(rect[0] + rect[2] / 2 + 1)
            y_new = rect[1]
            w_new = rect[0] + rect[2] - x_new
            h_new = rect[3]
            contour_row[i] = (rect[0], rect[1], int(rect[2] / 2), rect[3])
            contour_row.insert(i + 1, (x_new, y_new, w_new, h_new))
        i += 1
    return contour_row


def image_preprocess(img_input):
    gray_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 3)
    _, img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)  # 将一幅灰度图二值化 input-one channel

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.erode(img, kernel)
    # height,width=img.shape[:2]
    # img=cv2.resize(img,(int(width/2),int(height/2)),interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("img",img)
    return img


def get_segmentation_result(img):  # has been eroded
    projection_row = cv2.reduce(img, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)  # projection
    split_line_list = get_split_line(img, projection_row)  # split image as row
    segmentation_result = []
    for i in split_line_list:
        img_row = img[i[0]:i[1], :]
        contour_row = get_contours(img_row)
        contour_row = sort_merge(contour_row)
        contour_row = split_oversizeWidth(contour_row)
        contour_row = combine_verticalLine(contour_row)
        segmentation_result.append(contour_row)
        for (x, y, w, h) in contour_row:  # draw
            y+= i[0] 
            cv2.rectangle(img_input, (x,y), (x+w,y+h), (0, 0, 255)) 
            if w>0 and h>0:
                copy = img[y:y+h,x:x+w] #截取坐标为[y0:y1, x0:x1]
                cv2.imshow("img",copy)
                cv2.waitKey(0)
    return segmentation_result

pic_path = 'C:/Users//HYH/Documents/Program/Python/TYLTY/000.jpg'
img_input = cv2.imread(pic_path, 1)  # (2975, 1787, 3)   但是图片查看器显示的是  1787 * 2975
img = image_preprocess(img_input)  # erode
segmentation_result = get_segmentation_result(img)  # store segmentation result : [(x,y,w,h),(),...]

# cv2.imwrite("./save.jpg", img_input)
cv2.imshow("img_input",img_input)
cv2.waitKey(0)