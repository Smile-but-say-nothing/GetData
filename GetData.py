import cv2
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np
import re
import os
from math import ceil
import pandas as pd
from skimage import measure, morphology


def cv_show(img, name='img'):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 分割出xaxis和yaxis
def segAxis(img, direction='h'):
    img = img.copy()
    # direction = 'h' 分割x轴区域, 'v' 分割y轴区域， direction决定了扫描的方向
    height, width = img.shape[0], img.shape[1]
    startScan = 1 if direction == 'v' else height - 2
    endScan = width if direction == 'v' else 0
    step = 1 if direction == 'v' else -1
    backgroundPixelSum = np.sum(img[:height // 2, 0]) if direction == 'v' else np.sum(img[0, width // 2:])  # 取第0列（第0行）的二值化结果和作为对比标准
    for i in range(startScan, endScan, step):
        # 在此，scanLine和lastScanLine要么是单行，要么就是单列
        scanLine = img[:height // 2, i] if direction == 'v' else img[i, width // 2:]
        lastScanLine = img[:height // 2, i - step] if direction == 'v' else img[i - step, width // 2:]
        farScanLine = img[:height // 2, i + 5] if direction == 'v' else img[i - 5, width // 2:]
        if np.sum(scanLine) == backgroundPixelSum and np.sum(scanLine) != np.sum(lastScanLine) and np.sum(scanLine) == np.sum(farScanLine):
            # 画图显示一下
            # pt1 = (i, 0) if direction == 'v' else (0, i)
            # pt2 = (i, height) if direction == 'v' else (width, i)
            # seg = cv2.line(img, pt1, pt2, (255, 0, 0))
            # cv_show(seg)
            return i - step

def removeMarks(img):
    # 去除小物体
    height, width = img.shape[0], img.shape[1]
    img_copy = img.copy()
    img_copy[img_copy == 0] = 1
    img_copy[img_copy == 255] = 0
    img_copy = img_copy.astype(np.bool8)
    img_copy = morphology.remove_small_objects(img_copy, height * width // 8000, connectivity=2).astype(np.uint8)
    img_copy = (255 - (img_copy * 255)).astype(np.uint8)
    img = img_copy.copy()
    # 去除mark
    for i in range(0, 50, 1):
        col = np.where(img[:, i] == 0)[0]
        if len(col) <= 1:
            continue
        diffCol = np.diff(col)
        diffCol = np.delete(diffCol, np.where(diffCol == 1)[0])
        # 找到连续的0
        iszero = np.concatenate(([0], np.equal(img[:, i], 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        maxRange = max(ranges[:, 1] - ranges[:, 0])
        if len(diffCol) > 2 and maxRange < 100:
            img[:, i] = 255
    # 去除小物体
    height, width = img.shape[0], img.shape[1]
    img_copy = img.copy()
    img_copy[img_copy == 0] = 1
    img_copy[img_copy == 255] = 0
    img_copy = img_copy.astype(np.bool8)
    img_copy = morphology.remove_small_objects(img_copy, height * width // 8000, connectivity=2).astype(np.uint8)
    img_copy = (255 - (img_copy * 255)).astype(np.uint8)
    return img_copy


def segGraph(img, yAxisOffset, isGraphLine, isMarkLatent):
    # 传入的是GraphZone
    height, width = img.shape[0], img.shape[1]
    graphZoneBottom, graphZoneTop = 0, 0
    graphZoneLeft, graphZoneRight = 0, 0
    # 本质上就是找，黑色像素最多的某一行（列）
    # 无图线，有刻度线
    if not isGraphLine and not isMarkLatent:
        # top和bottom应该从中间向上下扫描
        for i in range(height // 2, height, 1):
            scanLine = img[i, :]
            if np.sum(scanLine) <= width * 255 // 4:
                graphZoneBottom = i
                break
        for i in range(height // 2, 0, -1):
            scanLine = img[i, :]
            if np.sum(scanLine) < width * 255 // 4:
                graphZoneTop = i
                break
        # left和right应该从中间向两边扫描
        for i in range(width // 2, 0, -1):
            scanLine = img[:, i]
            if np.sum(scanLine) <= height * 255 // 4:
                graphZoneLeft = i
                break
        for i in range(width // 2, width, 1):
            scanLine = img[:, i]
            if np.sum(scanLine) <= height * 255 // 4:
                graphZoneRight = i
                break
    else:
        for i in range(height - 1, height // 2, -1):
            scanLine = img[i, :]
            if np.sum(scanLine) != width * 255:
                graphZoneBottom = i + 1
                break
        for i in range(0, height // 2, 1):
            scanLine = img[i, :]
            if np.sum(scanLine) != width * 255:
                graphZoneTop = i - 1
                break
        for i in range(0, width // 2, 1):
            scanLine = img[:, i]
            if np.sum(scanLine) != height * 255:
                graphZoneLeft = i - 1
                break
        for i in range(width - 1, width // 2, -1):
            scanLine = img[:, i]
            if np.sum(scanLine) != height * 255:
                graphZoneRight = i + 1
                break
    return graphZoneBottom, graphZoneTop, graphZoneLeft + yAxisOffset, graphZoneRight + yAxisOffset, img


# 得到坐标轴最小最大值的标线，相对于图片左上角的位置
def getMarkLocation(img, direction='h'):
    markLocate = []
    # direction = 'h' x轴标线位置, 'v' y轴标线位置， direction决定了扫描的方向
    height, width = img.shape[0], img.shape[1]
    startScan = 1 if direction == 'v' else height - 2
    endScan = width if direction == 'v' else 0
    step = 1 if direction == 'v' else -1
    for i in range(startScan, endScan, step):
        # 在此，scanLine和lastScanLine要么是单行，要么就是单列
        scanLine = img[:, i] if direction == 'v' else img[i, :]
        lastScanLine = img[:, i - step] if direction == 'v' else img[i - step, :]
        if np.sum(scanLine) != np.sum(lastScanLine):  # 扫描线碰撞到了标线
            startSecondScan = height - 2 if direction == 'v' else 1
            endSecondScan = 0 if direction == 'v' else width - 1
            stepSecond = -1 if direction == 'v' else 1

            scanDict = dict(zip(*np.unique(scanLine, return_counts=True)))
            scanDict = sorted(scanDict.items(), key=lambda item: item[1], reverse=True)
            backGround = scanDict[0][0]
            for j in range(startSecondScan, endSecondScan, stepSecond):
                if scanLine[j] != backGround:
                    if direction == 'v':
                        markLocate.append((i, j))
                    else:
                        markLocate.append((j, i))
            break

    return markLocate


def getAxisValue(xAxisPixel, img, graphZoneBottom, graphZoneTop, graphZoneLeft, graphZoneRight, isFirstLast):
    # 第一个和最后一个去掉要去掉上下的杂质，所以有offset
    offset = 5 if isFirstLast != 'none' else 0
    graphZoneTop += 1  # slice从graphZoneTop + 1开始
    col = img[graphZoneTop + offset:graphZoneBottom - offset, xAxisPixel]
    col_index = [idx for idx, x in enumerate(col) if x == 0]
    # 对超过Graph左右边界的处理
    while xAxisPixel <= graphZoneLeft or (len(col_index) == 0 and isFirstLast == 'first'):
        xAxisPixel += 1
        col = img[graphZoneTop + offset:graphZoneBottom - offset, xAxisPixel]
        col_index = [idx for idx, x in enumerate(col) if x == 0]
    while xAxisPixel >= graphZoneRight or (len(col_index) == 0 and isFirstLast == 'last'):
        xAxisPixel -= 1
        col = img[graphZoneTop + offset:graphZoneBottom - offset, xAxisPixel]
        col_index = [idx for idx, x in enumerate(col) if x == 0]
    print('amended-xAxisPixel', xAxisPixel)
    if isFirstLast == 'none':
        # col_index_update = []
        # leftPlusRight = []
        # for idx in col_index:
        #     left, right = 0, 0
        #     while img[idx + graphZoneTop + offset, xAxisPixel - left] != 255:
        #         left += 1
        #     while img[idx + graphZoneTop + offset, xAxisPixel + right] != 255:
        #         right += 1
        #     col_index_update.append(abs(left - right))
        #     leftPlusRight.append(left + right)
        # col_index_update = np.where(col_index_update == np.min(col_index_update))[0]
        # # loc0是初始位置，中间那个
        # loc0 = col_index[int(np.median(col_index_update))]
        #
        # colMaxList, colMinList = [], []
        # for i in range(-3, 4):
        #     col = img[graphZoneTop + offset:graphZoneBottom - offset, xAxisPixel + i]
        #     col_index = [idx for idx, x in enumerate(col) if x == 0]
        #     if len(col_index) == 0:
        #         continue
        #     colMaxList.append(col_index[-1])
        #     colMinList.append(col_index[0])
        # colMaxList = np.array(colMaxList)
        # colMinList = np.array(colMinList)
        # colDiff = np.diff(colMaxList) + np.diff(colMinList)
        # # 从左向右看
        # downProb = np.sum(list(map(lambda x: x > 0, colDiff)))
        # upProb = np.sum(list(map(lambda x: x < 0, colDiff)))
        # # 把走势判断机制也引入中间点的取点
        # loc1 = int(np.max(col_index) - abs(np.max(col_index) - loc0) // 3) if downProb > upProb else int(np.max(col_index) + abs(np.max(col_index) - loc0) // 3) if downProb < upProb else loc0
        # print('downProb', downProb, 'upProb', upProb)
        #
        # leftPlusRight = np.diff(leftPlusRight)
        # leftPlusRight = np.delete(leftPlusRight, np.where(leftPlusRight == 0)[0])
        # A_Prob = np.sum(list(map(lambda x: x > 0, leftPlusRight)))
        # V_Prob = np.sum(list(map(lambda x: x < 0, leftPlusRight)))
        # print('A_Prob', A_Prob, 'V_Prob', V_Prob)
        # if A_Prob >= V_Prob + 6:
        #     loc = col_index[0]
        # elif V_Prob >= A_Prob + 6:
        #     loc = col_index[-1]
        # else:
        #     loc = loc1
        # loc += (graphZoneTop + offset)
        # return loc, xAxisPixel
        col_index_update = []
        leftPlusRight = []
        for idx in col_index:
            left, right = 0, 0
            while img[idx + graphZoneTop + offset, xAxisPixel - left] != 255:
                left += 1
            while img[idx + graphZoneTop + offset, xAxisPixel + right] != 255:
                right += 1
            col_index_update.append(abs(left - right))
            leftPlusRight.append(left + right)
        col_index_update = np.where(col_index_update == np.min(col_index_update))[0]
        loc0 = col_index[int(np.median(col_index_update))]
        loc = loc0
        leftPlusRight = np.diff(leftPlusRight)
        leftPlusRight = np.delete(leftPlusRight, np.where(leftPlusRight == 0)[0])
        A_Prob = np.sum(list(map(lambda x: x > 0, leftPlusRight)))
        V_Prob = np.sum(list(map(lambda x: x < 0, leftPlusRight)))
        print('A_Prob', A_Prob, 'V_Prob', V_Prob)
        if A_Prob >= V_Prob + 6:
            # loc = col_index[0] + abs(col_index[0] - loc0) // 4
            loc = col_index[0]
        if V_Prob >= A_Prob + 6:
            # loc = col_index[-1] - abs(col_index[-1] - loc0) // 4
            loc = col_index[-1]
        loc += (graphZoneTop + offset)
        return loc, xAxisPixel

    initCol_index = col_index
    colMaxList, colMinList = [], []
    start = 0 if isFirstLast == 'first' else -6 if isFirstLast == 'last' else 0
    stop = 5 if isFirstLast == 'first' else 1 if isFirstLast == 'last' else 9
    step = 1
    for i in range(start, stop, step):
        col = img[graphZoneTop + offset:graphZoneBottom - offset, xAxisPixel + i]
        col_index = [idx for idx, x in enumerate(col) if x == 0]
        if len(col_index) == 0:
            continue
        colMaxList.append(col_index[-1])
        colMinList.append(col_index[0])
    colMaxList = np.array(colMaxList)
    colMinList = np.array(colMinList)
    colDiff = np.diff(colMaxList) + np.diff(colMinList)
    # 从左向右看
    downProb = np.sum(list(map(lambda x: x > 0, colDiff)))
    upProb = np.sum(list(map(lambda x: x < 0, colDiff)))
    print('downProb', downProb, 'upProb', upProb)
    if isFirstLast == 'first':
        loc = int(np.min(initCol_index)) if downProb > upProb else int(np.max(initCol_index)) if downProb < upProb else int(np.median(initCol_index))
        loc += (graphZoneTop + offset)
        return loc, xAxisPixel
    if isFirstLast == 'last':
        loc = int(np.max(initCol_index)) if downProb > upProb else int(np.min(initCol_index)) if downProb < upProb else int(np.median(initCol_index))
        loc += (graphZoneTop + offset)
        return loc, xAxisPixel


def updateMarkLocate(MarkLocate, yAxisOffset, type):
    if type == 'x':
        tempLocate, mid, newMarkLocate = [], [], []
        for i in range(len(MarkLocate) - 1):
            if MarkLocate[i + 1][0] - MarkLocate[i][0] <= 50:
                tempLocate.append(MarkLocate[i][0])
            else:
                tempLocate.append(MarkLocate[i][0])
                mid.append(round(np.median(tempLocate)))
                tempLocate = []
        tempLocate.append(MarkLocate[-1][0])
        mid.append(round(np.median(tempLocate)))
        newMarkLocate = [MarkLocate[0][1] for i in range(len(mid))]
        mid = [item + yAxisOffset for item in mid]
        newMarkLocate = list(zip(mid, newMarkLocate))
        return newMarkLocate
    if type == 'y':
        tempLocate, mid, newMarkLocate = [], [], []
        for i in range(len(MarkLocate) - 1):
            if MarkLocate[i + 1][1] - MarkLocate[i][1] >= -50:
                tempLocate.append(MarkLocate[i][1])
            else:
                tempLocate.append(MarkLocate[i][1])
                mid.append(round(np.median(tempLocate)))
                tempLocate = []
        tempLocate.append(MarkLocate[-1][1])
        mid.append(round(np.median(tempLocate)))
        newMarkLocate = [MarkLocate[0][0] + yAxisOffset for i in range(len(mid))]
        newMarkLocate = list(zip(newMarkLocate, mid))
        return newMarkLocate


def getMarkLocationLatent(img, fx, fy, xAxisOffset, direction='h'):
    # 因为有些切割后的图，会把另一边的0包含进去，所以要消除，比如19.png
    idx = np.where(img[0, :] != 255)[0] if direction == 'h' else np.where(img[:, -1] != 255)[0]
    if direction == 'h' and len(idx):
        img[:, idx[0]:idx[-1]] = 255
    if direction == 'v' and len(idx):
        img[idx[0]:idx[-1], :] = 255
    # 腐蚀，将坐标轴的数字变黑
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    erosion = cv2.erode(img, kernel, iterations=5)
    if direction == 'h' and 0 in erosion[0, :]:
        erosion[0, :] = 255
    if direction == 'v' and 0 in erosion[:, -1]:
        erosion[:, -1] = 255
    cv2.imwrite('./' + direction + 'erosion/' + imgFileName, erosion)
    # 找轮廓
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    MarkLocate = []
    for cnt in contours[-1:-len(contours):-1]:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 3)
        if w < 100 or h < 100:
            continue
        if direction == 'h':
            MarkLocate.append(((x + w // 2) // fx, y // fy + xAxisOffset))
        if direction == 'v':
            MarkLocate.append(((x + w) // fx, (y + h // 2) // fy))
    if direction == 'h':
        MarkLocate = sorted(MarkLocate, key=lambda x: x[0], reverse=False)
    if direction == 'v':
        MarkLocate = sorted(MarkLocate, key=lambda x: x[1], reverse=True)
    return MarkLocate


def removeLineInGraph(img, type):
    img = img.copy()
    height, width = img.shape[0], img.shape[1]
    if type == 'detect':
        isGraphLine = False
        for i in range(width // 2, width // 4, -1):
            if np.sum(img[:, i]) <= height * 255 // 3:
                isGraphLine = True
        return img, isGraphLine

    if type == 'remove':
        # Method1，经过试验，该方法已被弃用
        # img1 = img.copy()
        # for row in range(1, height):
        #     if np.sum(img1[row, :]) <= width * 255 // 3:
        #         img1[row, :] = img1[row - 1, :]
        # for col in range(1, width):
        #     if np.sum(img1[:, col]) <= height * 255 // 3:
        #         img1[:, col] = img1[:, col - 1]
        # img1 = removeMarks(img1)
        # Method2
        img2 = img.copy()
        for row in range(1, height):
            if np.sum(img2[row, :]) <= width * 255 // 3:
                img2[row, :] = img2[row - 1, :]
        for col in range(1, width - 1):
            if np.sum(img2[:, col]) <= height * 255 // 3:
                img2[:, col] = 255
        img2 = removeMarks(img2)
        pt1, pt2, pt3, pt4 = (0, 0), (0, 0), (0, 0), (0, 0)
        flag = 1
        for col in range(1, width - 1):
            if np.sum(img2[:, col]) == height * 255 and np.sum(img2[:, col + 1]) != height * 255 and flag:
                flag = 0
                col += 1
                continue
            if np.sum(img2[:, col + 1]) == height * 255 and np.sum(img2[:, col]) != height * 255 and pt1 == (0, 0) and pt2 == (0, 0):
                left = np.where(img2[:, col] == 0)[0]
                leftTop, leftBottom = left[0], left[-1]
                pt1, pt2 = (col, leftTop), (col, leftBottom)
            if np.sum(img2[:, col]) == height * 255 and np.sum(img2[:, col + 1]) != height * 255 and pt3 == (0, 0) and pt4 == (0, 0):
                right = np.where(img2[:, col + 1] == 0)[0]
                rightTop, rightBottom = right[0], right[-1]
                pt3, pt4 = (col, rightTop), (col, rightBottom)
            if pt1 != (0, 0) and pt2 != (0, 0) and pt3 != (0, 0) and pt4 != (0, 0):
                for i in range(0, abs(pt1[1] - pt2[1])):
                    for j in range(0, abs(pt3[1] - pt4[1])):
                        cv2.line(img2, (pt1[0], pt1[1] + i), (pt3[0], pt3[1] + j), color=0, thickness=3)
                pt1, pt2, pt3, pt4 = (0, 0), (0, 0), (0, 0), (0, 0)
        # img1 = cv2.bitwise_not(img1)
        # img2 = cv2.bitwise_not(img2)
        # img = cv2.bitwise_not(cv2.add(img1, img2))
        return img2


def compareGraphZone(graphZoneCut, graphZoneGray):
    imgShape = graphZoneCut.shape  # H W C
    graphZoneCutOrigin = graphZoneCut.copy()
    height, width = imgShape[0], imgShape[1]
    graphZoneReversed = 255 - graphZoneGray[:, 50:-100]
    graphZoneAdd = graphZoneReversed + graphZoneCut[:, 50:-100]
    graphZoneCutInit = graphZoneCut[:, 50:-100]
    T = 200
    while True:
        graphZoneCut = graphZoneCutInit.copy()
        graphZoneCut[graphZoneAdd <= T] = 255
        if np.sum(255 - graphZoneCut[1:-1, 1:-1]) <= height * width * 255 // 8:
            T -= 10
        else:
            break
        if T == 0:
            print('T', T)
            return graphZoneCutOrigin
    print('T', T)
    graphZoneCutOrigin[:, 50:-100] = graphZoneCut
    # 去除小物体
    img = graphZoneCutOrigin
    height, width = img.shape[0], img.shape[1]
    img_copy = img.copy()
    img_copy[img_copy == 0] = 1
    img_copy[img_copy == 255] = 0
    img_copy = img_copy.astype(np.bool8)
    img_copy = morphology.remove_small_objects(img_copy, height * width // 2000, connectivity=2).astype(np.uint8)
    img_copy = (255 - (img_copy * 255)).astype(np.uint8)
    return img_copy


def GetData(img, k, yTrue, imgFileName, outputPath):
    # Step.0 放大图片，默认3倍
    img = img.copy()
    img = cv2.resize(img, (0, 0), fx=3, fy=3)
    imgShape = img.shape  # H W C
    height, width = imgShape[0], imgShape[1]
    print('Shape:', img.shape)

    # Step.1 灰度转换及去黑底去背景
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 有些图片是黑底的，翻转一下
    if gray[0, 0] == 0:
        gray = cv2.bitwise_not(gray)
    cv2.imwrite(outputPath + 'gray/' + imgFileName, gray)
    # 去背景，让大于10%的像素值为255
    grayDict = dict(zip(*np.unique(gray, return_counts=True)))
    grayList = sorted(grayDict.items(), key=lambda item: item[1], reverse=True)
    grayList = list(filter(lambda item: item[1] >= 0.10 * height * width, grayList))
    grayList = [item[0] for item in grayList]
    grayCopy = gray.copy()
    for i in grayList:
        grayCopy[grayCopy == i] = 255
    cv2.imwrite(outputPath + 'grayCopy/' + imgFileName, grayCopy)

    # Step.2 二值分割，选取较大的阈值
    ret, thresh = cv2.threshold(grayCopy, 240, 255, cv2.THRESH_BINARY)
    cv2.imwrite(outputPath + 'thresh/' + imgFileName, thresh)

    # Step.3 分割坐标轴区域
    # 分割出来的坐标轴区域，在文字右侧（上方）是有一行白色的，必须有，不然OCR结果可能识别不出来，所以有10像素的保留量
    xAxisOffset = segAxis(thresh, direction='h') - 10
    yAxisOffset = segAxis(thresh, direction='v') + 10
    print('xAxisOffset', xAxisOffset, 'yAxisOffset', yAxisOffset)
    # 坐标轴区域是先用裁剪灰度图，再用普通阈值来二值化的
    ret, xAxisZone = cv2.threshold(gray[xAxisOffset:height, :], 200, 255, cv2.THRESH_BINARY)
    ret, yAxisZone = cv2.threshold(gray[:, 0:yAxisOffset], 200, 255, cv2.THRESH_BINARY)
    # 图区，直接裁剪二值化图
    graphZone = thresh[0:xAxisOffset, yAxisOffset:width]
    # 检测一下graphZone里有没有背景里的那种线
    graphZone, isGraphLine = removeLineInGraph(graphZone, 'detect')
    print('isGraphLine', isGraphLine)

    # 为了后面OCR结果准确，坐标轴区域要放大
    fx, fy = 5, 5
    xAxisZone = cv2.resize(xAxisZone, (0, 0), fx=fx, fy=fy)
    yAxisZone = cv2.resize(yAxisZone, (0, 0), fx=fx, fy=fy)
    # 开运算，去除白色小毛刺
    kernel = np.ones((5, 5), np.uint8)
    xAxisZone = cv2.morphologyEx(xAxisZone, cv2.MORPH_OPEN, kernel)
    yAxisZone = cv2.morphologyEx(yAxisZone, cv2.MORPH_OPEN, kernel)
    # 中值滤波，去除孤立点
    xAxisZone = cv2.medianBlur(xAxisZone, 5)
    yAxisZone = cv2.medianBlur(yAxisZone, 5)
    # 其实此时又是灰度图像了
    cv2.imwrite(outputPath + 'xAxisZone/' + imgFileName, xAxisZone)
    cv2.imwrite(outputPath + 'yAxisZone/' + imgFileName, yAxisZone)
    cv2.imwrite(outputPath + 'graphZone/' + imgFileName, graphZone)

    # Step.4 OCR得到坐标轴数字
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    xAxisText = pytesseract.image_to_string(xAxisZone, config=custom_config).replace('J', '5').replace('A', '4').replace('o', '0').replace('O', '0')
    yAxisText = pytesseract.image_to_string(yAxisZone, config=custom_config).replace('J', '5').replace('A', '4').replace('o', '0').replace('O', '0')
    # 分割得到list
    xAxisText = xAxisText.split(' ')
    yAxisText = yAxisText.split('\n')
    # 利用正则表达式过滤
    xAxisText = [re.sub('[^\d]+', '', s) for s in xAxisText]
    yAxisText = [re.sub('[^\d]+', '', s) for s in yAxisText]
    # 去除空值
    xAxisText = list(filter(lambda i: i != '', xAxisText))
    yAxisText = list(filter(lambda i: i != '', yAxisText))
    print('xAxisText', xAxisText, '\nyAxisText', yAxisText)
    # 转int元素的List
    xAxisList = [int(item) for item in xAxisText]
    yAxisList = [int(item) for item in yAxisText]
    print('xAxis', xAxisList, 'yAxis', yAxisList)
    # 做差找到最可能的分隔
    xAxisSpacing = np.diff(xAxisList)
    yAxisSpacing = np.diff(yAxisList)
    xAxisSpacingDict = dict(zip(*np.unique(xAxisSpacing, return_counts=True)))
    yAxisSpacingDict = dict(zip(*np.unique(yAxisSpacing, return_counts=True)))
    xAxisSpacingDict = sorted(xAxisSpacingDict.items(), key=lambda item: item[1], reverse=True)
    yAxisSpacingDict = sorted(yAxisSpacingDict.items(), key=lambda item: item[1], reverse=True)
    xAxisSpacing = xAxisSpacingDict[0][0]
    yAxisSpacing = abs(yAxisSpacingDict[0][0])
    print('xAxisSpacing', xAxisSpacing, 'yAxisSpacing', yAxisSpacing)

    # Step.5 找刻度线位置，即找那些数字的位置
    # 图分为两种，一种是有凸起的刻度线的，另一种是光滑的，只能通过数字的位置来确定像素位置
    # 对于第一种，设置扫描线即可
    xMarkLocate = getMarkLocation(graphZone, 'h')  # 注意，这里的MarkLocate是没有加yAxisOffset的，是相对于GraphZone的
    yMarkLocate = getMarkLocation(graphZone, 'v')
    # 去掉多余的MarkLocate， 并且将坐标原点变换为原图的左上角，所以要传入yAxisOffset
    xMarkLocate = updateMarkLocate(xMarkLocate, yAxisOffset, 'x')
    yMarkLocate = updateMarkLocate(yMarkLocate, yAxisOffset, 'y')
    # 如果去重之后，发现xMarkLocate或者yMarkLocate的点只有一个，说明肯定是刻度线隐藏的情况
    isMarkLatent = False
    if len(xMarkLocate) == 1 or len(yMarkLocate) == 1:
        isMarkLatent = True

    # 对于第二种，要对坐标区进行腐蚀，再轮廓检测，近似得到数字的隐形刻度线像素位置，这种方法得到的刻度线位置肯定是正确的
    xMarkLocate2 = getMarkLocationLatent(xAxisZone, fx, fy, xAxisOffset, 'h')
    yMarkLocate2 = getMarkLocationLatent(yAxisZone, fx, fy, xAxisOffset, 'v')
    # 判断两种方法得到的刻度线数量是否一样，如果不一样，选第二个轮廓检测的方法，获得正确的刻度线位置
    if len(xMarkLocate) != len(xMarkLocate2) or len(yMarkLocate) != len(yMarkLocate2):
        xMarkLocate = xMarkLocate2
        yMarkLocate = yMarkLocate2
        print('Use second MarkLocate')
        isMarkLatent = True
    print('isMarkLatent', isMarkLatent)
    print('xMarkLocate', xMarkLocate, '\nyMarkLocate', yMarkLocate)
    # 把检测到的点加到图上看一下
    for point in xMarkLocate:
        cv2.circle(img, point, 5, (0, 0, 255), 3)
    for point in yMarkLocate:
        cv2.circle(img, point, 5, (0, 0, 255), 3)

    # Step.6 确定坐标轴范围
    xAxisMax = 0 + (len(xMarkLocate) - 1) * xAxisSpacing
    yAxisMax = 0 + (len(yMarkLocate) - 1) * yAxisSpacing
    xAxisMin, yAxisMin = 0, 0
    print('xAxis', xAxisMin, xAxisMax, 'yAxis', yAxisMin, yAxisMax)

    # Step.7 计算数值-像素比率
    xRatio = abs(xMarkLocate[4][0] - xMarkLocate[3][0]) / float(xAxisSpacing)
    yRatio = abs(yMarkLocate[4][1] - yMarkLocate[3][1]) / float(yAxisSpacing)
    print('xRatio', xRatio, 'yRatio', yRatio)
    # Step.8 分割图区，返回图的上下左右相对于原图的位置
    # 如果图区里有线，要先消掉这些线
    if isGraphLine or isMarkLatent:
        graphZone = removeLineInGraph(graphZone, 'remove')
    cv2.imwrite(outputPath + 'removeGap/' + imgFileName, graphZone)
    graphZoneBottom, graphZoneTop, graphZoneLeft, graphZoneRight, graphZone = segGraph(graphZone, yAxisOffset, isGraphLine, isMarkLatent)
    print('graphZoneBottom', graphZoneBottom, 'graphZoneTop', graphZoneTop, 'graphZoneLeft', graphZoneLeft, 'graphZoneRight', graphZoneRight)
    # 更新thresh
    thresh[0:xAxisOffset, yAxisOffset:width] = graphZone
    # 更新graphZone
    graphZoneGray = grayCopy[graphZoneTop:graphZoneBottom, graphZoneLeft:graphZoneRight]
    graphZoneCut = thresh[graphZoneTop:graphZoneBottom, graphZoneLeft:graphZoneRight]
    graphZoneCut = compareGraphZone(graphZoneCut, graphZoneGray)
    # 更新thresh
    thresh[graphZoneTop:graphZoneBottom, graphZoneLeft:graphZoneRight] = graphZoneCut
    cv2.imwrite(outputPath + 'processedThresh/' + imgFileName, thresh)
    maxNumber = (graphZoneRight - xMarkLocate[0][0]) / xRatio
    print('maxNumber', maxNumber)  # 右顶点值
    # maxNumber = round(maxNumber)
    # maxNumber = int(maxNumber) if round(maxNumber) % 2 == 0 else int(ceil(maxNumber)) if int(ceil(maxNumber)) % 2 == 0 else int(maxNumber)
    # print('maxNumberRound', maxNumber)

    # Step.9 四等分maxNumber
    xPoint = [i * ((maxNumber + 1) / 4) for i in range(0, 4)]
    xPoint.append(maxNumber)
    print('xPoint', xPoint)

    # Step.10 抽取点，计算误差
    diff = 0  # 误差
    for i, xNumber in enumerate(xPoint):
        xAxisPixel = int((xNumber - xAxisMin) * xRatio + xMarkLocate[0][0])
        print('xNumber', xNumber, 'xAxisPixel', xAxisPixel)
        isFirstLast = 'first' if i == 0 else 'last' if i == len(xPoint) - 1 else 'none'
        yAxisPixel, xAxisPixel = getAxisValue(xAxisPixel, thresh, graphZoneBottom, graphZoneTop, graphZoneLeft, graphZoneRight, isFirstLast)
        yNumber = float(abs(yMarkLocate[0][1] - yAxisPixel) / yRatio)
        print('yNumber', yNumber, 'yAxisPixel', yAxisPixel)
        cv2.line(img, (xAxisPixel, yMarkLocate[0][1]), (xAxisPixel, yAxisPixel), (255, 0, 0), 2)
        cv2.line(img, (xMarkLocate[0][0], yAxisPixel), (xAxisPixel, yAxisPixel), (255, 0, 0), 2)
        cv2.putText(img, 'xNumber:{:.4f}'.format(xNumber), (0, 50 + i * 150), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 0, 255), 4)
        cv2.putText(img, 'yPredict:{:.4f}'.format(yNumber), (0, 100 + i * 150), cv2.FONT_HERSHEY_PLAIN, 4.0, (255, 0, 0), 4)
        cv2.putText(img, 'yTrue:{:.4f}'.format(yTrue[i]), (0, 150 + i * 150), cv2.FONT_HERSHEY_PLAIN, 4.0, (255, 0, 0), 4)
        print('yTrue: {:.4f}, yPredict: {:.4f}, k: {:.4f},  yTrue/k: {:.4f}, yPredict/k: {:.4f}'.format(yTrue[i], yNumber, k, yTrue[i] / k, yNumber / k))
        diff += (yTrue[i] / k - yNumber / k) ** 2
    # 5个点的误差
    diff = np.sqrt(diff)
    print(f'diff: {diff}')
    cv2.putText(img, 'diff:{:.4f}'.format(diff), (600, 50), cv2.FONT_HERSHEY_PLAIN, 4.0, (255, 0, 0), 4)
    cv2.imwrite(outputPath + 'result/' + imgFileName, img)
    return diff


if __name__ == '__main__':
    # imgPath = './train/curve/fig/'
    # dbPath = './train/curve/db/'
    # outputPath = './train output/curve/'
    # 图片路径
    imgPath = './train100 curve/dataset100/'
    # db路径
    dbPath = './train100 curve/db100/'
    # 输出文件夹总路径
    outputPath = './train100 curve output/'
    # 输出的xlsx的路径
    xlsPath = './train100 curve output/curve.xlsx'
    d, errorCount = 0, 0
    print(len(os.listdir(imgPath)))
    loopCount = 0
    df = pd.DataFrame(columns=['pic', 'diff'])
    for imgFileName in os.listdir(imgPath):
        if imgFileName != '5.png':
            continue
        imgSubPath = imgPath + imgFileName
        # 读取图片
        img = cv2.imread(imgSubPath)
        # 读取db
        dbSubPath = dbPath + imgFileName.replace('.png', '.txt')
        dbFile = open(dbSubPath, 'r')  # 打开文件
        db = []
        for lines in dbFile:
            value = lines.split('\t')
            db.append(value)
        dbFile.close()
        # 常数k
        k = float(db[0][0])
        # yTrue
        yTrue = list(map(float, db[1][0].split(',')))
        print(f'正在抽取: {imgFileName}, imgShape: {img.shape}, k: {k}, yTrue: {yTrue}')
        # 传入抽取函数
        GetData(img, k, yTrue, imgFileName, outputPath)
        exit()
        try:
            imgSubPath = imgPath + imgFileName
            # 读取图片
            img = cv2.imread(imgSubPath)
            # 读取db
            dbSubPath = dbPath + imgFileName.replace('.png', '.txt')
            dbFile = open(dbSubPath, 'r')  # 打开文件
            db = []
            for lines in dbFile:
                value = lines.split('\t')
                db.append(value)
            dbFile.close()
            # 常数k
            k = float(db[0][0])
            # yTrue
            yTrue = list(map(float, db[1][0].split(',')))
            print(f'正在抽取: {imgFileName}, imgShape: {img.shape}, k: {k}, yTrue: {yTrue}')
            # 传入抽取函数
            diff = GetData(img, k, yTrue, imgFileName, outputPath)
            df.loc[loopCount] = [imgFileName, diff]
            d += diff
            print(f'累计误差:{d}\n')
        except BaseException as e:
            print(f'Error: {e}\n')
            errorCount += 1
            continue
        loopCount += 1
    df.to_excel(xlsPath, sheet_name='curve')
    print(f'平均误差:{d / 101}')
    print(f'错误次数:{errorCount}')
