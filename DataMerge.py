import os
import shutil

filePath = './服务外包大赛数据更新/训练数据/img_train_曲线/'
dstPath = './dataset/曲线图/'
count = 0
for i in os.listdir(filePath):
    subPath = filePath + str(i)
    for j in os.listdir(subPath):
        if j == 'draw.png':
            src = subPath + '/draw.png'
            dst = dstPath + '曲线图/' + str(i) + '.png'
            print('src', src, 'dst', dst)
            shutil.copy(src, dst)
        if j == 'db.txt':
            src = subPath + '/db.txt'
            dst = dstPath + 'db/' + str(i) + '.txt'
            print('src', src, 'dst', dst)
            shutil.copy(src, dst)
    count += 1
print(f'数据集大小: {count}')