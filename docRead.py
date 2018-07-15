# -*-coding:utf-8-*-
from win32com import client as wc
from win32com.client.gencache import EnsureDispatch
import os
import docx
import sys
import numpy
import zipfile

EnsureDispatch('Word.Application')


#字典-将等级转为分数
_switch = {
    "A+": 100,
    "A": 91,
    "A-": 85,
    "B": 79,
    "C": 69
}


#将doc转存为docx
def doSaveAs(path):
    word = wc.Dispatch('Word.Application')
    doc = word.Documents.Open(path)  # 目标路径下的文件
    doc.SaveAs(path + 'x', 12, False, "", True, "", False, False, False, False)  # 转化后路径下的文件
    doc.Close()
    word.Quit()


# 预处理-将目录下的doc转为docx
def doc2docx(dir):
    list = os.listdir(dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            path = os.path.join(path, u'文档')
            filelist = os.listdir(path)
            for j in range(0, len(filelist)):
                filepath = os.path.join(path, filelist[j])
                if filepath.endswith('.doc') and not os.path.exists(filepath + 'x'):
                    doSaveAs(filepath)
                if filepath.endswith('.docx'):
                    readdocx(filepath)


#将目录下的docx转为训练输入数组
def docx2data(dir):
    list = os.listdir(dir)  # 列出文件夹下所有的目录与文件
    matofdata = []
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            path = os.path.join(path, u'文档')
            filelist = os.listdir(path)
            for j in range(0, len(filelist)):
                filepath = os.path.join(path, filelist[j])
                if filepath.endswith('.docx'):
                    matofdata.append(readdocx(filepath))
    return matofdata


#将docx文件输出为输入数组-分别为各部分的字数
def readdocx(path):
    doc = docx.Document(path)
    # print(path + "   " + str(getImgNum(path)))
    text = ''.join([i.text + '\n' for i in doc.paragraphs])
    matOfData = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    textlist = text.split("实验目的")
    index = -1
    if len(textlist) == 2:
        index = 0

    textlist = textlist[-1].split("实验内容")
    if len(textlist) == 2:
        matOfData[index] = float(len(textlist[0]))
        index = 1

    textlist = textlist[-1].split("实验环境")
    if len(textlist) == 2:
        matOfData[index] = float(len(textlist[0]))
        index = 2

    textlist = textlist[-1].split("数据结构和算法模块")
    if len(textlist) == 2:
        matOfData[index] = float(len(textlist[0]))
        index = 3

    textlist = textlist[-1].split("核心代码")
    if len(textlist) == 2:
        matOfData[index] = float(len(textlist[0]))
        index = 4

    textlist = textlist[-1].split("运行说明")
    if len(textlist) == 2:
        matOfData[index] = float(len(textlist[0]))
        index = 5

    textlist = textlist[-1].split("测试与运行结果")
    if len(textlist) == 2:
        matOfData[index] = float(len(textlist[0]))
        index = 6

    textlist = textlist[-1].split("实验总结")
    if len(textlist) == 2:
        matOfData[index] = float(len(textlist[0]))
        index = 7

    if index != -1:
        matOfData[index] = float(len(textlist[-1]))
    else:
        matOfData[4] = (len(textlist[-1]))
    return matOfData


#得到docx中图片的个数
def getImgNum(path):
    zipf = zipfile.ZipFile(path)
    filelist = zipf.namelist()
    num = 0
    for fname in filelist:
        _, extension = os.path.splitext(fname)
        if extension in [".jpg", ".jpeg", ".png", ".bmp"]:
            num += 1
    return num



#将txt中的标签转为标签集
def txt2data(path, data):
    file = open(path)
    line = file.readline()
    while line:
        text = line.replace("\n", "")
        if len(text) != 0:
            data.append([_switch[text]])
        line = file.readline()
    file.close()
    return data


#将目录下的txt标签输出为标签集
def getTagData(dir):
    list = os.listdir(dir)  # 列出文件夹下所有的目录与文件
    data = []
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            path = os.path.join(path, u'评分.txt')
            txt2data(path, data)
    return data


# 获取训练数据的方法，输入样本所在的绝对路径。如：u'C:\\Users\\Uply\\Desktop\\人工智能实验评分样本'
# 获取两个数组，第一个数组为样本特征值，第二个数组为标签
def getTrainData(dir):
    return docx2data(dir), getTagData(dir)


def predictData(path):
    if path.endswith('.doc'):
        doSaveAs(path)
        path += 'x'
    if path.endswith('.docx'):
        return readdocx(path)
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0]


if __name__ == '__main__':
    rootdir = r'F:\495_work\Workspaces-School\学期资料\大三资料\大三下\人工智能\综合性实验\人工智能实验评分样本'

    # 将docx转doc
    # doc2docx(rootdir)

    # 查看训练数据的代码
    data1 = docx2data(rootdir)
    print(data1)
    print(len(data1))

    # 查看标签数据的代码
    data = getTagData(rootdir)
    print(data)
    print(len(data))

