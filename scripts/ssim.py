# -*- coding: utf-8 -*-
"""
https://github.com/chinue/Fast-SSIM
"""

import os
import numpy as np
import math
import skimage.measure
import scipy
import scipy.misc
import ctypes
import re
from pathlib import Path

ssim_dll_path = str(Path().resolve().joinpath("scripts"))
ssim_dll_name = "ssim.dll" if (os.name == "nt") else "libssim.so"

"""
by: Chen Yu
version 1.0.0.1 (2017-03-21)
 1. initial version with print2, printf, Timer, findFileList
version 1.0.0.2 (2017-08-01)
 1. add print array and save numpy array
version 1.0.0.3 (2017-11-12)
 1. add set_num for numpy array
version 1.0.0.4 (2018-04-06)
 1. add golbal time counter for Timer
"""
import platform
from builtins import *
from functools import cmp_to_key

isWindows = platform.system() == "Windows"
# print(isWindows)
import ctypes

isDebugPrint = True

STD_OUTPUT_HANDLE = -11


class TextColor:
    BLACK = 0  # - black
    DaBLUE = 1  # - dark blue
    DaGREEN = 2  # - dark green
    DaCYAN = 3  # - dark cyan
    DaRED = 4  # - dark red
    DaMAGENTA = 5  # - dark magenta
    GOLDEN = 6  # - golden
    GRAY = 7  # - gray
    DaGRAY = 8  # - dark gray
    BLUE = 9  # - blue
    GREEN = 10  # - green
    CYAN = 11  # - cyan
    RED = 12  # - red
    MAGENTA = 13  # - magenta
    YELLOW = 14  # - yellow
    WHITE = 15  # - white


if isWindows:
    _std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

# import sys
import types

_win_color_map = {
    "black": 0,
    "dark blue": 1,
    "dark green": 2,
    "dark cyan": 3,
    "dark red": 4,
    "dark magenta": 5,
    "dark pink": 5,
    "golden": 6,
    "gray": 7,
    "dark gray": 8,
    "blue": 9,
    "green": 10,
    "cyan": 11,
    "red": 12,
    "magenta": 13,
    "pink": 13,
    "yellow": 14,
    "white": 15,
}

_linux_color_map = {
    "end": "\033[0m",
    "black": "\033[0;30m",
    "red": "\033[1;31m",
    "green": "\033[1;32m",
    "yellow": "\033[1;33m",
    "blue": "\033[1;34m",
    "magenta": "\033[1;35m",
    "pink": "\033[1;35m",
    "cyan": "\033[1;36m",
    "white": "\033[1;37m",
    "gray": "\033[0;37m",
    "dark blue": "\033[0;34m",
    "dark red": "\033[0;31m",
    "dark green": "\033[0;32m",
    "golden": "\033[0;33m",
    "dark magenta": "\033[0;35m",
    "dark gray": "\033[0;37m",
    "dark cyan": "\033[0;36m",
}
_linux_color_list = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "pink",
    "cyan",
    "white",
    "gray",
    "dark blue",
    "dark red",
    "dark green",
    "golden",
    "dark magenta",
    "dark gray",
    "dark cyan",
]


def printf(print_text, *args, textColor="white", end="", isPrint=True):
    """
    printf is similar to print but with more text color parameter AND do not append a newline by default.
    textColor: use const int value [TextColor.RED, TextColor.GREEN, TextColor.BLUE, TextColor.YELLOW, TextColor.WHITE, ...]
               or string value ['red', 'green', 'blue', 'yellow', 'white', ...]
    print_text: text to print
    args: more arguments to print
    """
    if isPrint == False:
        return

    if isWindows:
        if type(textColor) == type("a"):
            textColor = textColor.lower()
            textColor = _win_color_map[textColor]
        ctypes.windll.kernel32.SetConsoleTextAttribute(_std_out_handle, textColor)
        print(print_text % args, end=end, flush=True)
        ctypes.windll.kernel32.SetConsoleTextAttribute(_std_out_handle, TextColor.WHITE)
    else:
        if type(textColor) == type(1):
            textColor = _linux_color_list[textColor]
        textColor = textColor.lower()
        s = print_text % args
        s = _linux_color_map[textColor] + s + _linux_color_map["end"]
        print(s, end=end, flush=True)


def print2(print_text, *args, textColor="white", end="\n", isPrint=True, isFlush=False):
    """
    print2 is similar to print but with more text color parameter.
    textColor: use const int value [TextColor.RED, TextColor.GREEN, TextColor.BLUE, TextColor.YELLOW, TextColor.WHITE, ...]
               or string value ['red', 'green', 'blue', 'yellow', 'white', ...]
    print_text: text to print
    args: more arguments to print
    """
    if isPrint == False:
        return

    flush = False if end == "\n" else True
    if isFlush:
        flush = True
    n = len(args)
    if isWindows:
        if type(textColor) == type("a"):
            textColor = textColor.lower()
            textColor = _win_color_map[textColor]
        ctypes.windll.kernel32.SetConsoleTextAttribute(_std_out_handle, textColor)
        if n == 0 or args == ((),):
            print(print_text, end=end, flush=flush)
        elif n == 1:
            print(print_text, args[0], end=end, flush=flush)
        else:
            print(print_text, args, end=end, flush=flush)
        ctypes.windll.kernel32.SetConsoleTextAttribute(_std_out_handle, TextColor.WHITE)
    else:
        if type(textColor) == type(1):
            textColor = _linux_color_list[textColor]
        textColor = textColor.lower()
        s = _linux_color_map[textColor] + str(print_text) + _linux_color_map["end"]
        if n == 0 or args == ((),):
            print(s, end=end, flush=flush)
        elif n == 1:
            print(s, args[0], end=end, flush=flush)
        else:
            print(s, args, end=end, flush=flush)


def print_debug(print_text, *args, textColor="white", end="\n", isPrint=True):
    if isDebugPrint:
        print2(print_text, args, textColor=textColor, end=end, isPrint=isPrint)


import time


class Timer:
    """
    usage:
          T=Timer()
          T.begin()
          # the code you want to estimate timing
          T.end("Fun")
    """

    def __init__(self):
        self.__freq = self._get_frequency()
        self.set_global_start()

    def _get_frequency(self):
        if isWindows:
            freq = ctypes.c_longlong(0)
            ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
            freq = freq.value
        else:
            freq = 1.0
        return freq

    def _get_time(self):
        if isWindows:
            t = ctypes.c_longlong(0)
            ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(t))
            _t = t.value
        else:
            _t = time.time()
        return _t

    def set_global_start(self):
        self.__start = self._get_time()
        return self.__start

    def begin(self):
        self.__t1 = self._get_time()
        return self.__t1

    def end(self, tag="run", isPrint=True, end="\n", textColor=None):
        """
        used to print program running time
        tag: main message to print
        isPrint: whether print or not

        the return value is the milliseconds which between begin() and end()
        """
        self.__t2 = self._get_time()
        millisec = 1000.0 * (self.__t2 - self.__t1) / self.__freq
        self.__end = self.__t2
        if isPrint:
            if millisec > 1000:
                printf("%s time=%.3f sec", tag, millisec / 1000.0, textColor="red" if (textColor == None) else textColor, end=end)
            elif millisec > 1:
                printf("%s time=%3.0f ms", tag, millisec, textColor="green" if (textColor == None) else textColor, end=end)
            else:
                printf("%s time=%3.0f us", tag, millisec * 1000.0, textColor="yellow" if (textColor == None) else textColor, end=end)
        self.__t1 = self.__t2
        self.millisec = millisec
        return millisec

    def pass_time(self, out_str=True, use_end_time=True):
        "pass_time(out_str=True, use_end_time=True)"
        if use_end_time != True:
            self.__end = self._get_time()
        self._pass_sec = (self.__end - self.__start) / self.__freq
        [h, m, s] = secToHMS(self._pass_sec)
        if out_str:
            return "%02d:%02d:%02d" % (h, m, s)
        return [h, m, s]

    def rest_time(self, pass_idx, total_idx, out_str=True):
        "rest_time(pass_idx, total_idx, out_str=True)"
        pass_idx = max(1, pass_idx + 1)
        rest_idx = total_idx - pass_idx
        self._pass_sec = (self.__end - self.__start) / self.__freq
        rest_sec = self._pass_sec * rest_idx / pass_idx
        [h, m, s] = secToHMS(rest_sec)
        if out_str:
            return "%02d:%02d:%02d" % (h, m, s)
        return [h, m, s]


def secToHMS(sec, out_str=False):
    s = int(sec)
    h = s // 3600
    s = s % 3600
    m = s // 60
    s = s % 60
    if out_str:
        return "%02d:%02d:%02d" % (h, m, s)
    return [h, m, s]


def calcRestHMS(pass_sec, pass_idx, total_idx, out_str=False):
    pass_idx = max(1, pass_idx + 1)
    rest_idx = total_idx - pass_idx
    rest_sec = pass_sec * rest_idx / pass_idx
    return secToHMS(rest_sec, out_str)

    # import tkinter.filedialog as tk
    # def getOpenFileName(initialdir='d:', filetypes=[('image format(*.jpg;*.png;*.bmp)', '*.jpg;*.png;*.bmp')]):
    #    filename=tk.askopenfilename(filetypes=filetypes, initialdir=initialdir)
    #    return filename

    # def getOpenFileNameList(initialdir='d:', filetypes=[('image format(*.jpg;*.png;*.bmp)', '*.jpg;*.png;*.bmp')]):
    #    filenames=tk.askopenfilenames(filetypes=filetypes, initialdir=initialdir)
    #    return filenames

    # def getSaveFileName(initialdir='d:', filetypes=[('image format(*.jpg;*.png;*.bmp)', '*.jpg;*.png;*.bmp')]):
    #    filename=tk.asksaveasfilename(filetypes=filetypes, initialdir=initialdir)
    return filename


import os


class SortType:
    NO_SORT = None
    LOWER_CASE = "lower_case"
    LENGTH_FIRST = "length_first"


def findFileList(dirPath="d:/", fileTypes=[".jpg", ".jpeg", ".bmp", ".png"], maxFileNum=None, sort_type=SortType.NO_SORT):
    assert os.path.exists(dirPath)
    fileList = []
    count = 0
    for s in os.listdir(dirPath):
        newDir = os.path.join(dirPath, s)
        if os.path.isfile(newDir):
            if fileTypes != None:
                if os.path.splitext(newDir)[1].lower() in fileTypes:
                    fileList.append(newDir)
                    count += 1
            else:
                fileList.append(newDir)
                count += 1
            if count % 1000 == 1:
                print2("\r%-s    " % s, textColor="gray", end="")
            if maxFileNum != None and count >= maxFileNum:
                break
    print2("\ncount=%d" % count, textColor="green")
    if sort_type == SortType.LENGTH_FIRST:
        fileList.sort(
            key=cmp_to_key(
                lambda x, y: (1 if (str.lower(x) > str.lower(y)) else (-1 if (str.lower(x) < str.lower(y)) else 0))
                if (len(x) == len(y))
                else (len(x) - len(y))
            )
        )
    elif sort_type == SortType.LOWER_CASE:
        fileList.sort(key=str.lower)
    return fileList


def printArray(a, name="a", fmt="%6.3f "):
    a = np.asarray(a, np.float32)
    s = a.shape
    if len(s) > 4:
        s = [x for x in a.shape if x > 1]
        print2("reshape array (%s) from %s => %s" % (name, str(list(a.shape)), str(s)), textColor="red")
        a = a.reshape(s)
    assert len(s) <= 4
    if len(s) == 1:
        printf("%s[%d]=\n    " % (name, s[0]), textColor="cyan")
        for x in range(s[0]):
            color = "green" if a[x] < 0 else ("gray" if a[x] == 0 else "red")
            printf(fmt % a[x], textColor=color)
        print()
    elif len(s) == 2:
        print2("%s[%dx%d]=" % (name, s[0], s[1]), textColor="cyan")
        for y in range(s[0]):
            printf("    ")
            for x in range(s[1]):
                color = "green" if a[y][x] < 0 else ("gray" if a[y][x] == 0 else "red")
                printf(fmt % a[y][x], textColor=color)
            print()
        print()
    elif len(s) == 3:
        print2("%s[%dx%dx%d]=" % (name, s[0], s[1], s[2]), textColor="cyan")
        for z in range(s[0]):
            printf("    ")
            for y in range(s[1]):
                printf("[ ")
                for x in range(s[2]):
                    color = "green" if a[z][y][x] < 0 else ("gray" if a[z][y][x] == 0 else "red")
                    printf(fmt % a[z][y][x], textColor=color)
                printf("] ")
            print()
        print()
    elif len(s) == 4:
        print2("%s[%dx%dx%dx%d]=" % (name, s[0], s[1], s[2], s[3]), textColor="cyan")
        for n in range(s[0]):
            print("..(%d)" % n)
            # print("  [")
            for z in range(s[1]):
                printf("    ")
                for y in range(s[2]):
                    printf("[ ")
                    for x in range(s[3]):
                        color = "green" if a[n][z][y][x] < 0 else ("gray" if a[n][z][y][x] == 0 else "red")
                        printf(fmt % a[n][z][y][x], textColor=color)
                    printf("] ")
                print("")
            # print("  ]")
        print()


def saveRaw(a, filename, isPrint=True):
    s = [x for x in a.shape if x > 1]
    assert len(s) == 2
    a = a.reshape(s)
    # a=np.array()
    name = os.path.splitext(filename)
    filename = "%s_%dx%d%s" % (name[0], s[1], s[0], name[1])
    print_debug("saveRaw_newName=%s" % filename, textColor="green", isPrint=isPrint)
    fp = open(filename, "wb")
    if fp == None:
        print2("Can not create '%s'" % filename, textColor="red")
        return False
    fp.write(a.tobytes())
    fp.close()
    return True


def saveData(filename, a, isPrint=True):
    fp = open(filename, "wb")
    if fp == None:
        print2("Can not create '%s'" % filename, textColor="red")
        return False
    fp.write(a.tobytes())
    fp.close()
    return True


def saveArray(a, filename, fmt="%6.3f ", isPrint=True):
    a = np.asarray(a, np.float32)
    s = a.shape
    name = "a"
    if len(s) >= 4:
        s = [x for x in a.shape if x > 1]
        print2("reshape array (%s) from %s => %s" % (name, str(list(a.shape)), str(s)), textColor="red", isPrint=isPrint)
        a = a.reshape(s)
    assert len(s) <= 4
    r = ""
    if len(s) == 1:
        print2("  save %s[%d] to '%s'" % (name, s[0], filename), textColor="cyan", isPrint=isPrint)
        for x in range(s[0]):
            r += fmt % a[x]
        r += "\n"
    elif len(s) == 2:
        print2("  save %s[%dx%d] to '%s'" % (name, s[0], s[1], filename), textColor="cyan", isPrint=isPrint)
        for y in range(s[0]):
            for x in range(s[1]):
                r += fmt % a[y][x]
            r += "\n"
        r += "\n"
    elif len(s) == 3:
        print2("  save %s[%dx%dx%d] to '%s'" % (name, s[0], s[1], s[2], filename), textColor="cyan", isPrint=isPrint)
        for z in range(s[0]):
            for y in range(s[1]):
                r += "["
                for x in range(s[2]):
                    r += fmt % a[z][y][x]
                r += "]"
            r += "\n"
        r += "\n"
    elif len(s) == 4:
        print2("  save %s[%dx%dx%dx%d] to '%s'" % (name, s[0], s[1], s[2], s[3], filename), textColor="cyan", isPrint=isPrint)
        for n in range(s[0]):
            r += "..(%d)\n" % n
            for z in range(s[1]):
                for y in range(s[2]):
                    r += "["
                    for x in range(s[3]):
                        r += fmt % a[n][z][y][x]
                    r += "]"
                r += "\n"
        r += "\n"
    fp = open(filename, "w")
    if fp == None:
        print2("Can not create '%s'" % filename, textColor="red", isPrint=isPrint)
        return False
    fp.write(r)
    fp.close()
    return True


def load_array_txt(a_shape, filename="a.txt"):
    # a_shape = [-1,15,26]
    with open(filename, "r") as fp:
        # arr = list(map(float, fp.read().replace('[',' ').replace(']', ' ').split()))
        arr = list(map(float, fp.read().translate(str.maketrans("[],", "   ")).split()))
    # print(arr)
    a = np.asarray(arr, np.float32).reshape(a_shape)
    return a


def load_array_bin(a_shape, filename="a.bin"):
    # a_shape = [-1,15,26]
    a_size = a_shape[-2] * a_shape[-3] if (a_shape[-1] == 1) else a_shape[-1] * a_shape[-2]
    with open(filename, "rb") as fp:
        arr = fp.read()
    type_size = len(arr) // (a_size)
    assert len(arr) % (a_size) == 0
    if type_size == 4:
        a = np.frombuffer(arr, np.float32).reshape(a_shape)
    else:
        a = np.asarray(np.frombuffer(arr, np.float64), np.float32).reshape(a_shape)
    return a


def load_array(a_shape, filename="a.txt"):
    if os.path.splitext(filename)[1] == ".txt":
        return load_array_txt(a_shape, filename)
    return load_array_bin(a_shape, filename)


_num_dict = {
    "0": "111101101101111",
    "1": "010010010010010",
    "2": "111001111100111",
    "3": "111001111001111",
    "4": "101101111001001",
    "5": "111100111001111",
    "6": "111100111101111",
    "7": "111001001001001",
    "8": "111101111101111",
    "9": "111101111001111",
    "A": "010111101111101",
    "B": "110101110101110",
    "C": "011100100100011",
    "D": "110101101101110",
    "E": "111100111100111",
    "F": "111100111100100",
    "G": "111100101101111",
    "H": "101101111101101",
    "I": "111010010010111",
    "J": "001001001101111",
    "K": "101101110101101",
    "L": "100100100100111",
    "M": "101111111101101",
    "N": "101111111111101",
    "O": "010101101101010",
    "P": "111101111100100",
    "Q": "111101111001001",
    "R": "111101110101101",
    "S": "011100010001110",
    "T": "111010010010010",
    "U": "101101101101011",
    "V": "101101101010010",
    "W": "101101111111101",
    "X": "101101010101101",
    "Y": "101101010010010",
    "Z": "111001010100111",
    "=": "000111000111000",
    "*": "000101010101000",
    ":": "000010000010000",
    "|": "010010010010010",
    ".": "000000000000010",
    "+": "000010111010000",
    "-": "000000111000000",
    " ": "000000000000000",
    "/": "001001010100100",
    '"': "101101000000000",
    "%": "101001010100101",
    "'": "010010000000000",
    "(": "010100100100010",
    ")": "010001001001010",
    "<": "001010100010001",
    ">": "100010001010100",
    "?": "110001010000010",
    "[": "011010010010011",
    "\\": "100100010001001",
    "]": "110010010010110",
    "^": "010101000000000",
    "_": "000000000000111",
    "`": "100010000000000",
    "{": "011010110010011",
    "}": "110010011010110",
    ",": "000000000010010",
    "!": "010010010000010",
    "~": "000011101000000",
    "#": "010111010111010",
    "@": "111101111100111",
    "$": "011110111011110",
    "&": "111101010111001",
}


def np_set_num_gray(gray_arr, left_top, num, text_color=0, back_color="mean", fmt="%6.3f", back_mixed=False):
    """
    gray_arr: an 2 dim np.array()
    left_top: [left, top]
    num: float type, int type or string type
    text_color: integer from 0 to 255
    back_color: integer from 0 to 255 or 'mean' or None
    """
    if type(num) != type(""):
        num = str(num) if (type(num) == type(0)) else fmt % num
    assert type(num) == type("")
    num = num.upper()
    assert len(gray_arr.shape) == 2

    [char_w, char_h] = [3, 5]
    [w, h] = [gray_arr.shape[1], gray_arr.shape[0]]
    [y_min, y_max] = [left_top[1], min(left_top[1] + char_h + 2, h)]
    [x_min, x_max] = [left_top[0], min(left_top[0] + len(num) * (char_w + 1) + 1, w)]
    mean_color = np.mean(gray_arr[y_min:y_max, x_min:x_max])
    if back_color != None:
        back_color = mean_color if (back_color == "mean") else back_color
        if back_mixed:
            gray_arr[y_min:y_max, x_min:x_max] = (np.array(gray_arr[y_min:y_max, x_min:x_max], np.int32) + back_color) // 2
        else:
            gray_arr[y_min:y_max, x_min:x_max] = back_color
    if text_color == None:
        text_color = 240 if (mean_color < 128) else 10
    for i in range(len(num)):
        s = _num_dict[num[i]] if (num[i] in _num_dict) else _num_dict[" "]
        # print_debug("%s:"%num[i])
        [y_min, y_max] = [left_top[1] + 1, min(left_top[1] + 1 + char_h, h)]
        [x_min, x_max] = [min(left_top[0] + i * (char_w + 1) + 1, w), min(left_top[0] + (i + 1) * (char_w + 1), w)]
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                j = (y - y_min) * char_w + (x - x_min)
                if s[j] == "1":
                    gray_arr[y][x] = text_color
    return gray_arr


def np_set_num_color(rgb_arr, left_top, num, text_color=[0, 0, 0], back_color="mean", fmt="%6.3f", back_mixed=False):
    """
    rgb_arr: an 3 dim np.array()
    left_top: [left, top]
    num: float type, int type or string type
    text_color: integer from 0 to 255
    back_color: integer from 0 to 255 or 'mean' or None
    """
    if type(num) != type(""):
        num = str(num) if (type(num) == type(0)) else fmt % num
    assert type(num) == type("")
    num = num.upper()
    assert len(rgb_arr.shape) == 3 and rgb_arr.shape[2] == 3

    [char_w, char_h] = [3, 5]
    [w, h] = [rgb_arr.shape[1], rgb_arr.shape[0]]
    [y_min, y_max] = [left_top[1], min(left_top[1] + char_h + 2, h)]
    [x_min, x_max] = [left_top[0], min(left_top[0] + len(num) * (char_w + 1) + 1, w)]
    mean_color = np.mean(rgb_arr[y_min:y_max, x_min:x_max].reshape([-1, 3]), 0)
    if back_color != None:
        back_color = mean_color if (back_color == "mean") else back_color
        if back_mixed:
            rgb_arr[y_min:y_max, x_min:x_max] = (np.array(rgb_arr[y_min:y_max, x_min:x_max], np.int32) + back_color) // 2
        else:
            rgb_arr[y_min:y_max, x_min:x_max] = back_color
    if text_color == None:
        text_color = [(240 if (mean_color[i] < 128) else 10) for i in range(3)]
    for i in range(len(num)):
        s = _num_dict[num[i]] if (num[i] in _num_dict) else _num_dict[" "]
        # print_debug("%s:"%num[i])
        [y_min, y_max] = [left_top[1] + 1, min(left_top[1] + 1 + char_h, h)]
        [x_min, x_max] = [min(left_top[0] + i * (char_w + 1) + 1, w), min(left_top[0] + (i + 1) * (char_w + 1), w)]
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                j = (y - y_min) * char_w + (x - x_min)
                if s[j] == "1":
                    rgb_arr[y][x] = text_color
    return rgb_arr


def np_set_num(arr, left_top, num, text_color=None, back_color="mean", fmt="%6.3f", back_mixed=False):
    if len(arr.shape) == 3:
        return np_set_num_color(arr, left_top, num, text_color, back_color, fmt, back_mixed)
    return np_set_num_gray(arr, left_top, num, text_color, back_color, fmt, back_mixed)


class Loader:

    if os.path.exists(os.path.join(ssim_dll_path, ssim_dll_name)):
        print_debug("load '%s'" % (os.path.join(ssim_dll_path, ssim_dll_name)), textColor="golden")
        dll = np.ctypeslib.load_library(ssim_dll_name, ssim_dll_path)
    else:
        print_debug("load '%s' FAILED" % (os.path.join(ssim_dll_path, ssim_dll_name)), textColor="red")

    type_dict = {
        "int": ctypes.c_int,
        "float": ctypes.c_float,
        "double": ctypes.c_double,
        "void": None,
        "int32": ctypes.c_int32,
        "uint32": ctypes.c_uint32,
        "int16": ctypes.c_int16,
        "uint16": ctypes.c_uint16,
        "int8": ctypes.c_int8,
        "uint8": ctypes.c_uint8,
        "byte": ctypes.c_uint8,
        "char*": ctypes.c_char_p,
        "float*": np.ctypeslib.ndpointer(dtype="float32", ndim=1, flags="CONTIGUOUS"),
        "int*": np.ctypeslib.ndpointer(dtype="int32", ndim=1, flags="CONTIGUOUS"),
        "byte*": np.ctypeslib.ndpointer(dtype="uint8", ndim=1, flags="CONTIGUOUS"),
    }

    @staticmethod
    def get_function(res_type="float", func_name="PSNR_Byte", arg_types=["Byte*", "int", "int", "int", "Byte*"]):
        func = Loader.dll.__getattr__(func_name)
        func.restype = Loader.type_dict[res_type]
        func.argtypes = [Loader.type_dict[str.lower(x).replace(" ", "")] for x in arg_types]
        return func

    @staticmethod
    def get_function2(c_define="DLL_API float PSNR_Byte(const Byte* pSrcData, int step, int width, int height, OUT Byte* pDstData);"):
        r = re.search(r"(\w+)\s+(\w+)\s*\((.+)\)", c_define)
        assert r != None
        r = r.groups()
        print(r)
        arg_list = r[2].split(",")
        arg_types = []
        for a in arg_list:
            a_list = a.split()
            if "*" in a_list[-1]:
                arg = a_list[-1].split("*")[0] + "*" if (a_list[-1][0] != "*") else a_list[-2] + "*"
            else:
                arg = a_list[-3] + "*" if (a_list[-2] == "*") else a_list[-2]
            arg_types.append(arg)
        print_debug(arg_types, textColor="magenta")
        # print_debug('res_type=%s, func_name=%s, arg_types=%s'%(r[0], r[1], str(arg_types)), textColor='yellow')
        return Loader.get_function(r[0], r[1], arg_types)

    @staticmethod
    def had_member(name="dll"):
        return name in Loader.__dict__.keys()


class DLL:
    @staticmethod
    def had_function(name="PSNR_Byte"):
        return name in DLL.__dict__.keys()

    if Loader.had_member("dll"):
        # float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal);
        PSNR_Byte = Loader.get_function("float", "PSNR_Byte", ["Byte*", "Byte*", "int", "int", "int", "int"])

        # float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal);
        PSNR_Float = Loader.get_function("float", "PSNR_Float", ["float*", "float*", "int", "int", "int", "double"])

        # float SSIM_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int win_size, int maxVal);
        SSIM_Byte = Loader.get_function("float", "SSIM_Byte", ["Byte*", "Byte*", "int", "int", "int", "int", "int"])

        # float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size, double maxVal);
        SSIM_Float = Loader.get_function("float", "SSIM_Float", ["float*", "float*", "int", "int", "int", "int", "double"])


def PSNR(x, y, max_value=None):
    [h, w, c] = x.shape
    x = x.astype("float32") if (x.dtype == "float64") else x
    y = y.astype("float32") if (y.dtype == "float64") else y
    if DLL.had_function("PSNR_Byte") and x.dtype == "uint8" and y.dtype == "uint8":
        return DLL.PSNR_Byte(x.reshape([-1]), y.reshape([-1]), w * c, w, h, 255 if (max_value == None) else int(max_value))
    if DLL.had_function("PSNR_Float") and x.dtype == "float32" and y.dtype == "float32":
        return DLL.PSNR_Float(x.reshape([-1]), y.reshape([-1]), w * c, w, h, 255.0 if (max_value == None) else float(max_value))
    return skimage.measure.compare_psnr(x, y, max_value)


def PSNR_slow(x_image, y_image, max_value=255.0):
    return skimage.measure.compare_psnr(x_image, y_image, max_value)


def SSIM(x, y, max_value=None, win_size=7):
    [h, w, c] = x.shape
    x = x.astype("float32") if (x.dtype == "float64") else x
    y = y.astype("float32") if (y.dtype == "float64") else y
    if DLL.had_function("SSIM_Byte") and x.dtype == "uint8" and y.dtype == "uint8":
        return DLL.SSIM_Byte(x.reshape([-1]), y.reshape([-1]), w * c, w, h, win_size, 255 if (max_value == None) else int(max_value))
    if DLL.had_function("SSIM_Float") and x.dtype == "float32" and y.dtype == "float32":
        return DLL.SSIM_Float(x.reshape([-1]), y.reshape([-1]), w * c, w, h, win_size, 255.0 if (max_value == None) else float(max_value))
    return skimage.measure.compare_ssim(x, y, win_size=win_size, data_range=max_value, multichannel=(x.ndim > 2))


def SSIM_slow(x_image, y_image, max_value=255.0, win_size=7, use_sample_covariance=True):
    x = np.asarray(x_image, np.float32)
    y = np.asarray(y_image, np.float32)
    return skimage.measure.compare_ssim(x, y, win_size=win_size, data_range=max_value, multichannel=(x.ndim > 2))
