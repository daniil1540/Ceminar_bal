from numpy import array, zeros, argmin, inf
from numba import jit

pervoe = list(map(int, input().split()))
vtoroe = list(map(int, input().split()))
function_name = input()


# Временные ряды


@jit(nopython=True)
def nulevaya():
    return zeros((dlinapervo + 1, dlinavtoro + 1))


# Просто использовние модуля numba


dlinapervo, dlinavtoro = len(pervoe), len(vtoroe)
nulev = nulevaya()
nulev[0, 1:] = inf
nulev[1:, 0] = inf
Mat_Rast = nulev[1:, 1:]


# Создание матриц, которые нам понадобятся


def MatRastoynii(dlinapervo, dlinavtoro, function_name):
    if function_name == 'manhattan':
        for i in range(dlinapervo):
            for j in range(dlinavtoro):
                Mat_Rast[i, j] = abs(pervoe[i] - vtoroe[j])
        return Mat_Rast
    elif function_name == 'evklid':
        for i in range(dlinapervo):
            for j in range(dlinavtoro):
                Mat_Rast[i, j] = (pervoe[i] - vtoroe[j]) ** 2
        return Mat_Rast


# Создаём исходную матрицу расстояний с помощью алгоритма вычисления манхэтоских расстояний
# На этом шаге мы можем использовать евклидово расстояние
Mat_Trans = MatRastoynii(dlinapervo, dlinavtoro, function_name)
nepoteryanya_rast = Mat_Rast.copy()
for i in range(dlinapervo):
    for j in range(dlinavtoro):
        Mat_Trans[i, j] += min(nulev[i, j], nulev[i, j + 1], nulev[i + 1, j])


# Создаём матрицу трансформаций


@jit(nopython=True)
def plusballi():
    koor_x, koor_y = array(nulev.shape) - 2
    return koor_x, koor_y


# Просто использовние модуля numba


# Самый короткий путь
koor_x, koor_y = plusballi()
oc_x, oc_y = [koor_x], [koor_y]
while koor_x > 0 or koor_y > 0:
    tb = argmin((nulev[koor_x, koor_y], nulev[koor_x, koor_y + 1], nulev[koor_x + 1, koor_y]))
    if tb == 0:
        koor_x -= 1
        koor_y -= 1
    elif tb == 1:
        koor_x -= 1
    else:
        koor_y -= 1
    oc_x.insert(0, koor_x)
    oc_y.insert(0, koor_y)

put = []
for i in range(len(oc_x)):
    put.append([oc_x[i], oc_y[i]])

print(nepoteryanya_rast)
# Исходная матрица расстояний
print(*put)
# Минимальный путь трансформации
print(Mat_Trans)
# Матрица трансформаций
print(Mat_Trans[-1, -1])
# Расстояние между последовательностями
