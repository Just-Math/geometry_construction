from manim import *
import numpy as mp

""" Вычисление длин, векторов, углов """
def distance(A, B):
    a = np.linalg.norm(B - A)
    return a

def perimeter_triangle(A, B, C):
    #A, B, C - координаты вершин треугольника с 
    AB = distance(B, A)
    BC = distance(C, B)
    AC = distance(C, A)
    P = AB + BC + AC
    return P

def area_triangle(A, B, C):
    #Возращает значение площади треугольника
    AB = distance(B, A)
    BC = distance(C, B)
    AC = distance(C, A)
    p = perimeter_triangle(A, B, C)/2
    S = (p * (p - AB) * (p - BC) * (p - AC)) ** 0.5
    return S

def inradius(A, B, C):
    #Возвращает радиус вписанной окружности
    S = area_triangle(A, B, C)
    p = perimeter_triangle(A, B, C)/2
    r = S / p
    return r

def circumradius(A, B, C):
    #Возращает радиус описанной окружности
    a = distance(A, B)
    b = distance(B, C)
    c = distance(A, C)
    S = area_triangle(A, B, C)
    R = (a * b * c) / (4 * S)
    return R

def exradius(A, B, C):
    #Возвращает значение радиуса вневписанной окружности
    #центр которой напротив вершины A
    S = area_triangle(A, B, C)
    p = perimeter_triangle(A, B, C) / 2
    a = distance(B, C)
    R_a = S / (p - a)
    return R_a

def normal_vector(A, B):
    #Возращает координаты вектора нормали прямой AB
    n = np.array([B[1] - A[1], A[0] - B[0], 0])
    return n

def direction_vector(A, B):
    #направляющей вектор прямой AB
    c = np.array([B[0] - A[0],B[1] - A[1], 0])
    return c

def angle_cosinus(A, B, C):
    #Возращает значение косинуса угла ABC
    a = distance(B, C)
    b = distance(C, A)
    c = distance(B, A)   
    cos = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    return cos

def angle_sin(A, B, C):
    #Возращает значение синуса угла ABC
    cos = angle_cosinus(A, B, C)
    sin = (1 - cos ** 2) ** 0.5
    return sin

def angle_tan(A, B, C):
    #Возвращает тангенс угла ABC
    cos = angle_cosinus(A, B, C)
    sin = angle_sin(A, B, C)
    tan = sin / cos
    return tan

def angle_cot(A, B, C):
    #Возвращает котангенс угла ABC
    cot = 1 / (angle_tan(A, B, C))
    return cot

"""Замечательные точки """

def incenter(A, B, C):
    #Возвращает координаты инцентра
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    I = (a * A + b * B + c * C)/(a + b + c)    
    return I

def circumcenter(A, B, C):
    #Центр окружности, описанной около треугольника
    M1 = midpoint(A, B)
    M2 = midpoint(A, C)
    n1 = normal_vector(A, B)
    n2 = normal_vector(A, C)
    O = find_intersection(M1, n1, M2, n2)
    return O

def point_to_line_projection(C, A, B):
    #Возвращает координаты проекции точки C на прямую AB
    n = normal_vector(A, B)
    a = direction_vector(A, B)
    H = find_intersection(C, n, A, a)
    return H

def eccentric(A, B, C):
    #Возвращает координаты вневписанной окружности
    #Напротив вершины A
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    I_A = (b * B + c * C - a * A) / (b + c - a)
    return I_A

def nagel_point(A, B, C):
    #Возвращает координаты точки Нагеля
    p = perimeter_triangle(A, B, C) / 2
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    N = ( (p - a) * A + (p - b) * B + (p - c) * C )/(-p)
    return N

def center_mass(A, B, C):
    #Возращает координаты центра масс треугольника ABC
    Z = (A + B + C) / 3
    return Z

def orthocenter(A, B, C):
    #Возвращает координаты ортоцентра треугольника
    H1 = point_to_line_projection(A, B, C)    
    H2 = point_to_line_projection(B, A, C) 
    H = find_intersection(A, H1 - A, B, H2 - B)   
    return H

def center_O9(A, B, C):
    #Возвращает центр окружности девяти точек
    O = circumcenter(A, B, C)
    H = orthocenter(A, B, C)
    O9 = 0.5 * (O + H)
    return O9

def nagel_point(A, B, C):
    #Возвращает координаты точки Нагеля
    p = perimeter_triangle(A, B, C) / 2
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    N = ( (p - a) * A + (p - b) * B + (p - c) * C )/(-p)
    return N

def bisectors(A, B, C):
    #Возвращает основание биссектрисы напротив первой вершины
    b = distance(A, C)
    c = distance(A, B)
    L = (b * B + c * C)/(b + c)
    return L

def split_segment(A, B, k):
    #Деление отрезка AB в отношении k
    N = np.array(
        [
            (A[0] + B[0] * k) / (1 + k),
            (A[1] + B[1] * k) / (1 + k),
            0
        ]
    )
    return N

def kasanie(c, A):
    O = c.get_center()
    r = c.radius
    AO = distance(A, O)
    AC = (AO ** 2 - r ** 2) ** 0.5
    cosA = AC / AO
    sinA = r / AO
    x1 = A[0] * cosA - A[1] * sinA
    y1 = A[0] * sinA + A[1] * cosA
    O1 = np.array([x1, y1, 0])
    vAO1 = O1 - A
    nAO1 = normal_vector(A, O1) 
    C = find_intersection(A, vAO1, O, nAO1)
    return C


"""Оформление чертежа """
def extend_segment(A, B, k):
    #Продлить отрезок AB на k единиц
    # за точку B
    dAB = distance(A, B)
    vAB = direction_vector(A, B) / dAB
    C = B + vAB * k
    return C

def mark_line(A, B, a):
    #Отметить отрезок AB штрихом
    # а - количество штрихов
    m = midpoint(A, B)
    n = normal_vector(A, B) / (np.linalg.norm(B - A))
    p = np.array([B[0] - A[0],B[1] - A[1], 0]) / (np.linalg.norm(B - A))
    lines = VGroup()
    F1 = m + 0.1 * n
    F2 = m - 0.1 * n
    if a > 0:
        s1 = Line(F1, F2)
        lines.add(s1)
        for i in range(a):
            s2 = Line(F1, F2).shift(i *0.3 * 0.2 * p)
            lines.add(s2)
    lines.move_to(m)
    return lines
