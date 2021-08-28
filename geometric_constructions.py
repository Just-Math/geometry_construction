from manim import *
import numpy as np
 
""" Вычисление длин, векторов, углов """

def distance(A, B):
    """ Расстояние между точками A и B """
    a = np.linalg.norm(B - A)
    return a

def perimeter_triangle(A, B, C):
    """ Периметр треугольника ABC """
    AB = distance(A, B)
    BC = distance(B, C)
    AC = distance(A, C)
    P = AB + BC + AC
    return P

def area_triangle(A, B, C):
    """ Площадь треугольника ABC """
    AB = distance(A, B)
    BC = distance(B, C)
    AC = distance(A, C)
    p = perimeter_triangle(A, B, C)/2
    S = (p * (p - AB) * (p - BC) * (p - AC)) ** 0.5
    return S

def inradius(A, B, C):
    """ Радиус окружности, вписанной в треугольник ABC """
    S = area_triangle(A, B, C)
    p = perimeter_triangle(A, B, C)/2
    r = S / p
    return r

def circumradius(A, B, C):
    """ Радиус окружности, описанной около треугольника ABC """
    a = distance(A, B)
    b = distance(B, C)
    c = distance(A, C)
    S = area_triangle(A, B, C)
    R = (a * b * c) / (4 * S)
    return R

def exradius(A, B, C):
    """ Радиус вневписанной окружности с центром напротив вершины A """
    S = area_triangle(A, B, C)
    p = perimeter_triangle(A, B, C) / 2
    a = distance(B, C)
    R_a = S / (p - a)
    return R_a

def normal_vector(A, B):
    """ Вектор нормали прямой AB """
    n = np.array([B[1] - A[1], A[0] - B[0], 0])
    return n

def direction_vector(A, B):
    """ Направляющий вектор прямой AB """
    v = np.array([B[0] - A[0], B[1] - A[1], 0])
    return v

def angle_cosinus(A, B, C):
    """ Косинус угла ABC """
    a = distance(B, C)
    b = distance(C, A)
    c = distance(B, A)   
    cos = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    return cos

def angle_sin(A, B, C):
    """ Синус угла ABC """
    cos = angle_cosinus(A, B, C)
    sin = (1 - cos ** 2) ** 0.5
    return sin

def angle_tan(A, B, C):
    """ Тангенс угла ABC """
    cos = angle_cosinus(A, B, C)
    sin = angle_sin(A, B, C)
    tan = sin / cos
    return tan

def angle_cot(A, B, C):
    """ Котангенс угла ABC """
    cot = 1 / (angle_tan(A, B, C))
    return cot

"""Замечательные точки """

def incenter(A, B, C):
    """ Координаты центра окружности, вписанной в треугольник ABC """
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    I = (a * A + b * B + c * C)/(a + b + c)    
    return I

def circumcenter(A, B, C):
    """ Координаты центра окружности, описанной около треугольника ABC """
    M1 = midpoint(A, B)
    M2 = midpoint(A, C)
    n1 = normal_vector(A, B)
    n2 = normal_vector(A, C)
    O = find_intersection(M1, n1, M2, n2)
    return O

def point_to_line_projection(C, A, B):
    """ Координаты проекции точки C на прямую AB """
    n = normal_vector(A, B)
    a = direction_vector(A, B)
    H = find_intersection(C, n, A, a)
    return H

def eccentric(A, B, C):
    """ Координаты центра вневписанной окружности с центром напротив вершины A """
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    I_A = (b * B + c * C - a * A) / (b + c - a)
    return I_A

def nagel_point(A, B, C):
    """ Координаты точки Нагеля """
    p = perimeter_triangle(A, B, C) / 2
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    N = ( (p - a) * A + (p - b) * B + (p - c) * C )/(-p)
    return N

def center_mass(A, B, C):
    """ Координаты центра масс треугольника ABC """
    Z = (A + B + C) / 3
    return Z

def orthocenter(A, B, C):
    """ Координаты ортоцентра треугольника ABC """
    H1 = point_to_line_projection(A, B, C)    
    H2 = point_to_line_projection(B, A, C) 
    H = find_intersection(A, H1 - A, B, H2 - B)   
    return H

def center_O9(A, B, C):
    """ Координаты центра окружности девяти точек для треугольника ABC """
    O = circumcenter(A, B, C)
    H = orthocenter(A, B, C)
    O9 = 0.5 * (O + H)
    return O9

def bisectors(A, B, C):
    """ Координаты основания биссектрисы угла A треугольника ABC """
    b = distance(A, C)
    c = distance(A, B)
    L = (b * B + c * C)/(b + c)
    return L

def split_segment(A, B, k):
    """ Координаты точки, которая делит отрезок AB в отношении 1:k, считая от A """
    N = np.array(
        [
            (A[0] + B[0] * k) / (1 + k),
            (A[1] + B[1] * k) / (1 + k),
            0
        ]
    )
    return N

"""Оформление чертежа """

def extend_segment(A, B, k):
    """ Координаты точки, лежащей на продолжении отрезка AB за точку B через k единиц """
    dAB = distance(A, B)
    vAB = direction_vector(A, B) / dAB
    C = B + vAB * k
    return C

def mark_line(A, B, a, width = 1, **kwargs):
    """ Штрихи, отмечающие отрезок AB
    а - количество штрихов """
    m = midpoint(A, B)
    n = normal_vector(A, B) / (np.linalg.norm(B - A))
    p = np.array([B[0] - A[0], B[1] - A[1], 0]) / (np.linalg.norm(B - A))
    lines = VGroup()
    F1 = m + 0.1 * n
    F2 = m - 0.1 * n
    if a > 0:
        s1 = Line(F1, F2, stroke_width = width * 0.7 * DEFAULT_STROKE_WIDTH, **kwargs)
        lines.add(s1)
        for i in range(a):
            s2 = Line(F1, F2, stroke_width = width * 0.7 * DEFAULT_STROKE_WIDTH, **kwargs).shift(i * width *.07 * p)
            lines.add(s2)
    lines.move_to(m)
    return lines
