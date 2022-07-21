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
    #M1 = midpoint(A, B)
    #M2 = midpoint(A, C)
    #n1 = normal_vector(A, B)
    #n2 = normal_vector(A, C)
    #O = find_intersection(M1, n1, M2, n2)
    #return O
    sin_a = angle_sin(B, A, C)
    cos_a = angle_cosinus(B, A, C)
    sin_b = angle_sin(A, B, C)
    cos_b = angle_cosinus(A, B, C)
    sin_c = angle_sin(B, C, A)
    cos_c = angle_cosinus(B, C, A)
    sin_2a = 2 * sin_a * cos_a
    sin_2b = 2 * sin_b * cos_b
    sin_2c = 2 * sin_c * cos_c
    I = (A * sin_2a + B * sin_2b + C * sin_2c) / (sin_2a + sin_2b + sin_2c)
    return I

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
    """ Координаты точки, которая делит отрезок AB в отношении k:1, считая от A """
    N = np.array(
        [
            (A[0] + B[0] * k) / (1 + k),
            (A[1] + B[1] * k) / (1 + k),
            0
        ]
    )
    return N

def tangent_dots(c, A):
    """ Координаты 2 точек, в которых касательные, проведённые из точки A, касаются окружности c.
    A лежит вне окружности c """
   """Автор функции: Love Math (clck.ru/X9bbP)"""
    c_x, c_y = c.get_x(), c.get_y()
    A_x, A_y = A[0], A[1]
    d_x, d_y = A_x - c_x, A_y - c_y
    r = c.radius
    if d_x ** 2 + d_y ** 2 > r ** 2:
        minus_b2 = -d_x * d_y
        sqrt_D4 = r * np.sqrt(d_x ** 2 + d_y ** 2 - r ** 2)
        a = r ** 2 - d_x ** 2
        if abs(a) > 1e-15:
            k1 = (minus_b2 + sqrt_D4) / a
            k2 = (minus_b2 - sqrt_D4) / a
            x1 = (c_x + k1 * (k1 * A_x - d_y)) / (1 + k1 ** 2)
            x2 = (c_x + k2 * (k2 * A_x - d_y)) / (1 + k2 ** 2)
            y1 = k1 * (x1 - A_x) + A_y
            y2 = k2 * (x2 - A_x) + A_y
        else:
            a = r ** 2 - d_y ** 2
            if abs(a) > 1e-15:
                k1 = (minus_b2 + sqrt_D4) / a
                k2 = (minus_b2 - sqrt_D4) / a
                y1 = (c_y + k1 * (k1 * A_y - d_x)) / (1 + k1 ** 2)
                y2 = (c_y + k2 * (k2 * A_y - d_x)) / (1 + k2 ** 2)
                x1 = k1 * (y1 - A_y) + A_x
                x2 = k2 * (y2 - A_y) + A_x
            else:
                x1 = c_x
                y2 = c_y
                if A_y > c_y:
                    y1 = c_y + r
                else:
                    y1 = c_y - r
                if A_x > c_x:
                    x2 = c_x + r
                else:
                    x2 = c_x - r
        return [np.array([x1, y1, 0]), np.array([x2, y2, 0])]
    else:
        return (np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]))

def tangent_dots_for_2_circles(c1, c2):
    """ Координаты 4 точек, в которых общие касательные касаются окружностей c1 и c2.
    На выходе первые 2 точки в списке - для окружности c1, последние 2 точки - для окружности c2 """
    """Автор функции: Love Math (clck.ru/X9bbP)"""
    r1, r2 = c1.radius, c2.radius
    c1_c, c2_c = c1.get_center(), c2.get_center()
    if abs(r1 - r2) < 1e-15:
        normal_c = normal_vector(c1_c, c2_c)
        dot1 = c1_c + normal_c / np.linalg.norm(normal_c) * r1
        dot2 = c1_c - normal_c / np.linalg.norm(normal_c) * r1
        dot3 = c2_c + normal_c / np.linalg.norm(normal_c) * r2
        dot4 = c2_c - normal_c / np.linalg.norm(normal_c) * r2
    else:
        if r1 - r2 <= -1e-15:
            cs_c, cl_c = c1_c, c2_c
            rs, rl = r1, r2
        else:
            cs_c, cl_c = c2_c, c1_c
            rs, rl = r2, r1
        d = distance(c1_c, c2_c)
        shift = rs * d / (rl - rs)
        A = extend_segment(cl_c, cs_c, shift)
        [dot1, dot2] = tangent_dots(c1, A)
        [dot3, dot4] = tangent_dots(c2, A)
    return [dot1, dot2, dot3, dot4]

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
