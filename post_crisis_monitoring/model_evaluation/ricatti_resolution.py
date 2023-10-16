import math


def diff_equation_compute_t_value(t, p, q, r):
    final_value = r

    if t >= 2:
        for i in range(1, t):
            final_value = diff_equation_recursion(final_value, p, q, r)
    return final_value


def diff_equation_recursion(previous_value, p, q, r):
    a = diff_equation_a(p, q, r)
    b = diff_equation_b(p, q, r)
    c = diff_equation_c(p, q, r)
    d = diff_equation_d(p, q, r)

    return (a * previous_value + b) / (c * previous_value + d)


def diff_equation_a(p, q, r):
    return r - q - p * r


def diff_equation_b(p, q, r):
    return q


def diff_equation_c(p, q, r):
    return - p


def diff_equation_d(p, q, r):
    return 1


def diff_equation_R(p, q, r):
    a = diff_equation_a(p, q, r)
    b = diff_equation_b(p, q, r)
    c = diff_equation_c(p, q, r)
    d = diff_equation_d(p, q, r)

    return (a * d - b * c) / (a + d) ** 2


def diff_equation_postive_root(p, q, r):
    R = diff_equation_R(p, q, r)

    solution = (1 + math.sqrt(1 - 4 * R)) / 2
    return solution


def diff_equation_negative_root(p, q, r):
    R = diff_equation_R(p, q, r)

    solution = (1 - math.sqrt(1 - 4 * R)) / 2
    return solution
