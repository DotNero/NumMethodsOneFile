import numpy as np
import matplotlib.pyplot as plt

al = 2

be = 1
#whatthefuck
ga = 1

n_ = 40

eps_ = (1 / n_) ** 3


def create_list(llim, rlim, n):
    step = (rlim - llim) / (n - 1)
    return list(map(lambda i: llim + step * i, range(n)))


def p(x, gamma):
    return 1 + np.power(x, gamma)


def q(x):
    return x + 1


def u(x, alpha, betta):
    return np.power(x, alpha) * np.power(-1 * x + 1, betta)


def func(x):
    return x ** 6 - 3 * x ** 5 - 23 * x ** 4 + 46 * x ** 3 - 9 * x ** 2 - 19 * x + 7


def Jacobi(A: np.ndarray, b: np.ndarray, epsilon: float, x=None):
    n = len(A)
    if x is None:
        x = np.zeros(n)
    x_ = np.zeros(n)

    norm = float('inf')
    iters = 0
    while norm > epsilon:
        for i in range(n):
            x_[i] = b[i]
            for j in range(n):
                if i != j:
                    x_[i] -= A[i, j] * x[j]
            x_[i] /= A[i, i]
        x = np.copy(x_)
        norm = np.max(np.abs(A.dot(x) - b))
        iters += 1
    return x, iters


def Seidel(A: np.ndarray, b: np.ndarray, epsilon, x=None):
    n = len(A)
    if x is None:
        x = np.zeros(n)
    x_ = np.zeros(len(x))

    it = 0
    norm = float('inf')
    while norm > epsilon:
        for i in range(n):
            s1 = sum(A[i][j] * x_[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_[i] = (b[i] - s1 - s2) / A[i][i]
        norm = np.max(np.abs(A.dot(x_) - b))
        x = np.copy(x_)
        it += 1
    return x, it


def optimal_w(A: np.ndarray, b: np.ndarray, epsilon, x=None,  parts=20, left: float = 1, right: float = 2):
    if right - left > 0.01:
        if x is None:
            x = np.zeros(shape=(len(A),))
        step = (right - left) / parts
        min_iters = np.Infinity
        min_w = [left]
        w = left
        while w <= right:
            iters = Relaxation(A=A, b=b, x=x, w=w, epsilon=epsilon)[1]
            if iters < min_iters:
                min_w = [w]
                min_iters = iters
            elif iters == min_iters:
                min_w.append(w)
            w = w + step
        return min_w, min_iters
    return [left + (right - left) / 2]


def Relaxation(A: np.ndarray, b: np.ndarray, epsilon, x=None, w=1.8):
    n = len(A)
    if x is None:
        x = np.zeros(n)
    it = 0

    x_ = np.copy(x)

    norm = float('inf')
    while norm > epsilon:
        for i in range(n):
            s1 = sum(A[i][j] * x_[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_[i] = (1 - w) * x[i] + w * (b[i] - s1 - s2) / A[i][i]
        norm = np.max(np.abs(A.dot(x_) - b))
        x = np.copy(x_)
        it += 1
    return x, it


def generate_A(n, gamma):
    h = 1 / n
    a = np.array([p(i * h, gamma=gamma) for i in range(1, n + 1)])
    g = np.array([q(i * h) for i in range(1, n + 1)])

    A = np.array([np.array([a[0] + a[1] + np.power(h, 2) * g[0], -1 * a[1]] + [0 for i in range(2, n - 1)])])
    for i in range(1, n - 2):
        A = np.append(A, np.array([np.array(
            [0 for k in range(i - 1)] + [-1 * a[i], a[i] + a[i + 1] + np.power(h, 2) * g[i], -1 * a[i + 1]]
            + [0 for j in range(n - 1 - i - 2)])]), axis=1)

    A = np.append(A, np.array(
        [np.array([0 for i in range(n - 3)] + [-1 * a[-2], a[-2] + a[-1] + np.power(h, 2) * g[-1]])]), axis=1)
    return A.reshape((n - 1, n - 1))


def explore_w_by_n(gamma, eps, ns=None):
    print('Исследование w в зависимости от n')
    if ns is None:
        ns = [5 + i * 15 for i in range(7)]
    ws = []
    for n in ns:
        h = 1 / n
        A = generate_A(n, gamma=gamma)
        f = np.array([func(i * h) * np.power(h, 2) for i in range(1, n)])

        opt_w = optimal_w(A, f, epsilon=eps, parts=100)
        opt_ws = opt_w[0]
        ws.append(opt_ws[-1])
        print("n={}, h={}, iters={}, eps={}, w=[{}, {}], count={}".format(n, h, opt_w[1], eps, opt_ws[0], opt_ws[-1],
                                                                          len(opt_ws)))
    plt.plot(ns, ws)
    plt.xlabel('n')
    plt.ylabel('w')
    plt.title('Зависимость параметра w \n от размерности задачи n')
    plt.grid(True)
    plt.show()


def explore_w_by_e(alpha, beta, gamma, n, es=None):
    print('Исследование w в зависимости от epsilon')
    if es is None:
        es = [0.1 / (10 ** i) for i in range(0, 8)]
    ws = []
    for e in es:
        h = 1 / n
        A = generate_A(n, gamma=gamma)
        f = np.array([func(i * h) * np.power(h, 2) for i in range(1, n)])

        x = np.array(create_list(0, 1, n))
        opt_w = optimal_w(A, f, epsilon=e, parts=100)
        opt_ws = opt_w[0]
        res = Relaxation(A, f, w=opt_w[0][-1], epsilon=e)
        ws.append(opt_ws[-1])
        print("n={}, h={},iters={}, eps={}, w=[{}, {}], count={}, макс.ошибка={}".format(n, h, opt_w[1], e, opt_ws[0],
                                                                                         opt_ws[-1],
                                                                                         len(opt_ws), np.max(
                np.abs(res[0] - u(x, alpha, beta)[:-1]))))


#explore_w_by_n(1, eps, [10, 20, 30, 40, 50, 55, 60, 65, 70, 80, 90, 110])
#explore_w_by_e(1, 40, [1e-3, 1e-5, eps, 1e-6, 1e-7])


def explore_n_by_y(alpha, betta, gamma, n):
    print('Исследование количества итераций в зависимости от начального приближения y')
    ys = []
    val = [-1, -0.5, -0.1, -0.05, 0., 0.05, .1, 0.5, 1]
    for v in val:
        ys.append(np.array([v for i in range(n - 1)]))

    h = 1 / n
    x = np.array(create_list(0, 1, n))
    A = generate_A(n, gamma=gamma)
    f = np.array([func(i * h) * np.power(h, 2) for i in range(1, n)])

    eps = 1e-5
    print("n={}, eps={}, h={}, h^2={}".format(n, eps, h, h ** 2))

    print('метод Якоби')
    for i in range(len(ys)):
        y = np.copy(ys[i])

        res = Jacobi(A, f, x=y, epsilon=eps)
        print('y: ', y[0],
              'итераций={}, максимальная ошибка={}'.format(res[1], np.max(
                  np.abs(res[0] - u(x, alpha, betta)[:-1]))))

    print('метод Зейделя')
    for i in range(len(ys)):
        y = np.copy(ys[i])

        res = Seidel(A, f, x=y, epsilon=eps)
        print('y: ', y[0],
              'итераций={}, максимальная ошибка={}'.format(res[1], np.max(
                  np.abs(res[0] - u(x, alpha, beta)[:-1]))))

    print('метод Верхней релаксации')
    for i in range(len(ys)):
        y = np.copy(ys[i])
        w = optimal_w(A, f, x=y, epsilon=eps)[0][-1]
        print(w)
        res = Relaxation(A, f, x=y, w=w, epsilon=eps)
        print('y: ', y[0],
              'итераций={}, максимальная ошибка={}'.format(res[1], np.max(
                  np.abs(res[0] - u(x, alpha, beta)[:-1]))))


#explore_n_by_y()


def compare_methods(alpha, betta, gamma, eps):
    print('Сравнение трех методов')
    ns = [10 + i * 15 for i in range(5)]

    print("eps={}".format(eps))

    print('метод Якоби')
    j_iters = []
    for n in ns:
        h = 1 / n
        A = generate_A(n, gamma=gamma)
        f = np.array([func(i * h) * np.power(h, 2) for i in range(1, n)])
        x = np.array(create_list(0, 1, n))

        print("n={}, h={}, h^2={}".format(n, h, h ** 2))

        res = Jacobi(A, f, epsilon=eps)
        j_iters.append(res[1])
        print('итераций={}, максимальная ошибка={}'.format(res[1], np.max(
            np.abs(res[0] - u(x, alpha, betta)[:-1]))))
    plt.plot(ns, j_iters, label='метод Якоби')

    print('метод Зейделя')
    z_iters = []
    for n in ns:
        h = 1 / n
        A = generate_A(n, gamma=gamma)
        f = np.array([func(i * h) * np.power(h, 2) for i in range(1, n)])
        x = np.array(create_list(0, 1, n))

        print("n={}, h={}, h^2={}".format(n, h, h ** 2))

        res = Seidel(A, f, epsilon=eps)
        z_iters.append(res[1])
        print('итераций={}, максимальная ошибка={}'.format(res[1], np.max(
            np.abs(res[0] - u(x, alpha, betta)[:-1]))))
    plt.plot(ns, z_iters, label='метод Зейделя')

    print('метод Верхней релаксации')
    r_iters = []
    for n in ns:
        h = 1 / n
        A = generate_A(n, gamma=gamma)
        f = np.array([func(i * h) * np.power(h, 2) for i in range(1, n)])
        x = np.array(create_list(0, 1, n))

        print("n={}, h={}, h^2={}".format(n, h, h ** 2))

        w = optimal_w(A, f, epsilon=eps)[0][-1]
        res = Relaxation(A, f, epsilon=eps, w=w)
        r_iters.append(res[1])
        print('итераций={}, максимальная ошибка={}'.format(res[1], np.max(
            np.abs(res[0] - u(x, alpha, betta)[:-1]))))
    plt.plot(ns, r_iters, label='метод врехней релаксации')

    legend = plt.legend(loc='upper center', shadow=True, fontsize='medium')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.xlabel('n')
    plt.ylabel('количество итераций')
    plt.title('Сравнение методов')
    plt.grid(True)
    plt.show()

compare_methods(al, be, ga, eps_)


def explore_e_by_n(alpha=2, betta=1, gamma=1, ns=None, epss=None):
    print('Исследование epsilon в зависимости от n')
    if ns is None:
        ns = [5 + i * 15 for i in range(8)]
    if epss is None:
        epss = [0.1 / (10 ** i) for i in range(0, 8)]
    for n in ns:
        h = 1 / n
        A = generate_A(n, gamma=gamma)
        f = np.array([func(i * h) * np.power(h, 2) for i in range(1, n)])
        x = np.array(create_list(0, 1, n))

        print("n={}, h={}".format(n, h))

        print('метод Якоби')
        for eps in epss:
            res = Jacobi(A, f, epsilon=eps)
            print('eps={}, h^2={} , итераций={}, максимальная ошибка={}'.format(eps, h ** 2, res[1], np.max(
                np.abs(res[0] - u(x, alpha, betta)[:-1]))))

        print('метод Зейделя')

        for eps in epss:
            res = Seidel(A, f, epsilon=eps)
            print('eps={}, h^2={}, итераций={}, максимальная ошибка={}'.format(eps, h ** 2, res[1], np.max(
                np.abs(res[0] - u(x, alpha, betta)[:-1]))))
        print('метод Верхней релаксации')
        for eps in epss:
            w = optimal_w(A, f, epsilon=eps)[0][-1]
            print(w)
            res = Relaxation(A, f, w=w, epsilon=eps)
            print('eps={}, h^2={}, w={}, итераций={}, максимальная ошибка={}'.format(eps, h ** 2, w, res[1], np.max(
                np.abs(res[0] - u(x, alpha, betta)[:-1]))))


#explore_e_by_n()



