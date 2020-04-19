import numpy as np

def gradient_descent(x, y):
    m = b = 0
    iter = 100000
    n = len(x)
    learn_rate = 0.001   # our assumption
    for i in range(iter):
        yp = m * x + b
        cost = (1/n) * sum([val**2 for val in (y-yp)])
        md = -(2/n)*sum(x*(y-yp))
        bd = -(2/n)*sum(y-yp)
        m = m - learn_rate * md
        b = b - learn_rate * bd
        print("m: {}, b: {}, iteration: {}, cost: {}".format(m, b, i, cost))



x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)
