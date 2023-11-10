import numpy as np
from multipoly import *



#1+2pi+3e+4pie+5pi^2e
#https://www.wolframalpha.com/input?i=1%2B2π%2B3e%2B4πe%2B5π%5E2e
x = (np.pi, np.e)
c = ((1, 3),
     (2, 4),
     (0, 5))
assert np.isclose(polyvalnd(x, c), 183.7387991710541)
#swaped arguments
#https://www.wolframalpha.com/input?i=1%2B2e%2B3π%2B4eπ%2B5e%5E2π
assert np.isclose(polyvalnd((np.e, np.pi), c), 166.08730029519867)
#vectorisation
assert np.allclose(polyvalnd(((np.pi, np.e),
                              (np.e, np.pi)), c),
                   (polyvalnd((np.pi, np.e), c),
                    polyvalnd((np.e, np.pi), c)))



c = ((1, 3),
     (2, 4),
     (0, 5))
X, y = [], []
for _ in range(10):
    x = np.random.normal(size=2)
    X += [x]
    y += [polyvalnd(x, c)]
fit = polyfitnd(X, y, (2, 1))
assert np.allclose(c, fit)





#single value
a = ((1, 2), (3, 4), (5, 6))
p = MultiPoly(a)
assert np.isclose(p(np.pi, np.e), 260.33849829919226241991897829427872868844018150211046741506505724) #https://www.wolframalpha.com/input?i=260.33849829919226241991897829427872868844018150211046741506505724&assumption=%22ClashPrefs%22+-%3E+%7B%22Math%22%7D
#vectorisation
assert np.allclose(p([np.pi, np.e], [2, 3]), [p(np.pi, np.e), p(2, 3)])



#random factory
for _ in range(20):
    dim = np.random.randint(1, 5)
    deg = np.random.randint(0, 5, size=dim)
    p = MultiPoly.random(deg, offsets=np.random.choice([False, True]))
    x = np.random.normal(size=dim)
    p(x)



#single polynomial
f = lambda x, y: 1 + 2*(y-2) \
        + 3*(x-1) + 4*(x-1)*(y-2) \
        + 5*(x-1)**2 + 6*(x-1)**2*(y-2)

X, y = [], []
for _ in range(100):
    x = np.random.normal(size=2)
    X += [x.tolist()]
    y += [f(*x)]
p = MultiPoly.fit(X, y, (2, 1), (1, 2))

assert np.allclose(p.a, a)
for _ in range(20):
    x = np.random.normal(size=2)
    assert np.isclose(f(*x), p(x))



#random polynomials
for _ in range(100):
    dim = np.random.randint(1, 5)
    deg = np.random.randint(0, 3, dim)
    a = np.random.normal(size=deg+1)
    c = np.random.normal(size=dim)
    p = MultiPoly(a, c)
    
    X, y = [], []
    for _ in range(100):
        x = np.random.normal(size=p.dim)
        X += [x.tolist()]
        y += [p(x)]
    
    fit = MultiPoly.fit(X, y, deg, c)
    assert np.allclose(p.a, fit.a)
