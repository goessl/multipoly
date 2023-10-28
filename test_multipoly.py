import numpy as np
from multipoly import MultiPoly



if __name__ == '__main__':
    #single value
    a = ((1, 2), (3, 4), (5, 6))
    p = MultiPoly(a)
    assert np.isclose(p(np.pi, np.e), 260.33849829919226241991897829427872868844018150211046741506505724) #https://www.wolframalpha.com/input?i=260.33849829919226241991897829427872868844018150211046741506505724&assumption=%22ClashPrefs%22+-%3E+%7B%22Math%22%7D
    
    
    
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
    
    assert np.allclose(p.a, ((1, 2), (3, 4), (5, 6)))
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
