#helper functions
def Diff(li1, li2):
    """
    Difference of sets
    """
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

def flip(p):
    """
    Biased coin flipper
    """
    return 1 if random.random() < p else 0

def sample_trunc_normal(mean, sd, lb = 0, ub = 1):
    """
    Sample a truncated normal random variable with specified mean, sd, lower bound and upper bound.
    """
    valid = False
    while not valid:
        s = np.random.normal(mean, sd, 1)[0]
        if s >= lb and s <= ub:
            valid = True
  
    return s