from jax import jit


def filter_super(X, thresh):
    '''
    Find superlevel set of X at threshold thresh
    '''
    return X > thresh


def filter_sub(X, thresh):
    '''
    Find sublevel set of X at threshold thresh
    '''
    return X < thresh


# jit wrapped around level set functions
filter_super_jit = jit(filter_super)
filter_sub_jit = jit(filter_sub)
