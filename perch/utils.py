import os
import subprocess
from jax import jit
import jax.numpy as jnp

CMD = '''
on run argv
  display notification (item 2 of argv) with title (item 1 of argv)
end run
'''
def _notify(title, text):
    '''
    Send a notification to the user
    '''
    subprocess.call(['osascript', '-e', CMD, title, text])

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


