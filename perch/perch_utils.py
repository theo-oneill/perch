import os
import subprocess
CMD = '''
on run argv
  display notification (item 2 of argv) with title (item 1 of argv)
end run
'''
def notify(title, text):
  subprocess.call(['osascript', '-e', CMD, title, text])

def seg_struc(struc,img_jnp=None):
    '''
    Segment a structure using the structure's segment method.
    '''
    if struc.saved_indices_exist():
      print('loading')
      struc.load_indices()
    if not struc.saved_indices_exist():
      struc.compute_segment(img=img_jnp)
      struc.save_indices()
    struc.clear_indices()
