from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt

click = [None,None]
release = [None,None]

def line_select_callback(eclick,erelease):
    click[:] = eclick.xdata, eclick.ydata
    release[:] = erelease.xdata, erelease.ydata
    
def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        toggle_selector.RS.set_active(True)
        
        