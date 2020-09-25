
from PyQt5 import QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):

    def __init__(self):
        
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        
        FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        
        FigureCanvas.updateGeometry(self)

class MplWidget(QtWidgets.QWidget):
    
    def __init__(self, parent=None):
        
        QtWidgets.QWidget.__init__(self, parent)
        
        self.canvas = MplCanvas()
        
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        
        self.vbl = QtWidgets.QVBoxLayout()
        
        self.vbl.addWidget(self.canvas)
        
        self.vbl.addWidget(self.mpl_toolbar)  #posizionare al centro TODO
        
        self.setLayout(self.vbl)
