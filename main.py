import os
import subprocess
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QMessageBox, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
import PyQt5.QtCore as QtCore
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import numpy as np
from scipy import signal, ndimage
import cv2
from Ui_tsa import Ui_MainWindow
from rectangle_selector import *
import matplotlib as plt
from datetime import date,datetime
from PIL import Image
import time

class ShowInterface(QtWidgets.QMainWindow,Ui_MainWindow):
    
    def __init__(self,parent=None):
        
        super(ShowInterface, self).__init__(parent)
        QtWidgets.QMainWindow.__init__(self)
        
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        
        def resource_path(relative_path):
            if hasattr(sys, '_MEIPASS'):
                return os.path.join(sys._MEIPASS, relative_path)
            return os.path.join(os.path.abspath("."), relative_path)
        
        self.true = QPixmap(resource_path('true.png'))
        self.false = QPixmap(resource_path('false.png'))
        self.empty = QPixmap(resource_path('empty.png'))
        
        #Inizializzo Croci Rosse
        self.ui.verification_folder.setPixmap(self.false)
        self.ui.verification_video.setPixmap(self.false)
        self.ui.verification_param.setPixmap(self.false)
        self.ui.verification_lockin.setPixmap(self.false)
        self.ui.verification_save.setPixmap(self.false)
        self.ui.verification_position.setPixmap(self.false)

        #variable
        self.video_count = 0
        self.lock_count = 0
        self.plot_count = 0
        self.limit = 0
        self.moltiplicator = 1
        self.ver_path = 0
        self.ver_temp = 0
        self.ver_crop = 0
        self.ver_cal = 0
        self.mean_temp = 0
        
        ver = '0.1.1'
        
        self.setWindowTitle('Thermal Stress Analysis Toolbox - ver_'+ver)
        
        self.setWindowIcon(QtGui.QIcon('logo.png'))

        #date
        self.ui.date.setText(str(date.today())+' - '+str(datetime.now().strftime('%H:%M:%S')))
        self.ui.version.setText('ver_'+ver)
        
        #LCD Font Color
        palette = self.ui.lcd.palette()
        palette.setColor(palette.Light, QtGui.QColor(0, 0, 0))
        palette.setColor(palette.Dark, QtGui.QColor(100, 100, 100))
        self.ui.lcd.setPalette(palette)
        self.ui.frame_start.setPalette(palette)
        self.ui.frame_end.setPalette(palette)
        
        #menubar
        self.ui.actionclose.triggered.connect(self.close)
        self.ui.actioncredit.triggered.connect(self.credit)
        self.ui.actionversion.triggered.connect(self.version)
        
        #disable button
        self.ui.param_box.setEnabled(0)
        self.ui.auto_freq.setEnabled(0)
        self.ui.load_freq.setEnabled(0)
        self.ui.set_button.setEnabled(0)
        self.ui.ROI_box.setEnabled(0)
        self.ui.crop_box.setEnabled(0)
        self.ui.resume.setEnabled(0)
        self.ui.accept.setEnabled(0)
        self.ui.param_box.setEnabled(0)
        self.ui.frame_slider.setEnabled(0)
        self.ui.lockin_button.setEnabled(0)
        self.ui.filtering_box.setEnabled(0)
        self.ui.gaussian_box.setEnabled(0)
        self.ui.calibration_box.setEnabled(0)
        self.ui.temporal_box.setEnabled(0)
        self.ui.name_box.setEnabled(0)
        self.ui.open_folder.setEnabled(0)
        self.ui.save_box.setEnabled(0)
        self.ui.cal_save.setEnabled(0)
        self.ui.progress_filter.setEnabled(0)
        self.ui.button_point.setEnabled(0)
        self.ui.button_line.setEnabled(0)
        self.ui.tabWidget.setTabEnabled(1,False)
        
    #menubar
    
    def close(self):
        sys.exit(app.exec_())  
        
    def credit(self):
        msg = QMessageBox()
        msg.setWindowTitle('Credit')
        msg.setText('DEVELOPERS:\n Lorenzo Capponi - lorenzocapponi@outlook.it \n Tommaso Tocci - tommaso.tocci@outlook.it \n\n Property of MMT Group (Univesity of Perugia, Department of Engineering)\n Email: mmt.unipg@gmail.com')
        
        x = msg.exec()
        
    def version(self):
        msg = QMessageBox()
        msg.setWindowTitle('Version')
        msg.setText('Beta version: 0.0.1 \n 12/11/2019')
        
        x = msg.exec()
        
#####################################################################################
#-------------------------LOAD VIDEO --------------------------------
#####################################################################################

    def disableProgress(self):
        if self.ui.median_filter.isChecked():
                self.ui.progress_filter.setEnabled(1)
        else:
            self.ui.progress_filter.setEnabled(0)

    def importVideo(self):
            
        self.videoPath_tup = QtWidgets.QFileDialog.getOpenFileName(self,'import video...',filter = '(*.sfmov)')   
        self.videoPath_str = ''.join(self.videoPath_tup[0])
        
        def get_meta_data(filename):
            with open(filename,'rt', errors='ignore') as f:
                meta = {}
                for line in f:
                    if line[:11]=='saf_padding':
                        break
                    a = line[:-1].split(' ')
                    meta[a[0]] = a[1]
            int_values = ['xPixls', 'yPixls', 'NumDPs']
            for i in int_values:
                meta[i] = int(meta[i])
            return meta
        
        def get_data(filename):
            meta = get_meta_data(filename=filename)
            f = open(filename,'rb') 
            f.seek(f.read().find(b'DATA')+6)

            if meta['DaType'] == 'Flt32':

                ty = np.float32
                self.ver_temp = 1

            else:

                ty = np.uint16

                self.ui.calibration_alert.setText("Calibration disabled \n for Digital Level Image")

            return np.fromfile(f, dtype=ty).reshape(-1,meta['yPixls'],meta['xPixls'])
        
        filename = self.videoPath_str 
        
        if filename != '':
        
            meta_data = get_meta_data(filename=filename) 
            data = get_data(filename=filename) 
            
            frame_width = data.shape[2]
            frame_height = data.shape[1]
            frame_count = data.shape[0]

            self.frame_count = frame_count

            self.ui.slider_frame.setMaximum(frame_count-1)
            self.ui.frame_slider.setEnabled(1)
        
            self.data = data
            
            #Median Filter
            
            if self.ui.median_filter.isChecked():
                kernel = 3
                data_median = np.zeros(data.shape)
                
                count = 0
                self.progress = 1
                bar_count = 100/(frame_count)
                
                for i in range(data.shape[0]):
                    data_median[i,:,:] = ndimage.median_filter(data[i,:,:], kernel) 
                    count += 1
                    self.progress += bar_count
                    self.ui.progress_filter.setValue(self.progress)
                self.data = data_median
                
            self.ui.width_text.setText(str(frame_width))
            self.ui.height_text.setText(str(frame_height))
            self.ui.frame_text.setText(str(frame_count))
            self.ui.verification_video.setPixmap(self.true)   
                
    #####################################################################################
    #-------------------------PLOT DEMO FRAME--------------------------------
    #####################################################################################
            self.ui.mpl.canvas.ax.clear()

            if self.video_count == 0:
                div = make_axes_locatable(self.ui.mpl.canvas.ax)
                self.ui.mpl.canvas.cax = div.append_axes('right','4%','2%')
                self.video_count = 1
            else:
                #Clear all axes
                self.ui.mpl.canvas.cax.clear()
                self.ui.tabWidget.setTabEnabled(1,False)
                
                #Ri-inizializzo Croci Rosse
                self.ui.verification_folder.setPixmap(self.false)
                self.ui.verification_param.setPixmap(self.false)
                self.ui.verification_lockin.setPixmap(self.false)
                self.ui.verification_save.setPixmap(self.false)
                self.ui.verification_position.setPixmap(self.false)
                self.ui.verification_ROI.setPixmap(self.empty)
                
                
            img = self.ui.mpl.canvas.ax.imshow(self.data[0,:,:],cmap='magma')
            cb = self.ui.mpl.canvas.fig.colorbar(img,cax = self.ui.mpl.canvas.cax)
            
            if self.ver_temp == 0:
                cb.set_label('[DL]')
            elif self.ver_temp == 1:
                cb.set_label('[°C]')

            self.ui.mpl.canvas.ax.set_title('Demo Frame')
            self.ui.mpl.canvas.ax.set_xlabel('x [pixel]')
            self.ui.mpl.canvas.ax.set_ylabel('y [pixel]')
            self.ui.mpl.canvas.draw()

            #######################################

            self.ui.param_box.setEnabled(1)
            self.ui.ROI_box.setEnabled(1)
            # self.ui.open_button.setEnabled(0)
            
            self.ui.info_video.setText(self.videoPath_str)

#####################################################################################
#-------------------------CHANGE FRAME--------------------------------
#####################################################################################

    def changeFrame(self):

            value = self.ui.slider_frame.value()

            self.ui.mpl.canvas.ax.clear()
           
            img = self.ui.mpl.canvas.ax.imshow(self.data[value,:,:],cmap='magma')
           
            if self.ver_crop == 0:
                self.ui.mpl.canvas.ax.set_title('Demo Frame')

            elif self.ver_crop == 1:
                self.ui.mpl.canvas.ax.set_title('Demo Frame - Cropped')


            self.ui.mpl.canvas.ax.set_xlabel('x [pixel]')
            self.ui.mpl.canvas.ax.set_ylabel('y [pixel]')
            self.ui.mpl.canvas.draw()

#####################################################################################
#-------------------------SET PARAMETERS--------------------------------
#####################################################################################
    def enableFreq(self):

        if self.ui.sampling_freq.text() == '':

            self.ui.auto_freq.setEnabled(0)
            self.ui.load_freq.setEnabled(0)
            self.ui.set_button.setEnabled(0)

        else:
            self.ui.auto_freq.setEnabled(1)
            self.ui.load_freq.setEnabled(1)
            self.ui.temporal_box.setEnabled(1)
        
        if self.ui.load_freq.text() == '':
            self.ui.set_button.setEnabled(0)
            
        if self.ui.sampling_freq.text() != '' and self.ui.load_freq.text() != '':
            self.ui.set_button.setEnabled(1)

        self.ui.verification_param.setPixmap(self.false)

        
    def setParam(self):

            self.ui.verification_lockin.setPixmap(self.false)
            self.samplingFreq = int(self.ui.sampling_freq.text())
            self.loadFreq = float(self.ui.load_freq.text())
            self.ui.verification_param.setPixmap(self.true)
        
            self.ui.lockin_button.setEnabled(1)
            self.ui.verification_save.setPixmap(self.false)
            
    def clearParam(self):
        self.ui.verification_param.setPixmap(self.false)
        self.ui.verification_save.setPixmap(self.false)

#####################################################################################
#-------------------------FREQUENCY AUTODETECT--------------------------------
#####################################################################################

    def autoDetect(self):

        self.ui.stacked_import.setCurrentIndex(1)
        self.ui.num_sample.setMaximum(self.frame_count-1)
        self.ui.start_frame.setMaximum(self.frame_count)
        
        self.ui.auto_freq.setStyleSheet("background-color: rgba(255, 170, 0, 150)")

        self.ui.mpl_5.canvas.ax.clear()
        img = self.ui.mpl_5.canvas.ax.imshow(self.data[self.ui.slider_frame.value(),:,:],cmap='magma')
       
        self.ui.mpl_5.canvas.ax.set_title('Frequency detection ROI')
        self.ui.mpl_5.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_5.canvas.ax.set_ylabel('y [pixel]')
        self.ui.mpl_5.canvas.draw()

        toggle_selector.RS = RectangleSelector(self.ui.mpl_5.canvas.ax, line_select_callback,
                                       drawtype='box', useblit=False,
                                       button=[1, 3], 
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

    def returnImport(self):

        self.ui.stacked_import.setCurrentIndex(0)
        self.ui.auto_freq.setStyleSheet('')

    def searchFrequency(self):
        
        num_sample = self.ui.num_sample.value()
        
        ax_min = self.ui.slider_freq.value()
        ax_max = self.ui.slider_freq.value() + num_sample 

        self.ui.frame_start.display(ax_min)
        self.ui.frame_end.display(ax_max)
        
        if click != [None,None] or release != [None,None]:
            
            data = self.data
            
            x1 = int(click[0])
            y1 = int(click[1])
            x2 = int(release[0])
            y2 = int(release[1])

            self.data_freq = data[:,y1:y2,x1:x2]

            self.mean_value = np.zeros((self.frame_count,2))

            dt = 1/int(self.ui.sampling_freq.text())

            for i in range(0,self.frame_count):
                
                self.mean_value[i,0] = i*dt
                self.mean_value[i,1] = np.mean(self.data_freq[i,:,:])

            n = ax_max-ax_min

            FFT = np.fft.rfft(self.mean_value[ax_min:ax_max,1], n=n) * 2/n
            freq = np.fft.rfftfreq(n, dt)

            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx

            bound_low = find_nearest(freq,self.ui.min_freq.value())
            bound_up = find_nearest(freq,self.ui.max_freq.value())

            mask = np.abs(FFT)[bound_low:bound_up]
            real_freq = freq[np.argmax(mask)+bound_low]

            self.ui.load_freq.setText(str(np.round(real_freq,2)))
            self.ui.info_load_2.setText(str(np.round(real_freq,2)))

            self.ui.mpl_6.canvas.ax.clear()

            if self.plot_count == 0:
                div = make_axes_locatable(self.ui.mpl_6.canvas.ax)
                self.ui.mpl_6.canvas.cax = div.append_axes('bottom','100%','25%')
            else:
                self.ui.mpl_6.canvas.cax.clear()
            
            t1 = self.ui.mpl_6.canvas.ax.plot(self.mean_value[ax_min:ax_max,0],self.mean_value[ax_min:ax_max,1],color='C1')
            t2 = self.ui.mpl_6.canvas.cax.plot(freq[bound_low:bound_up],mask,color='C2')

            self.ui.mpl_6.canvas.ax.set_title('Time Plot - FFT')
            self.ui.mpl_6.canvas.ax.set_xlabel('Time [s]')
            self.ui.mpl_6.canvas.cax.set_xlabel('Frequency [Hz]')
            if self.ver_temp == 0:
                self.ui.mpl_6.canvas.ax.set_ylabel('[DL]')
                self.ui.mpl_6.canvas.cax.set_ylabel('[DL]')
            elif self.ver_temp == 1:
                self.ui.mpl_6.canvas.ax.set_ylabel('[°C]')
                self.ui.mpl_6.canvas.cax.set_ylabel('[°C]')

        
            self.ui.mpl_6.canvas.ax.grid()
            self.ui.mpl_6.canvas.cax.grid()
            self.ui.mpl_6.canvas.draw()

            self.plot_count += 1

        else:
            msg = QMessageBox()
            msg.setWindowTitle('Error!')
            msg.setText('Select a ROI')
        
            x = msg.exec()


    def rescaleFreq(self):

        self.ui.slider_freq.setMaximum(self.frame_count-self.ui.num_sample.value())
     
#####################################################################################
#----------------------------------CROP-----------------------------------------
#####################################################################################

    def ROI(self,event):
        
        self.ui.crop_box.setEnabled(1)
        self.ui.crop_button.setEnabled(1)
        
        toggle_selector.RS = RectangleSelector(self.ui.mpl.canvas.ax, line_select_callback,
                                       drawtype='box', useblit=False,
                                       button=[1, 3], 
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
        

    def cropEnabled(self):
        self.ui.enable_ROI.setStyleSheet("background-color: rgba(255, 170, 0, 150)")
        self.ui.stacked_import.setCurrentIndex(0)
    
    def cropDisabled(self):
        self.ui.enable_ROI.setStyleSheet('')

    def crop(self):
        
        if click != [None,None] or release != [None,None]:
            
            data = self.data
            
            x1 = int(click[0])
            y1 = int(click[1])
            x2 = int(release[0])
            y2 = int(release[1])
            
            self.ui.x1.setText(str(x1))
            self.ui.x2.setText(str(x2))
            self.ui.y1.setText(str(y1))
            self.ui.y2.setText(str(y2))
        
            self.data_crop = data[:,y1:y2,x1:x2]
            
            self.ui.mpl.canvas.ax.clear()
            self.ui.mpl.canvas.cax.clear()
            img = self.ui.mpl.canvas.ax.imshow(self.data_crop[0,:,:],cmap='magma')
            cb = self.ui.mpl.canvas.fig.colorbar(img,cax = self.ui.mpl.canvas.cax) 
            
            if self.ver_temp == 0:
                cb.set_label('[DL]')
            elif self.ver_temp == 1:
                cb.set_label('[°C]')
        
            self.ui.mpl.canvas.ax.set_title('Demo Frame - Cropped')
            self.ui.mpl.canvas.ax.set_xlabel('x [pixel]')
            self.ui.mpl.canvas.ax.set_ylabel('y [pixel]')
            self.ui.mpl.canvas.draw()
            
            self.ui.crop_button.setEnabled(0)
            self.ui.resume.setEnabled(1)
            self.ui.accept.setEnabled(1)
            
            self.ui.enable_ROI.setStyleSheet('')
            
        else:
            msg = QMessageBox()
            msg.setWindowTitle('Error!')
            msg.setText('Select a ROI on the right axes')
        
            x = msg.exec()
        
    def resume(self):
        
        self.ui.mpl.canvas.ax.clear()
        self.ui.mpl.canvas.cax.clear()
        img = self.ui.mpl.canvas.ax.imshow(self.data[0,:,:],cmap='magma')
        cb = self.ui.mpl.canvas.fig.colorbar(img,cax = self.ui.mpl.canvas.cax)
        
        if self.ver_temp == 0:
            cb.set_label('[DL]')
        elif self.ver_temp == 1:
            cb.set_label('[°C]')
                
        self.ui.mpl.canvas.ax.set_title('First Frame')
        self.ui.mpl.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl.canvas.ax.set_ylabel('y [pixel]')
        self.ui.mpl.canvas.draw()
        
        click = [None,None]
        release = [None,None]
        
        self.ui.x1.setText('')
        self.ui.x2.setText('')
        self.ui.y1.setText('')
        self.ui.y2.setText('')
        
        self.ui.crop_button.setEnabled(0)
        self.ui.accept.setEnabled(0)
    
    def accept(self):
    
        if click != [None,None] or release != [None,None]:
            self.data = self.data_crop
            self.ui.verification_ROI.setPixmap(self.true)
            self.ver_crop = 1
            
            self.ui.resume.setEnabled(0)
        else:
            return

#####################################################################################
#----------------------------------LOCK-IN--------------------------------------
#####################################################################################

    def lockIn(self):

        self.ui.verification_lockin.setPixmap(self.false)
                
        data = self.data

        t = np.linspace(0,data.shape[0]/self.samplingFreq,data.shape[0]) 
        sine = np.sin(self.loadFreq*t*2*np.pi) 
        cosine = np.cos(self.loadFreq*t*2*np.pi) 
        S = (np.ones((data.shape[0],data.shape[1],data.shape[2])).T*sine).T 
        C = (np.ones((data.shape[0],data.shape[1],data.shape[2])).T*cosine).T 
        L1 = S*data 
        L2 = C*data 
                
        L = np.sqrt(L1**2+ L2**2)
        Re = np.sum(L1,0)/data.shape[0]
        Img = np.sum(L2,0)/data.shape[0] 
                
        self.Mag = 2*np.sqrt(Re**2+Img**2)
        self.Phase = np.arctan(Img/Re)*(180/np.pi) 
                
        self.ui.verification_lockin.setPixmap(self.true)
        self.ui.tabWidget.setTabEnabled(1,True)
         
        if self.lock_count <1:
            div = make_axes_locatable(self.ui.mpl_2.canvas.ax)
            self.ui.mpl_2.canvas.cax = div.append_axes('right','4%','2%')

        self.ui.mpl_2.canvas.ax.clear()
        self.ui.mpl_2.canvas.cax.clear()
        img = self.ui.mpl_2.canvas.ax.imshow(self.Mag,cmap='magma')

        def fmt(x, pos):
            a, b = '{:.2e}'.format(x).split('e')
            b = int(b)
            return r'${}e{{{}}}$'.format(a, b)

        cb = self.ui.mpl_2.canvas.fig.colorbar(img,cax = self.ui.mpl_2.canvas.cax,format=ticker.FuncFormatter(fmt))
        
        if self.ver_cal == 0:
            if self.ver_temp == 0:
                cb.set_label('[DL]')
            elif self.ver_temp == 1:
                cb.set_label('[°C]')
        elif self.ver_cal == 1:
            cb.set_label('[MPa]')
            
        self.ui.mpl_2.canvas.ax.set_title('Magnitude')
        self.ui.mpl_2.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_2.canvas.ax.set_ylabel('y [pixel]')
        self.ui.mpl_2.canvas.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_2.canvas.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_2.canvas.draw()
        
        if self.lock_count <1:
            div = make_axes_locatable(self.ui.mpl_3.canvas.ax)
            self.ui.mpl_3.canvas.cax = div.append_axes('right','4%','2%')
        self.ui.mpl_3.canvas.ax.clear()
        self.ui.mpl_3.canvas.cax.clear()
        img = self.ui.mpl_3.canvas.ax.imshow(self.Phase,cmap='magma')
        cb = self.ui.mpl_2.canvas.fig.colorbar(img,cax = self.ui.mpl_3.canvas.cax)
        cb.set_label('[°]')
        self.ui.mpl_3.canvas.ax.set_title('Phase')
        self.ui.mpl_3.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_3.canvas.ax.set_ylabel('y [pixel]')
        self.ui.mpl_3.canvas.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_3.canvas.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_3.canvas.draw()

        self.lock_count +=1

        self.ui.info_sample.setText(str(self.samplingFreq))
        self.ui.info_load.setText(str(self.loadFreq))
                
        self.ui.filtering_box.setEnabled(1)

        if self.ver_temp == 1:
            self.ui.calibration_box.setEnabled(1)
        
        if self.ver_path == 1:
            self.ui.save_box.setEnabled(1) 

        if self.ver_cal == 1:

            self.Mag = np.float32(self.Mag)/self.km

            self.ui.verification_calibration.setPixmap(self.true) 
            self.ui.filter_2.setText('Calibration: ON')
            self.ui.filter.setText('Filter: OFF')
            self.ui.colormap.setCurrentIndex(0)
            self.ui.calibration_box.setEnabled(0)
            
    def changeLimit(self): 
        
        self.limit = 1
        
        if self.ui.cblim_down.text() and self.ui.cblim_up.text() !='':
        
            self.ui.mpl_2.canvas.ax.clear()
            self.ui.mpl_2.canvas.cax.clear()
            img = self.ui.mpl_2.canvas.ax.imshow(self.Mag,cmap='magma')

            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${}e{{{}}}$'.format(a, b)

            cb = self.ui.mpl_2.canvas.fig.colorbar(img,cax = self.ui.mpl_2.canvas.cax,format=ticker.FuncFormatter(fmt))
            
            cb.mappable.set_clim(self.ui.cblim_down.text(),self.ui.cblim_up.text())
            
            if self.ver_cal == 0:
                if self.ver_temp == 0:
                    cb.set_label('[DL]')
                elif self.ver_temp == 1:
                    cb.set_label('[°C]')
            elif self.ver_cal == 1:
                cb.set_label('[MPa]')
                
            self.ui.mpl_2.canvas.ax.set_title('Magnitude')
            self.ui.mpl_2.canvas.ax.set_xlabel('x [pixel]')
            self.ui.mpl_2.canvas.ax.set_ylabel('y [pixel]')
            self.ui.mpl_2.canvas.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
            self.ui.mpl_2.canvas.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
            self.ui.mpl_2.canvas.draw()
        
#####################################################################################
#------------------------------------------------------------------------------------
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#------------------------------------------------------------------------------------
#####################################################################################
#----------------------------------SWITCH GRAPH--------------------------------------
#####################################################################################
    def switchGraph(self):

        value = self.ui.slider_graph.value()

        if value == 0:

            self.ui.stacked_post.setCurrentIndex(1)
            self.ui.temp_plot.setStyleSheet("")

        elif value == 1:

            self.ui.stacked_post.setCurrentIndex(0)
            self.ui.temp_plot.setStyleSheet("")

#####################################################################################
#----------------------------------FILTER--------------------------------------
#####################################################################################

    def filterLock(self):
        
        self.ui.filter.setText('Filter: OFF')
        
        self.blur_Mag = np.float32(self.Mag)  
        self.blur_Phase = np.float32(self.Phase) 
        
        cmap = self.ui.colormap.currentText() 
      
        if self.ui.enable_filter.isChecked():
        
            scale = self.ui.scale_spinbox.value()
            
            kernel_size = self.ui.kernel_combobox.currentIndex()
            
            if kernel_size is 0:
                kernel = (3,3)
            elif kernel_size is 1:
                kernel = (5,5)
            
            img_Mag = cv2.resize(self.Mag, (0,0), fx = scale, fy = scale)
            self.blur_Mag = cv2.GaussianBlur(img_Mag,kernel,0)
            
            img_Phase = cv2.resize(self.Phase, (0,0), fx = scale, fy = scale)
            self.blur_Phase = cv2.GaussianBlur(img_Phase,kernel,0)
            
            self.ui.filter.setText('Filter: ON')
            
            self.moltiplicator = scale

        def fmt(x, pos):
            a, b = '{:.2e}'.format(x).split('e')
            b = int(b)
            return r'${}e{{{}}}$'.format(a, b)

        self.ui.mpl_2.canvas.ax.clear()
        self.ui.mpl_2.canvas.cax.clear()
        img = self.ui.mpl_2.canvas.ax.imshow(self.blur_Mag,cmap=cmap)
        cb = self.ui.mpl_2.canvas.fig.colorbar(img,cax = self.ui.mpl_2.canvas.cax,format=ticker.FuncFormatter(fmt))
        
        if self.limit == 1:
            cb.mappable.set_clim(self.ui.cblim_down.text(),self.ui.cblim_up.text())
    
        if self.ver_cal == 0:
            if self.ver_temp == 0:
                cb.set_label('[DL]')
            elif self.ver_temp == 1:
                cb.set_label('[°C]')
        elif self.ver_cal == 1:
            cb.set_label('[MPa]')
       
        self.ui.mpl_2.canvas.ax.set_title('Magnitude')
        self.ui.mpl_2.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_2.canvas.ax.set_ylabel('y [pixel]')

        self.ui.mpl_2.canvas.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_2.canvas.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        
        self.ui.mpl_2.canvas.draw()
             
        self.ui.mpl_3.canvas.ax.clear()
        self.ui.mpl_3.canvas.cax.clear()
        img = self.ui.mpl_3.canvas.ax.imshow(self.blur_Phase,cmap=cmap)
        cb = self.ui.mpl_3.canvas.fig.colorbar(img,cax = self.ui.mpl_3.canvas.cax)
        cb.set_label('[°]')
        self.ui.mpl_3.canvas.ax.set_title('Phase')
        self.ui.mpl_3.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_3.canvas.ax.set_ylabel('y [pixel]')
        
        self.ui.mpl_3.canvas.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_3.canvas.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
       
        self.ui.mpl_3.canvas.draw()
    
    def enableGaussian(self):
            
            if self.ui.enable_filter.isChecked():
                self.ui.gaussian_box.setEnabled(1)
            else:
                self.ui.gaussian_box.setEnabled(0)
    
    def resumeLock(self):
        
        self.moltiplicator = 1
        
        self.ui.mpl_2.canvas.ax.clear()
        self.ui.mpl_2.canvas.cax.clear()
        img = self.ui.mpl_2.canvas.ax.imshow(self.Mag,cmap='magma')


        def fmt(x, pos):
            a, b = '{:.2e}'.format(x).split('e')
            b = int(b)
            return r'${}e{{{}}}$'.format(a, b)

        cb = self.ui.mpl_2.canvas.fig.colorbar(img,cax = self.ui.mpl_2.canvas.cax,format=ticker.FuncFormatter(fmt))

        if self.ver_cal == 0:
            if self.ver_temp == 0:
                cb.set_label('[DL]')
            elif self.ver_temp == 1:
                cb.set_label('[°C]')
        elif self.ver_cal == 1:
            cb.set_label('[MPa]')

        self.ui.mpl_2.canvas.ax.set_title('Magnitude')
        self.ui.mpl_2.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_2.canvas.ax.set_ylabel('y [pixel]')
        
        self.ui.mpl_2.canvas.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_2.canvas.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
    
        self.ui.mpl_2.canvas.draw()
             
        self.ui.mpl_3.canvas.ax.clear()
        self.ui.mpl_3.canvas.cax.clear()
        img = self.ui.mpl_3.canvas.ax.imshow(self.Phase,cmap='magma')
        cb = self.ui.mpl_3.canvas.fig.colorbar(img,cax = self.ui.mpl_3.canvas.cax)
        cb.set_label('[°]')
        self.ui.mpl_3.canvas.ax.set_title('Phase')
        self.ui.mpl_3.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_3.canvas.ax.set_ylabel('y [pixel]')
        
        self.ui.mpl_3.canvas.ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
        self.ui.mpl_3.canvas.ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/self.moltiplicator), ',')))
                
        self.ui.mpl_3.canvas.draw()
        
        self.ui.filter.setText('Filter: OFF')
    
#####################################################################################
#----------------------------------CALIBRATION--------------------------------------
#####################################################################################
        
    def calibration(self):
        
        ind = self.ui.stacked_calibration.currentIndex()

        self.ui.filter.setText('Filter: OFF')
        self.ui.colormap.setCurrentIndex(0)
        
        if ind !=2:
            
            self.ui.cal_save.setEnabled(1)

            if ind == 0:
                
                mat_ind = self.ui.calibration_material.currentIndex()
                
                if mat_ind == 0:
                    
                    self.km = 8.8*(10**-6)
                    
                elif mat_ind == 1:
                    
                    self.km = 6.2*(10**-5)
                    
                elif mat_ind == 2:
                    
                    self.km = 3.5*(10**-6)
                    
                elif mat_ind == 3:
                    
                    self.km = 3.5*(10**-6)
                
            elif ind == 1:
                
                unit = np.round(self.ui.calibration_unit.value(),2)
                exp = self.ui.calibration_exp.value()
                
                self.km = unit*(10**exp)

            self.ui.km_text.setText('Km = %.2E' % self.km)
            
            self.cal_Mag = np.float32(self.Mag)/self.km
            self.cal_Phase = np.float32(self.Phase)
            
            self.ui.mpl_2.canvas.ax.clear()
            self.ui.mpl_2.canvas.cax.clear()
            img = self.ui.mpl_2.canvas.ax.imshow(self.cal_Mag,cmap='magma')

            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${}e{{{}}}$'.format(a, b)

            cb = self.ui.mpl_2.canvas.fig.colorbar(img,cax = self.ui.mpl_2.canvas.cax,format=ticker.FuncFormatter(fmt))
            
            cb.set_label('[MPa]')
            self.ui.mpl_2.canvas.ax.set_title('Magnitude')
            self.ui.mpl_2.canvas.ax.set_xlabel('x [pixel]')
            self.ui.mpl_2.canvas.ax.set_ylabel('y [pixel]')

            self.ui.mpl_2.canvas.draw()
                
            self.ui.mpl_3.canvas.ax.clear()
            self.ui.mpl_3.canvas.cax.clear()
            img = self.ui.mpl_3.canvas.ax.imshow(self.cal_Phase,cmap='magma')
            cb = self.ui.mpl_3.canvas.fig.colorbar(img,cax = self.ui.mpl_3.canvas.cax)
            cb.set_label('[°]')
            self.ui.mpl_3.canvas.ax.set_title('Phase')
            self.ui.mpl_3.canvas.ax.set_xlabel('x [pixel]')
            self.ui.mpl_3.canvas.ax.set_ylabel('y [pixel]')
            
            self.ui.mpl_3.canvas.draw()

        else:

                self.km = 0

                y = self.ui.calibration_strain_y.text()
                p = self.ui.calibration_strain_p.text()
                s = self.ui.calibration_strain_s.text()

                if y and p and s != '' and self.mean_temp != 0:

                    self.km = (self.mean_temp*(1-float(p)))/(float(y)*float(s))

                    self.ui.cal_save.setEnabled(1)
                    self.ui.km_text.setText('Km = %.2E' % self.km)
            
                    self.cal_Mag = np.float32(self.Mag)/self.km
                    self.cal_Phase = np.float32(self.Phase)
                    
                    self.ui.mpl_2.canvas.ax.clear()
                    self.ui.mpl_2.canvas.cax.clear()
                    img = self.ui.mpl_2.canvas.ax.imshow(self.cal_Mag,cmap='magma')

                    def fmt(x, pos):
                        a, b = '{:.2e}'.format(x).split('e')
                        b = int(b)
                        return r'${}e{{{}}}$'.format(a, b)

                    cb = self.ui.mpl_2.canvas.fig.colorbar(img,cax = self.ui.mpl_2.canvas.cax,format=ticker.FuncFormatter(fmt))
                    cb.set_label('[MPa]')
                    self.ui.mpl_2.canvas.ax.set_title('Magnitude')
                    self.ui.mpl_2.canvas.ax.set_xlabel('x [pixel]')
                    self.ui.mpl_2.canvas.ax.set_ylabel('y [pixel]')

                    self.ui.mpl_2.canvas.draw()
                        
                    self.ui.mpl_3.canvas.ax.clear()
                    self.ui.mpl_3.canvas.cax.clear()
                    img = self.ui.mpl_3.canvas.ax.imshow(self.cal_Phase,cmap='magma')
                    cb = self.ui.mpl_3.canvas.fig.colorbar(img,cax = self.ui.mpl_3.canvas.cax)
                    cb.set_label('[°]')
                    self.ui.mpl_3.canvas.ax.set_title('Phase')
                    self.ui.mpl_3.canvas.ax.set_xlabel('x [pixel]')
                    self.ui.mpl_3.canvas.ax.set_ylabel('y [pixel]')
                    
                    self.ui.mpl_3.canvas.draw()

                else:
                
                    msg = QMessageBox()
                    msg.setWindowTitle('Error!')
                    msg.setText('Fill in all fields!')
        
                    x = msg.exec()
                    
    def applyCalibration(self):
        
        self.Mag = self.cal_Mag 

        self.ver_cal = 1

        self.ui.verification_calibration.setPixmap(self.true) 
        self.ui.filter_2.setText('Calibration: ON')
        self.ui.filter.setText('Filter: OFF')
        self.ui.colormap.setCurrentIndex(0)
        self.ui.calibration_box.setEnabled(0)

    def gaugePosition(self):
        
        self.ui.stacked_post.setCurrentIndex(0)
        self.ui.slider_graph.setValue(0)

        self.ui.stacked_multi.setCurrentIndex(2)
        
        self.ui.mpl_4.canvas.ax.clear()
        img = self.ui.mpl_4.canvas.ax.imshow(self.data[0,:,:],cmap='magma')
       
        self.ui.mpl_4.canvas.ax.set_title('Strain Gauge Position')
        self.ui.mpl_4.canvas.ax.set_xlabel('x [pixel]')
        self.ui.mpl_4.canvas.ax.set_ylabel('y [pixel]')
        self.ui.mpl_4.canvas.draw()

        toggle_selector.RS = RectangleSelector(self.ui.mpl_4.canvas.ax, line_select_callback,
                                       drawtype='box', useblit=False,
                                       button=[1, 3], 
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

        self.ui.enable_ROI.setStyleSheet("background-color: rgba(170, 255, 0, 150)")

    def getGaugePosition(self):
        
        if click != [None,None] or release != [None,None]:
            
            data = self.data
            
            x1 = int(click[0])
            y1 = int(click[1])
            x2 = int(release[0])
            y2 = int(release[1])

            self.data_gauge = data[:,y1:y2,x1:x2]

            self.mean_temp = np.mean(self.data_gauge)

            xm = np.mean([x1,x2])
            ym = np.mean([y1,y2])

            coordinate = ('(%.1f , %.1f)' % (xm,ym))

            self.ui.verification_position.setPixmap(self.true)
            
            self.ui.coordinate_text.setText(coordinate)
            self.ui.temperature_text.setText(str(np.round(self.mean_temp,2)))

            self.ui.gauge_button.setStyleSheet("background-color: rgba(170, 255, 0, 150)")

    def returnPost(self):

        self.ui.stacked_multi.setCurrentIndex(0)
        self.ui.stacked_post.setCurrentIndex(0)
        self.ui.slider_graph.setValue(0)
        self.ui.temp_plot.setStyleSheet("")

#####################################################################################
#----------------------------------TEMPORAL PLOT--------------------------------------
##################################################################################### 

    def goTemporal(self):

            self.ui.stacked_post.setCurrentIndex(2)
            self.ui.temp_plot.setStyleSheet("background-color: rgba(255, 170, 0, 150)")

            if self.ver_temp == 0:
                self.ui.results_box.setTitle("Results [DL]")
            elif self.ver_temp == 1:
                self.ui.results_box.setTitle("Results [°C]")

            self.ui.num_sample_2.setMaximum(self.frame_count-1)
            self.ui.start_frame_2.setMaximum(self.frame_count)

            self.ui.mpl_8.canvas.ax.clear()
            img = self.ui.mpl_8.canvas.ax.imshow(self.data[0,:,:],cmap='magma')

            self.ui.mpl_8.canvas.ax.set_title('Demo Frame')
            self.ui.mpl_8.canvas.ax.set_xlabel('x [pixel]')
            self.ui.mpl_8.canvas.ax.set_ylabel('y [pixel]')
            self.ui.mpl_8.canvas.draw()

    def temporalPlot(self):

        num_sample = self.ui.num_sample_2.value()
        
        ax_min = self.ui.slider_plot.value()
        ax_max = self.ui.slider_plot.value() + num_sample 

        self.ui.frame_start_2.display(ax_min)
        self.ui.frame_end_2.display(ax_max)

        if self.ui.button_point.isChecked():

            if click != [None,None] or release != [None,None]:
            
                data = self.data
                
                x1 = int(click[0])
                y1 = int(click[1])

        elif self.ui.button_line.isChecked():
            pass
        elif self.ui.button_area.isChecked():
            
            if click != [None,None] or release != [None,None]:
            
                data = self.data
                
                x1 = int(click[0])
                y1 = int(click[1])
                x2 = int(release[0])
                y2 = int(release[1])

                self.data_plot = data[:,y1:y2,x1:x2]

                self.mean_plot = np.zeros((self.frame_count,2))

                dt = 1/int(self.ui.sampling_freq.text())

                for i in range(0,self.frame_count):
                    
                    self.mean_plot[i,0] = i*dt
                    self.mean_plot[i,1] = np.mean(self.data_plot[i,:,:])

                plot_mean = np.max(self.mean_plot[ax_min:ax_max,1])
                plot_max = np.min(self.mean_plot[ax_min:ax_max,1])
                plot_min = np.mean(self.mean_plot[ax_min:ax_max,1])

                self.ui.mpl_7.canvas.ax.clear()

                self.ui.mpl_7.canvas.ax.plot(self.mean_plot[ax_min:ax_max,0],self.mean_plot[ax_min:ax_max,1],color='C1')
                self.ui.mpl_7.canvas.ax.hlines(plot_mean,self.mean_plot[ax_min,0],self.mean_plot[ax_max,0],linestyles='dashed',color='C2',label='Max = '+str(np.round(plot_mean,3)))
                self.ui.mpl_7.canvas.ax.hlines(plot_max,self.mean_plot[ax_min,0],self.mean_plot[ax_max,0],linestyles='dashed',color='C3',label='Min = '+str(np.round(plot_max,3)))
                self.ui.mpl_7.canvas.ax.hlines(plot_min,self.mean_plot[ax_min,0],self.mean_plot[ax_max,0],linestyles='dashed',color='C0',label='Mean = '+str(np.round(plot_min,3)))
                
                if self.ui.legend_plot.isChecked():
                    self.ui.mpl_7.canvas.ax.legend()
        
                self.ui.mpl_7.canvas.ax.set_title('Temporal Plot')
                self.ui.mpl_7.canvas.ax.set_xlabel('Time [s]')

                if self.ver_temp == 0:
                    self.ui.mpl_7.canvas.ax.set_ylabel('[DL]')
                elif self.ver_temp == 1:
                    self.ui.mpl_7.canvas.ax.set_ylabel('[°C]')
            
                self.ui.mpl_7.canvas.ax.grid()
                self.ui.mpl_7.canvas.draw()

                self.ui.info_mean.setText(str(np.round(plot_mean,3)))
                self.ui.info_max.setText(str(np.round(plot_max,3)))
                self.ui.info_min.setText(str(np.round(plot_min,3)))

    def enablePoint(self):
        
        pass

    def enableLine(self):
        pass

    def enableArea(self):
        
        toggle_selector.RS = RectangleSelector(self.ui.mpl_8.canvas.ax, line_select_callback,
                                       drawtype='box', useblit=False,
                                       button=[1, 3], 
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

    def rescaleTemp(self):

        self.ui.slider_plot.setMaximum(self.frame_count-self.ui.num_sample_2.value())
            
#####################################################################################
#----------------------------------SAVE--------------------------------------
#####################################################################################  

    def goSave(self):

            self.ui.stacked_post.setCurrentIndex(0)
            self.ui.slider_graph.setValue(0)

            self.ui.stacked_multi.setCurrentIndex(1)

    def addPath(self):
        
        self.pathName = QtWidgets.QFileDialog.getExistingDirectory(self,'add path...')
        self.ui.path_text.setText(self.pathName)
        
        self.ui.name_box.setEnabled(1)
    
    def createFolder(self):

        self.folderName = self.ui.folder_name.text()
         
        if hasattr(self,'folderName'):
            
            self.fullFolderName = os.path.join(self.pathName,self.folderName)
      
            try:
                os.mkdir(self.fullFolderName)
                self.ui.verification_folder.setPixmap(self.true)
                
                self.ui.open_folder.setEnabled(1)
                
                self.ui.createfolder_button.setEnabled(0)
                self.ui.path_button.setEnabled(0)
                
                if self.lock_count > 0:
                    self.ui.save_box.setEnabled(1)  
                
                self.ui.info_path.setText(self.fullFolderName)
                
                self.ver_path = 1
            
            except FileExistsError:
                
                self.ver_path = 1
                
                msg = QMessageBox()
                msg.setWindowTitle('Error!')
                msg.setText('Folder Already Existing')
        
                x = msg.exec()
            
        else:
            return

    def openFolder(self):
        
       os.startfile(self.fullFolderName)
        
            
    def save(self):

            name = self.ui.project_name.text()
    
            if self.ui.plot_format.isChecked() or self.ui.data_format.isChecked():
                  
                if self.ui.plot_format.isChecked():
                
                    self.ui.mpl_2.canvas.fig.savefig(os.path.join(self.fullFolderName,name+'_Mag_'+str(self.loadFreq)+'_'+str(self.samplingFreq)+'.png'),dpi=600)
                    self.ui.mpl_3.canvas.fig.savefig(os.path.join(self.fullFolderName,name+'_Pha_'+str(self.loadFreq)+'_'+str(self.samplingFreq)+'.png'),dpi=600)
                    
                if self.ui.data_format.isChecked():
                    
                    if self.ui.csv.isChecked():
                        np.savetxt(os.path.join(self.fullFolderName,name+'_Mag_'+str(self.loadFreq)+'_'+str(self.samplingFreq)+'.csv'), self.Mag,delimiter=',')
                        np.savetxt(os.path.join(self.fullFolderName,name+'_Pha_'+str(self.loadFreq)+'_'+str(self.samplingFreq)+'.csv'), self.Phase,delimiter=',')
                    
                    if self.ui.npy.isChecked():
                        np.save(os.path.join(self.fullFolderName,name+'_Mag_'+str(self.loadFreq)+'_'+str(self.samplingFreq)), self.Mag)
                        np.save(os.path.join(self.fullFolderName,name+'_Pha_'+str(self.loadFreq)+'_'+str(self.samplingFreq)), self.Phase)
                    
                self.ui.verification_save.setPixmap(self.true)
                
            else:
                msg = QMessageBox()
                msg.setWindowTitle('Error!')
                msg.setText('Choose format to save')
        
                x = msg.exec()
                
    def disableData(self):
        
        if self.ui.data_format.isChecked():
            self.ui.csv.setEnabled(1)
            self.ui.npy.setEnabled(1)
        else:
            self.ui.csv.setEnabled(0)
            self.ui.npy.setEnabled(0)
        
 
if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    window = ShowInterface()
    window.setFixedSize(window.size());
    window.show()
    
    sys.exit(app.exec_())
    
