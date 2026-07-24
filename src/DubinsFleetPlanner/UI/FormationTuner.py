import numpy as np

import math

import PyQt5
from PyQt5 import QtWidgets,QtGui,QtCore

from DubinsFleetPlanner.Formation import Formation
from DubinsFleetPlanner.Formation import (
    circle_formation_from_sep,
    line_formation,
    column_formation,
    echelon_right_formation,
    echelon_left_formation,
    v_formation,
    chevron_formation,
    rectangle_formation,
    staggered_formation,
    stacked_chevrons_formation,
    stacked_echelon_formation,
)

formation_funs = {
    "Circle":           lambda nlin,ncol,sep,angle,clockwise: circle_formation_from_sep(nlin*ncol,sep,clockwise=clockwise),
    "Line":             lambda nlin,ncol,sep,angle,clockwise: line_formation(nlin*ncol,sep),
    "Column":           lambda nlin,ncol,sep,angle,clockwise: column_formation(nlin*ncol,sep),
    "Echelon":          lambda nlin,ncol,sep,angle,clockwise: echelon_right_formation(nlin*ncol,sep,angle) if clockwise else echelon_left_formation(nlin*ncol,sep,angle),
    "V Formation":      lambda nlin,ncol,sep,angle,clockwise: v_formation(nlin*ncol,sep,angle,clockwise),
    "Chevron":          lambda nlin,ncol,sep,angle,clockwise: chevron_formation(nlin*ncol,sep,angle,clockwise),
    "Rectangle":        lambda nlin,ncol,sep,angle,clockwise: rectangle_formation(nlin,ncol,sep,angle),
    "Staggered":        lambda nlin,ncol,sep,angle,clockwise: staggered_formation(nlin,ncol,sep,angle),
    "Stacked Chevrons": lambda nlin,ncol,sep,angle,clockwise: stacked_chevrons_formation(nlin,ncol,sep,2*sep,angle),
    "Stacked Echelon":  lambda nlin,ncol,sep,angle,clockwise: stacked_echelon_formation(nlin,ncol,sep,2*sep,angle,clockwise),
}

import pyqtgraph as pg
from pyqtgraph.parametertree import interact,RunOptions,ParameterTree

class FormationTuner(QtWidgets.QWidget):
    
    formationChanged = QtCore.pyqtSignal(Formation,name="FormationChanged")
    
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Formation Tuner')

        ##### Building widgets

        # self._top_label = QtWidgets.QLabel("Formation Tuner")
        # self._top_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # self._top_label.setFont(QtGui.QFont(None, 16, QtGui.QFont.Weight.Bold))

        self._n_lin_widget = pg.SpinBox(self,3,
            int=True,bounds=(0,None))

        self._n_col_widget = pg.SpinBox(self,3,
            int=True,bounds=(0,None)) 
        
        self._n_value_label = QtWidgets.QLabel("")
        
        self._sep_widget = pg.SpinBox(self,5.,
            bounds=(0,None),suffix='[L]',step=0.1,dec=True,minStep=0.01,finite=True)

        self._angle_widget = pg.SpinBox(self,0.,
            bounds=(0,360),suffix='°',step=0.1,
            finite=True,wrapping=True) 

        self._clockwise_widget = QtWidgets.QCheckBox(self)
        self._clockwise_widget.setChecked(True)
        self._clockwise_widget.setTristate(False)

        self._formation_widget = QtWidgets.QComboBox(self)
        self._formation_widget.setDuplicatesEnabled(False)
        self._formation_widget.setEditable(False)
        for k,v in formation_funs.items():
            self._formation_widget.addItem(k,v)
        
        ##### Interactions
        
        self._n_lin_widget.valueChanged.connect(self.updateNValue)
        self._n_col_widget.valueChanged.connect(self.updateNValue)
        
        self._n_lin_widget.valueChanged.connect(self.updateFormation)
        self._n_col_widget.valueChanged.connect(self.updateFormation)
        self._sep_widget.valueChanged.connect(self.updateFormation)
        self._angle_widget.valueChanged.connect(self.updateFormation)
        self._clockwise_widget.stateChanged.connect(self.updateFormation)
        self._formation_widget.currentIndexChanged.connect(self.updateFormation)
        
        ##### Setting up layout
        
        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)
        
        layout.addRow("Number of lines:",self._n_lin_widget)
        layout.addRow("Number of columns:",self._n_col_widget)
        layout.addRow("Total number of agents:",self._n_value_label)
        layout.addRow("Separation:",self._sep_widget)
        layout.addRow("Angle:",self._angle_widget)
        layout.addRow("Clockwise:",self._clockwise_widget)
        layout.addRow("Formation:",self._formation_widget)
        
        ##### Initializing formation
        self.updateNValue()
        self.updateFormation()
        
    def updateFormation(self):
        formation_fun = self._formation_widget.currentData()
        n_lin = self._n_lin_widget.value()
        n_col = self._n_col_widget.value()
        sep = self._sep_widget.value()
        angle = self._angle_widget.value()
        clockwise = self._clockwise_widget.isChecked()
        self.formation = formation_fun(n_lin,n_col,sep,angle,clockwise)
        self.formationChanged.emit(self.formation)

    def updateNValue(self):
        n_lin = self._n_lin_widget.value()
        n_col = self._n_col_widget.value()
        n = n_lin * n_col
        self._n_value_label.setText(str(n))

def test():
    app = pg.mkQApp("Formation Example")
    win = FormationTuner()
    win.show()
    app.exec()
    
def test2():
    from DubinsFleetPlanner.UI.FormationDisplay import FormationHandler,FormationDisplayParam
    app = pg.mkQApp("Formation Example with display")
    win = QtWidgets.QWidget()
    layout = QtWidgets.QGridLayout()
    layout.setColumnStretch(1,2)
    win.setLayout(layout)
    
    editor = FormationTuner(win)
    layout.addWidget(editor,0,0)
    
    display_params = FormationDisplayParam(win)
    layout.addWidget(display_params,1,0)
    
    form_handler = FormationHandler(editor.formation.positions,display_params.center,display_params.angle,display_params.arrow_size, pen='r', movable=True)
        
    w = pg.PlotWidget(win)
    w.setMinimumSize(400,400)
    layout.addWidget(w, 0, 1, 2, 2)

    w.setAspectLocked(True,1)
    w.addItem(form_handler)
    
    def formation_changed(formation):
        nonlocal form_handler
        print("Formation changed: ",formation)
        w.removeItem(form_handler)
        form_handler = FormationHandler(formation.positions,display_params.center,display_params.angle,display_params.arrow_size, pen='r', movable=True)
        w.addItem(form_handler)
        
    def param_changed():
        nonlocal form_handler
        form_handler.setPos(display_params.center[0],display_params.center[1])
        form_handler.setAngle(display_params.angle)
        form_handler.setArrowSizes(display_params.arrow_size)
        
        
    def plot_changed():
        display_params.setCenter(*form_handler.pos())
        display_params.setAngle(form_handler.angle())
        
    editor.formationChanged.connect(formation_changed)
    display_params.paramChanged.connect(param_changed)
    form_handler.sigRegionChangeFinished.connect(plot_changed)
    
    win.show()
    app.exec()