from __future__ import annotations
from typing import Optional

import numpy as np

import math

import PyQt5
from PyQt5 import QtWidgets,QtGui,QtCore

import pyqtgraph as pg

from pyqtgraph import ROI,PolyLineROI,Point,ArrowItem
from pyqtgraph.graphicsItems.ROI import Handle,_PolyLineSegment

def makeArrowPolygon(headLen=20, headWidth=None, tipAngle=20, tailLen=20, tailWidth=3, baseAngle=0):
    """
    Construct a list of points outlining an arrow with the given dimensions.
    The arrow points in the -x direction with tip positioned at 0,0.
    If *headWidth* is supplied, it overrides *tipAngle* (in degrees).
    If *tailLen* is None, no tail will be drawn.
    """
    if headWidth is None:
        headWidth = headLen * math.tan(math.radians(tipAngle * 0.5))
            
    vertices = [(0,0)]
    vertices.append((headLen,-headWidth))
    if tailLen is None:
        innerY = headLen - headWidth * math.tan(math.radians(baseAngle))
        vertices.append((innerY,0))
    else:
        tailWidth *= 0.5
        innerY = headLen - (headWidth-tailWidth) * math.tan(math.radians(baseAngle))
        
        vertices.append((innerY, -tailWidth))
        vertices.append((headLen + tailLen, -tailWidth))
        vertices.append((headLen + tailLen, tailWidth))
        vertices.append((innerY, tailWidth))

    vertices.append((headLen,headWidth))
    
    poly = QtGui.QPolygonF()
    for p in vertices:
        poly.append(QtCore.QPointF(*p))
    return poly

class ArrowROI(ROI):
    def __init__(self,
                 pos, 
                 size,
                 angle=0,
                 **kwargs):
        

        super().__init__(pos, [1,1], angle,
                         resizable=False,
                         removable=False,
                         aspectLocked=True,
                         **kwargs)
        
        self.addTranslateHandle([0,0])
        self.makeArrow(size)
        self.addRotateHandle(self.poly[0],[0,0])
    
    def makeArrow(self,size):
        try:
            headLen = size[0]
            headWidth = size[1]
        except:
            headLen = size
            headWidth = headLen * math.tan(math.radians(20 * 0.5))
        self.poly = makeArrowPolygon(headLen,headWidth,tailLen=None,baseAngle=20)
        dx = self.poly[2].x()
        dy = self.poly[2].y()
        for p in self.poly:
            p.setX(p.x()-dx)
            p.setY(p.y()-dy)
        
    def updateHandles(self):
        rotate_handle:Handle = self.handles[1]['item']
        rotate_handle.setPos(self.poly[0])
        
    def boundingRect(self):
        return self.poly.boundingRect()
    
    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.drawPolygon(self.poly)
        
    def shape(self):
        self.path = QtGui.QPainterPath()
        self.path.addPolygon(self.poly)
        return self.path

    def getArrayRegion(self, *args, **kwds):
        return self._getArrayRegionForArbitraryShape(*args, **kwds)

class AgentHandler(ArrowROI):
    def __init__(self, pose:np.ndarray|list, size, **args):
        super().__init__(pose[0:2],
                         size,
                         np.rad2deg(pose[-1])-180,
                         **args)
    
    @property
    def pose(self):
        state = self.getState()
        pos = state['pos']
        angle = state['angle']
        return [pos[0],pos[1],np.deg2rad((angle-180)%360)]



class FormationHandler(ROI):
    def __init__(self, positions:np.ndarray, center:np.ndarray=np.zeros(2), angle=0.,
                 agent_size=(1,0.5), **kwargs):
        self.agents = []
        
        super().__init__(
            center,(1,1),np.rad2deg(angle),
            removable=False, rotatable=True, resizable=False, **kwargs
        )
        
        for p in positions:
            self.addAgent(p,agent_size)
        
        self.handleSize = 10
        self.addTranslateHandle(center)
        
        rotateHandlePos = (max(agent_size) * 1.5+center[0],center[1])
        self.addRotateHandle(rotateHandlePos, center)
        
    def roiChangedEvent(self):
        self.sigRegionChanged.emit(self)
                    
    def roiChangeStartedEvent(self):
        self.sigRegionChangeStarted.emit(self)
        
    def roiChangeFinishedEvent(self):
        self.sigRegionChangeFinished.emit(self)

    def addAgent(self,pose,size):
        pose[0] = pose[0]*self.size()[0] + self.pos().x()
        pose[1] = pose[1]*self.size()[1] + self.pos().y()
        pose[-1] += self.angle()
        handler = AgentHandler(pose,size,pen=self.pen,parent=self)
        handler.sigRegionChanged.connect(self.roiChangedEvent)
        handler.sigRegionChangeFinished.connect(self.roiChangeStartedEvent)
        handler.sigRegionChangeStarted.connect(self.roiChangeFinishedEvent)
        self.agents.append(handler)
        
    def setArrowSizes(self,size):
        for ag in self.agents:
            ag.makeArrow(size)
            ag.updateHandles()
            
        rot_handle:Handle = self.handles[1]['item']
        pos_handle:Handle = self.handles[0]['item']
        cx = pos_handle.pos().x()
        cy = pos_handle.pos().y()
        dx = rot_handle.pos().x() - cx
        dy = rot_handle.pos().y() - cy
        
        dist = np.sqrt(dx**2+dy**2)
        ndist = max(size) * 1.5
        dx *= ndist/dist
        dy *= ndist/dist
        rot_handle.setPos(cx+dx,cy+dy)
        
        
    def getState(self):
        state = ROI.getState(self)
        state['poses'] = [ag.pose for ag in self.agents]
        return state

    def paint(self, p, opt, widget):
        pass
           
    def shape(self):
        p = QtGui.QPainterPath()
        state = self.getState()
        poses = state['poses']
        if len(poses) == 0:
            return p
        
        p.moveTo(*poses[0][0:2])
        for i in range(len(poses)):
            p.lineTo(*poses[i][0:2])
        p.lineTo(*poses[0][0:2])
        return p
        
class FormationDisplayParam(QtWidgets.QWidget):
    
    paramChanged = QtCore.pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        ##### Building widgets
        
        self._centerx = pg.SpinBox(self,0.,
            bounds=(None,None),suffix='[L]',step=0.1,minStep=0.01,finite=True)
        self._centery = pg.SpinBox(self,0.,
            bounds=(None,None),suffix='[L]',step=0.1,minStep=0.01,finite=True)
        self._centerz = pg.SpinBox(self,0.,
            bounds=(None,None),suffix='[L]',step=0.1,minStep=0.01,finite=True)
        self._angle = pg.SpinBox(self,0.,
            bounds=(0,360),suffix='°',step=0.1,finite=True,wrapping=True)
        
        self._arrow_len = pg.SpinBox(self,1.,
            bounds=(0,None),suffix='[L]',step=0.1,finite=True)
        self._arrow_width = pg.SpinBox(self,0.5,
            bounds=(0,None),suffix='[L]',step=0.1,finite=True)
        
        ##### Interactions
        
        self._centerx.valueChanged.connect(self.updateCenter)
        self._centery.valueChanged.connect(self.updateCenter)
        self._centerz.valueChanged.connect(self.updateCenter)
        self._angle.valueChanged.connect(self.updateAngle)
        self._arrow_len.valueChanged.connect(self.updateArrowSize)
        self._arrow_width.valueChanged.connect(self.updateArrowSize)

        ##### Setting up layout
        
        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)
        
        layout.addRow("Center X:",self._centerx)
        layout.addRow("Center Y:",self._centery)
        layout.addRow("Center Z:",self._centerz)
        layout.addRow("Angle:",self._angle)
        layout.addRow("Arrow length:",self._arrow_len)
        layout.addRow("Arrow width:",self._arrow_width)
        
        ##### Initializing formation
        self.center = np.zeros(3)
        self.updateCenter()
        self.updateAngle()
        self.updateArrowSize()

    def updateCenter(self):
        self.center[0] = self._centerx.value()
        self.center[1] = self._centery.value()
        self.center[2] = self._centerz.value()

        self.paramChanged.emit(self)
        
    def updateAngle(self):
        self.angle = self._angle.value()

        self.paramChanged.emit(self)
        
    def updateArrowSize(self):
        self.arrow_size = (self._arrow_len.value(),self._arrow_width.value())

        self.paramChanged.emit(self)
        
    def setCenter(self,x:float,y:float,z:Optional[float]=None):
        self._centerx.setValue(x)
        self._centery.setValue(y)
        if z is not None:
            self._centerz.setValue(z)
    
    def setAngle(self,angle:float):
        self._angle.setValue(angle)

def test():
    from DubinsFleetPlanner.Formation import circle_formation_from_sep
    
    formation = circle_formation_from_sep(3, 5)
    print("Init formation: ",formation)
    app = pg.mkQApp("Formation Example")

    
    win = QtWidgets.QWidget()
    win.setWindowTitle('Test of formation display')
    layout = QtWidgets.QGridLayout()
    win.setLayout(layout)
    layout.setContentsMargins(0, 0, 0, 0)
    duplicateButton = QtWidgets.QPushButton('Duplicate formation')
    layout.addWidget(duplicateButton, 0, 0, 1 ,2)
    w = pg.GraphicsLayoutWidget()
    layout.addWidget(w, 1, 0, 1, 2)
    win.show()

    
    p = w.addPlot()
    p.setAspectLocked()
    
    form_handler = FormationHandler(formation.positions,formation.center,formation.orientation, pen='r', movable=True)
    
    p.addItem(form_handler)
    
    # p.addItem(agent_item)
        
    def duplicate_arrow():
        poses = form_handler.getState()['poses']
        new_handler = FormationHandler(poses,form_handler.pos(),form_handler.angle(), pen='b', movable=True)
        p.addItem(new_handler)
        
    duplicateButton.clicked.connect(duplicate_arrow)
    pg.exec()