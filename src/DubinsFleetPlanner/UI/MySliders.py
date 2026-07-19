# Copyright (C) 2025 Mael FEURGARD <mael.feurgard@enac.fr>
# 
# This file is part of DubinsFleetPlanner.
# 
# DubinsFleetPlanner is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# DubinsFleetPlanner is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with DubinsFleetPlanner.  If not, see <https://www.gnu.org/licenses/>.

from typing import Literal,Any

import matplotlib.pyplot as plt
import matplotlib.widgets as wd

from matplotlib.axes import Axes

from .UI.util import ColorType

class BinarySlider(wd.Slider):
    """ A Matplotlib Slider specification to implement a 0-1 toggle """
    
    class Formatter01:
        """ Enrich a %-format string by instead providing a class with the modulus operator, suited for True/False display """
        
        def __init__(self,false_txt:str,true_txt:str,cutoff:float=0.5) -> None:
            self.false_txt  = false_txt
            self.true_txt   = true_txt 
            self.cutoff     = cutoff
            
        def __mod__(self,other:float) -> str:
            return self.true_txt if (other >= self.cutoff) else self.false_txt
            
    
    def __init__(self, ax: Axes, label: str, *, 
                 valinit: bool = False, false_txt:str = "False", true_txt:str = "True", 
                 slidermin: wd.Slider | None = None, slidermax: wd.Slider | None = None, 
                 dragging: bool = True, orientation: Literal['horizontal'] | Literal['vertical'] = 'horizontal', 
                 initcolor: ColorType = 'r', track_color: ColorType = 'lightgrey', 
                 handle_style: dict[str, Any] | None = None, **kwargs) -> None:
        """ 
        A specialized implementation Matplotlib's Slider. For all arguments not explicitely presented, 
        please refer to Matplotlib documentation: https://matplotlib.org/stable/api/widgets_api.html#matplotlib.widgets.Slider

        This implemenation works by setting a Slider on the range 0-1 but allowing only steps of size 1 (with endpoints enabled),
        thus resulting in a binary slider, ie a toggle.

        Args:
            valinit (bool, optional): Initial boolean value. Defaults to False.
            false_txt (str, optional): Text to display when set to False. Defaults to "False".
            true_txt (str, optional): Text to display when set to True. Defaults to "True".
        """
        
        valmin = 0.
        valmax = 1.
        
        if (valinit):
            _valinit = 1.
        else:
            _valinit = 0.
        
        super().__init__(ax, label, valmin, valmax, 
                         valinit=_valinit, 
                         valfmt=self.Formatter01(false_txt,true_txt),  # type: ignore (Trust me, using mod overload instead of str %-format will work)
                         closedmin=True, closedmax=True, 
                         slidermin=slidermin, slidermax=slidermax, dragging=dragging, 
                         valstep=1., orientation=orientation, 
                         initcolor=initcolor, track_color=track_color, 
                         handle_style=handle_style, **kwargs)
        
        
        
class IntSlider(wd.Slider):
    """ A Matplotlib Slider specification to implement an integer slider """
    
    def __init__(self, ax: Axes, label: str, 
                 valmin: int, valmax: int, *, 
                 valinit: int | None = None, 
                 valfmt: str | None = None, 
                 slidermin: wd.Slider | None = None, slidermax: wd.Slider | None = None, 
                 dragging: bool = True, 
                 valstep: int | list[int] = 1, 
                 orientation: Literal['horizontal'] | Literal['vertical'] = 'horizontal', 
                 initcolor: ColorType = 'r', track_color: ColorType = 'lightgrey', 
                 handle_style: dict[str, Any] | None = None, **kwargs) -> None:
        """
        A specialized implementation Matplotlib's Slider. For all arguments not explicitely presented, 
        please refer to Matplotlib documentation: https://matplotlib.org/stable/api/widgets_api.html#matplotlib.widgets.Slider

        This implemenation works by forcing the use of integers on min and max values, as well as for stepping.

        Args:
            valinit (int | None, optional): Initial value. If None, uses valmin. Defaults to None.
            valstep (int | list[int], optional): Integer stepping. 
                If list, specifies the possible values for the slider. 
                Otherwise only the step between allowed values. 
                Defaults to 1.
        """
        
        _valinit = valmin if valinit is None else valinit
        
        
        super().__init__(ax, label, valmin, valmax, 
                         valinit=float(_valinit), valfmt=valfmt, 
                         closedmin=True, closedmax=True, 
                         slidermin=slidermin, slidermax=slidermax, 
                         dragging=dragging, 
                         valstep=valstep, 
                         orientation=orientation, 
                         initcolor=initcolor, track_color=track_color, 
                         handle_style=handle_style, **kwargs)