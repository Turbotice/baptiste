
import numpy as np
from accum import accum
from tools import cart2pol

def radialavg2(data, radial_step,x0,y0):
    """
    Radialavg2 computes the average along the radius of a unit circle inscribed in the square matrix data. The radial average is returned in Zr and the mid-points of the M bins are returned in vector R. 

    :param: 
        * data :  data is the square matrix;
        * radial_step: number of steps;
        * x0, y0 : are the coordinates of the center of the circle.
    :return: 
        * Tics : is the vector field of the mid points;
        * Average : if the radial average vector. 

    Examples
    --------
    >>> import numpy as np
    >>> from accum import accum
    >>> x0 = 0
    >>> y0 = 100
    >>> fitlength = 200
    >>> phase_locale = np.ones((2*fitlength,2*fitlength))*np.exp(1j*np.angle(c[x0,y0]))
    >>> signal_local=np.zeros(phase_locale.shape)
    >>> signal_local[:,:] = np.real(c[x0-fitlength:x0+fitlength, y0-fitlength:y0+fitlength]*phase_locale)
    >>> [r2,zr2] = radialavg2(signal_local, 1, fitlength+1, fitlength+1)

    """


    l = np.int(data.shape[0]/2)
    x = np.arange(0,data.shape[1]) - data.shape[1]/2+(l-x0)
    y = np.arange(0,data.shape[0]) - data.shape[0]/2+(l-y0)
    [X,Y] = np.meshgrid(x,y)
    [R, Theta] = cart2pol(X,Y)
    Zinteger = np.zeros(R.shape)
    Zinteger = R/radial_step
    Zinteger = np.abs(X+1j*Y)/radial_step+1
    Tics = accum(Zinteger.astype(int), np.abs(X+1j*Y), func = np.mean)
    Average = accum(Zinteger.astype(int), data, func = np.mean)
    return Tics[1:], Average[1:]