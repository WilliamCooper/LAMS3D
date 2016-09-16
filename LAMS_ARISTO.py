# -*- coding: utf-8 -*-
'''
# LAMS Multi-Beam Processing: 

# <headingcell level=1>

# All steps from the originally recorded histograms to the Earth-referenced wind measurements

# <rawcell>

# This routine reads a netCDF file as produced by nimbus, with LAMS histograms (with folding) representing the recorded LAMS measurements. The steps are:
#   1. Smooth the histogram to remove the large variation in the histogram values that varies slowly and does not represent wind measurements.
#   2. Identify the best peak in each beam, with reference to the value expected from the measured TAS to address folding and avoid spurious peaks.
#   3. Convert to line-of-sight airspeeds for each beam. Where no peak can be identified (e.g., because of insufficient SNR), record the value as masked
#      (converted to -32767. in the new netcdf file).
#   4. Set up new netCDF variables with these airspeeds, with attributes etc. These will be added to the new netCDF file.
#   5. Following the steps described in the note named 'LAMSprocessing3Dwind.pdf' on the RAF science wiki: 
#      http://wiki.eol.ucar.edu/rafscience/LAMS%20Processing?action=AttachFile&do=view&target=LAMSprocessing3Dwind.pdf, 
#      calculate new relative and Earth-reference wind measurements. Set these for addition to the netCDF file, with appropriate attributes.
#   6. Write a new netCDF file that includes the new variables.
# 
# If the original file is named OldFileName.nc, the new file is OldFileNameWAC.nc . It will be overwritten if present!
# 
# Some sections of the note referenced above are reproduced below to indicate how they relate to the segments of the code that follow.
'''
# <codecell>

# initialization:
import numpy as np    # array manipulations
import numpy.ma as ma # masked arrays -- masks used to track and propagate missing values
import matplotlib as mpl
mpl.use("Qt5Agg", force=True)
from pylab import *   # included to be able to plot easily. Some plots are generated here.
from netCDF4 import Dataset  # used to read and write the netCDF file
import os             # for copying netCDF file before changing it
import time           # just for timing execution
from copy import deepcopy       # not used now...
#from pandas import DataFrame    # not used at this time....
#from scipy import signal        # used for the wavelet smoothing and peak identification
from math import factorial      # used by the Savitzky-Golay routine
#%lsmagic                       # used to check for available magics; %matplotlib isn't among them on my system...
#%matplotlib inline             # doesn't work, need next version of ipython (coming soon)
window_size_SG = -1             # used to calculate Savitzky-Golay coefficients only on first call
order_SG = -1                   # preserve these for re-use; global in S-G
deriv_SG = -1 
BEAM1last = 0. 
BEAM1slope = 0.
BEAM2last = 0. 
BEAM2slope = 0.
BEAM3last = 0. 
BEAM3slope = 0.
BEAM4last = 0. 
BEAM4slope = 0.


# <codecell>

# This routine, obtained as referenced below, is used to fit the background in 
# the LAMS histogram using Savitzky-Golay polynomials.
# See http://nbviewer.ipython.org/github/pv/SciPy-CookBook/blob/master/ipython/SavitzkyGolay.ipynb

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
# The following is, for this application, constant so it is not necessary
# to recalculate all this each time. Do once and then skip on subsequent
# entries (this application only):
    global first_SG_call, half_window, window_size_SG, m_SG, order_SG, deriv_SG    # preserve these for re-use
    if window_size != window_size_SG or order_SG != order or deriv_SG != deriv:
#       from math import factorial
        window_size_SG = window_size
        order_SG = order
        deriv_SG = deriv    
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
    # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m_SG = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
# End of first-call initialization
        
    # pad the signal at the extremes with
    # values taken from the signal itself
#   firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )   # this is standard
#   lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])  # this is standard
#           # change this to reflect that this is a folded histogram
    firstvals = y[1:half_window+1][::-1]
    lastvals  = y[-half_window-1:-1][::-1]
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m_SG[::-1], y, mode='valid')

# <headingcell level=2>
'''
# Constants that define the geometry:

# <markdowncell>

# 
# From Scott Spuler's email of 19 Sept 2013:                        
#                    
#  "Here are the distances (in meters) from LAMS C-MIGITS to the three GV IMU locations.  
#   Coordinates are z-longitudinal (fore is positive), x-horizontal (port is positive), y-vertical (up is positive)
# 
#      DZ    DX     DY
#   1  -10.305 6.319  -1.359                                                                                
#   2  -10.394 7.452  -1.486                                                                                   
#   3  -10.470 6.319  -1.359                                                  
# 
#   Also, from the laser tracker survey, the beams theta and phi angles are
#   A: 34.95 deg, 0.08 deg; 
#   B: 34.96 deg, 119.93 deg; 
#   C: 35.01 deg, -120.08 deg "   
# 
# In the 'LL' statement below, I have changed the signs of the distances. These are obviously distances from the IMUs to the CMIGITS.
# 
# Matt Hayman has new values that should be used when this program is used for CONTRAST data. Also, have not yet added the 4th beam.

# <markdowncell>

# For 4 beams, the least-squares-fit for the wind components (if all errors are taken to be the same) is
# 
# $$\hskip2in \mathrm{relative\ wind\ components} = \mathrm{M}\ b $$
# 
# where $\mathrm{M}$ is a 3x4 matrix given below and $b$ is the column matrix of beam-speed measurements.
# 
# $$\hskip2in \mathrm{M}=\left(\begin{array}{ccc}0.272026 & 0.271492 & 0.272002 & 0.3317481 \\\ -0.001869 & 1.008053 & -1.005966 & -0.000681 \\\ -1.163110 & 0.582780 & 0.581060 & -0.000191 \end{array}\right)$$
# 
# Once 4-beam data are available, will need to adapt the code below (which only uses the three off-axis beams) to use the information from the fourth beam. A comparison of the above matrix to the three-beam version (see Si below) shows that the 4th beam has neglible influence on the lateral components but significant influence on the longitudinal component of the relative wind.

----
From Matt's 12 May 2014 email:
Per Cooper's request, here are the final measured LAMS pointing vectors/angles.

These are the final vectors I get for the beam pointing in my coordinate system:
Beam 1 = [0.8183, 0.0079, 0.5747]
Beam 2 = [0.8207, -0.5001, -0.2764]
Beam 3 = [0.8207, 0.4906, -0.2930]
Beam 4 = [1.0000, 0.0000, 0.0000]
I think the second and third components need to be negated for Cooper's coordinate system defined in the LAMS processing memo.

I calculate the following angles in the Cooper coordinate frame (to more significant figures than are reasonable):

Theta1 = 35.0802
Phi1 = 179.2171

Theta2 = 34.8501
Ph2 = -61.0712

Theta3 = 34.8496
Phi3 = 59.1506

Theta4 = 0.0000
Phi4 = -56.8530



'''
# <codecell>

# The polar and azimuthal angles of the three beams wrt the GV longitudinal axis
#Theta = np.array ([34.95, 34.96, 35.01, 0.]) * np.pi / 180. # values in radians
#Phi = np.array ([180.08, -60.07, 59.92, 0.]) * np.pi / 180. #  "  "
Theta = np.array ([35.0802, 34.8501, 34.8496, 0.]) * np.pi / 180. # revised 12 May 2014
Phi = np.array ([179.2171, -61.0712, 59.1506, 0.]) * np.pi / 180. #  "  "
## SPECIAL ARISTO
Theta[1] = 0

# also need the distances from the IRS to LAMS: (x,y,z)
LL = ma.array ([-10.305, -6.319, 1.359])                # see Scott's email, recorded above.
# unit vectors along beams are then:
#   a[i] = [cos(Theta[i]), -sin(Theta[i])*sin(Phi[i]), sin(Theta[i])*cos(Phi[i])]
# and the dot products with the (i,j,k) unit vectors give the direction cosine matrix:
S = np.array ([[cos(Theta[0]), -sin(Theta[0])*sin(Phi[0]), sin(Theta[0])*cos(Phi[0])], \
               [cos(Theta[1]), -sin(Theta[1])*sin(Phi[1]), sin(Theta[1])*cos(Phi[1])], \
               [cos(Theta[2]), -sin(Theta[2])*sin(Phi[2]), sin(Theta[2])*cos(Phi[2])]])
Si = linalg.inv (S)  # calculate the inverse of S -- this is the 3-beam version
#print Si
S4 = np.vstack ((S, [cos(Theta[3]), -sin(Theta[3])*sin(Phi[3]), sin(Theta[3])*cos(Phi[3])]))
StS =  linalg.inv (ma.dot (S4.T, S4))
M = ma.dot (StS, S4.T)      # matrix for finding relative wind from 4-beam LAMS
#print 'M', shape(M)
#for i in range (0,3):
#    print '%.6f %.6f %.6f %.6f' % (M[i,0], M[i,1], M[i,2], M[i,3])

# <headingcell level=2>

# Acquire the data as initially processed by nimbus

# <codecell>

# copy the netCDF file to another before opening, to be able to write to the new file
#InputFile = 'IDEASGrf04HR'
#InputFile = 'IDEASGrf08HR25'
# InputFile = 'IDEASGrf04SMS'
InputFile = 'ARISTOrf03'
InputFile = 'ARISTOrf04_LAMS'
InputFile = 'AR16rf06hr'
DataDirectory = '/Data/ARISTO/'
if DataDirectory[-1] != '/':
    DataDirectory +=  '/'
#InputFile = raw_input("Original file in "+DataDirectory+" (without .nc): ")
#print "you entered ", InputFile
#OutputFile = raw_input("New file name (careful, will be overwritten if present): ")
#print "you entered ", OutputFile
FullFileName = DataDirectory + InputFile + '.nc'       # used this to be able to compare to Scott's values
RevisedFileName = DataDirectory + InputFile + 'LAMS.nc'   
copyCommand = 'cp ' + FullFileName + ' ' + RevisedFileName
Stime = time.time()
localtime = time.asctime( time.localtime(Stime) )
print "Processing started at ", localtime
os.system(copyCommand)
#!cp /Data/LAMS/IDEASGrf04SMS.nc /Data/LAMS/IDEASGrf04WAC.nc  # shell command
netCDFfile = Dataset (RevisedFileName, 'a')
DLEN = len (netCDFfile.dimensions['Time'])

# variables from netCDF file needed for processing:
varNames = ['TASX', 'THDG', 'PITCH', 'ROLL', 'AKRD', 'SSLIP', 'Time', \
            'CPITCH_LAMS', 'CROLL_LAMS', 'CTHDG_LAMS', 'VNSC', 'VEWC', 'VSPD', \
#            'CPITCH_LAMS', 'CROLL_LAMS', 'CTHDG_LAMS', 'CVNS_LAMS', 'CVEW_LAMS', 'VSPD', \
#            'BEAM1_speed', 'BEAM2_speed', 'BEAM3_speed']  # Scott's values, not used here 
#                                                           but kept for reference if present
            'BEAM1_LAMS', 'BEAM2_LAMS', 'BEAM3_LAMS', 'BEAM4_LAMS', 'TAS_A']   # LAMS histograms for the three beams; 4B add BEAM4_LAMS
# if any are not found, routine will crash when they are used
# see better way to do this in newVnetCDF; eventually convert to that
Vars = []   # set up list used temporarily for transfer to masked arrays
for h in varNames:
    for v in netCDFfile.variables:
        if h == v:
            Vars.append (netCDFfile.variables[v][:])
            
# set up masked arrays with these variables:
TASX =       ma.array (Vars[0], fill_value=-32767.)

# the following segment is to handle high-rate data, which need to be flattened
Shape = netCDFfile.variables['TASX'].shape
DL = 1
SampleRate = 1
if len (Shape) > 1:
    SampleRate = Shape[1]
for i in range (0, len (Shape)):
    DL *= Shape[i]
TASX.shape = (DL)

THDG =       ma.array (Vars[1].reshape (DL), fill_value=-32767.)
PITCH =      ma.array (Vars[2].reshape (DL), fill_value=-32767.)
ROLL =       ma.array (Vars[3].reshape (DL), fill_value=-32767.)
AKRD =       ma.array (Vars[4].reshape (DL), fill_value=-32767.)
SSLIP =      ma.array (Vars[5].reshape (DL), fill_value=-32767.)
Time =       ma.array (Vars[6], fill_value=-32767.)  # Time is not high-rate
CPITCH =     ma.array (Vars[7].reshape (DL), fill_value=-32767.)
CROLL =      ma.array (Vars[8].reshape (DL), fill_value=-32767.)
CTHDG =      ma.array (Vars[9].reshape (DL), fill_value=-32767.)
VNSC =       ma.array (Vars[10].reshape (DL), fill_value=-32767.)
VEWC =       ma.array (Vars[11].reshape (DL), fill_value=-32767.)
VSPD =       ma.array (Vars[12].reshape (DL), fill_value=-32767.)
## working with 50-Hz variables; 25 Hz does not work:
## start of SPECIAL section 1:
BEAM1hist =  ma.array (Vars[13].reshape (DL*2,512), fill_value=-32767.)
BEAM2hist =  ma.array (Vars[14].reshape (DL*2,512), fill_value=-32767.)
BEAM3hist =  ma.array (Vars[15].reshape (DL*2,512), fill_value=-32767.)
BEAM4hist =  ma.array (Vars[16].reshape (DL*2,512), fill_value=-32767.)
## end of SPECIAL section 1
# BEAM1hist =  ma.array (Vars[13].reshape (DL,512), fill_value=-32767.)
# BEAM2hist =  ma.array (Vars[14].reshape (DL,512), fill_value=-32767.)
# BEAM3hist =  ma.array (Vars[15].reshape (DL,512), fill_value=-32767.)
# BEAM4hist =  ma.array (Vars[16].reshape (DL,512), fill_value=-32767.)
#4B BEAM4hist = ma.array (Vars[16].reshape (DL,512), fill_value=-32767.)
TAS_A =      ma.array (Vars[17].reshape (DL), fill_value=-32767.)  #4B: change to [17]
CTHDG = ma.masked_where (CTHDG == -32767., CTHDG)

## start of SPECIAL section 2:
# average to 25 Hz
for ix in range (0, DL):        # don't worry about endpoints
    BEAM1hist[ix,:] = (BEAM1hist[2*ix,:] + BEAM1hist[2*ix+1,:])/2.
    BEAM2hist[ix,:] = (BEAM2hist[2*ix,:] + BEAM2hist[2*ix+1,:])/2.
    BEAM3hist[ix,:] = (BEAM3hist[2*ix,:] + BEAM3hist[2*ix+1,:])/2.
    BEAM4hist[ix,:] = (BEAM4hist[2*ix,:] + BEAM4hist[2*ix+1,:])/2.
BEAM1hist = BEAM1hist[0:DL,:]
BEAM2hist = BEAM2hist[0:DL,:]
BEAM3hist = BEAM3hist[0:DL,:]
## end of SPECIAL section 2

# also reserve arrays that will hold line-of-sight speeds:
BEAM1speed = ma.empty (DL)
BEAM2speed = ma.empty (DL)
BEAM3speed = ma.empty (DL)
BEAM4speed = ma.empty (DL)

# special masked regions in IDEASG rf04 25Hz, where CTHDG oscillates while going through 180 deg.
#CTHDG[4340+125000:4390+125000] = ma.masked
#CTHDG[9075+125000:9125+125000] = ma.masked
#CTHDG[19155+125000:19205+125000] = ma.masked
#CTHDG[24080+125000:24130+125000] = ma.masked
print 'netCDF variables loaded; Elapsed time: ', time.time () - Stime, ' s'

# <headingcell level=2>

# Now find line-of-sight speed from histograms, with consideration of folding based on TAS_A

# <codecell>

# There is probably a better way to do this, vectorized, but this doesn't take much time
# (Using previous index to extend instead of next index in case there are sequences...)
hma1 = ma.getmaskarray (BEAM1hist)
hma2 = ma.getmaskarray (BEAM2hist)
hma3 = ma.getmaskarray (BEAM3hist)
hma4 = ma.getmaskarray (BEAM4hist)
for ix in xrange (1,DL):
    if BEAM1hist[ix,0] < 0. or hma1[ix,0]:
        BEAM1hist[ix,:] = BEAM1hist[ix-1,:]
    if BEAM2hist[ix,0] < 0. or hma2[ix,0]:
        BEAM2hist[ix,:] = BEAM2hist[ix-1,:]
    if BEAM3hist[ix,0] < 0. or hma3[ix,0]:
        BEAM3hist[ix,:] = BEAM3hist[ix-1,:]
    if BEAM4hist[ix,0] < 0. or hma4[ix,0]:
        BEAM4hist[ix,:] = BEAM4hist[ix-1,:]
#       print 'substituted previous element for index ', ix
#print len(BEAM1hist[BEAM1hist[:,0] < 0.])
#BEAM1hist =  ma.array (Vars[13].reshape (DL,512), fill_value=-32767.)
lamsB1 = ma.array(BEAM1hist, fill_value=-32767.)
lamsB2 = ma.array(BEAM2hist, fill_value=-32767.)
lamsB3 = ma.array(BEAM3hist, fill_value=-32767.)
lamsB4 = ma.array(BEAM4hist, fill_value=-32767.)
print 'loaded lams histograms after mods; elapsed time: ', time.time () - Stime, ' s'

# <rawcell>

# optional: 
# average data in moving average
Average = False
if Average:
    for ix in range (1, DL-1):        # don't worry about endpoints
        lamsB1[ix] = (BEAM1hist[ix-1,:] + 2.*BEAM1hist[ix,:] + BEAM1hist[ix+1,:])/4.
        lamsB2[ix] = (BEAM2hist[ix-1,:] + 2.*BEAM2hist[ix,:] + BEAM2hist[ix+1,:])/4.
        lamsB3[ix] = (BEAM3hist[ix-1,:] + 2.*BEAM3hist[ix,:] + BEAM3hist[ix+1,:])/4.
        lamsB4[ix] = (BEAM4hist[ix-1,:] + 2.*BEAM4hist[ix,:] + BEAM4hist[ix+1,:])/4.


# have the histograms in the variables lamsBx, analogous to SMS lams2

# <rawcell>

# clf()
# plot (lamsB1[8100][200:250], color = 'black')
# plot (lamsB1[8110][200:250], color = 'blue')
# plot (lamsB1[8120][200:250], color = 'green')
# plot (lamsB1[8130][200:250], color = 'orange')
# show(block=False)
# pause(1.)
# # it appears here that the peak has width about 10 channels. 

# <headingcell level=3>

# Smooth the histogram with Savitzky-Golay polynomials; subtract that; then use first-derivative of Savitzky-Golay trace to identify peaks

# <codecell>

#cwr = np.arange(2,15)             # range of wavelet sizes used in find)peaks (2,50)      
speed_per_bin = 78. / 512         # folding speed is 78 m/s; 512 bins in histogram
ptol_standard = 50                # bin difference tolerated in search to match TAS
svl = 25                          # length of Savitzky-Golay sequence (std=25; also tried 31)
svo = 3                           # order of S-G sequence (std=3, cubic)
cftr = cos(35. * np.pi / 180.) / speed_per_bin
cftr4 = 1. / speed_per_bin
p_cor1 = 0.                       # maintain correction needed to find peak, and
p_cor2 = 0.                       # use it to adjust predictions. This helps
p_cor3 = 0.                       # smooth results through folds.
p_cor4 = 0.
SGtime = 0; CWTtime = 0; OtherTime = 0.    # for timing different sections


for tcheck in xrange (0, len(lamsB1)):
#for tcheck in xrange (9258,9263):
    SG1time = time.time()
    hsts1 = savitzky_golay(lamsB1[tcheck,:], svl, svo)
    hsts2 = savitzky_golay(lamsB2[tcheck,:], svl, svo)
    hsts3 = savitzky_golay(lamsB3[tcheck,:], svl, svo)
    hsts4 = savitzky_golay(lamsB4[tcheck,:], svl, svo)
    hst1 = lamsB1[tcheck,:] - hsts1
    hst2 = lamsB2[tcheck,:] - hsts2
    hst3 = lamsB3[tcheck,:] - hsts3
    hst4 = lamsB4[tcheck,:] - hsts4
#                               # Note: savitzky_golay adds segments of svl/2
                                # to working array but does not return the
                                # extra bins, so redo that here
#           extend the smoothed array and derivative array to handle near-edge peaks better
#           Transition through fold is even for spectrum but odd for derivative (if that is used)
    firstvals = hst1[1:svl//2+1][::-1]
    lastvals  = hst1[-svl//2:-1][::-1]
    hst1 = np.concatenate((firstvals, hst1, lastvals))
#   if UseDerivatives:
#       firstvals = -hstd1[1:svl//2+1][::-1]
#       lastvals  = -hstd1[-svl//2:-1][::-1]
#       hstd1 = np.concatenate((firstvals, hstd1, lastvals))
    firstvals = hst2[1:svl//2+1][::-1]
    lastvals  = hst2[-svl//2:-1][::-1]
    hst2 = np.concatenate((firstvals, hst2, lastvals))
#   if UseDerivatives:
#        firstvals = -hstd2[1:svl//2+1][::-1]
#        lastvals  = -hstd2[-svl//2:-1][::-1]
#        hstd2 = np.concatenate((firstvals, hstd2, lastvals))
    firstvals = hst3[1:svl//2+1][::-1]
    lastvals  = hst3[-svl//2:-1][::-1]
    hst3 = np.concatenate((firstvals, hst3, lastvals))
#   if UseDerivatives:
#       firstvals = -hstd3[1:svl//2+1][::-1]
#       lastvals  = -hstd3[-svl//2:-1][::-1]
#       hstd3 = np.concatenate((firstvals, hstd3, lastvals))
    firstvals = hst4[1:svl//2+1][::-1]
    lastvals  = hst4[-svl//2:-1][::-1]
    hst4 = np.concatenate((firstvals, hst4, lastvals))
#   if UseDerivatives:
#       firstvals = -hstd4[1:svl//2+1][::-1]
#       lastvals  = -hstd4[-svl//2:-1][::-1]
#       hstd4 = np.concatenate((firstvals, hstd4, lastvals))
    SGtime += time.time() - SG1time
#    CWT1time = time.time()
#    peakind1 = signal.find_peaks_cwt(hst1, cwr)	# left from wavelet-method attempt
#    peakind2 = signal.find_peaks_cwt(hst2, cwr)
#    peakind3 = signal.find_peaks_cwt(hst3, cwr)
#    CWTtime += time.time() - CWT1time
#4B    peakind4 = signal.find_peaks_cwt(hst4, cwr)

# predict peak location from TASX, with consideration of folding: 
    if TASX[tcheck] is ma.masked:
        pred1 = 0
        pred2 = 0
        pred3 = 0
        pred4 = 0
    else:
        pred1 = min(TASX[tcheck]*cftr+p_cor1, 2048)
        pred2 = min(TASX[tcheck]*cftr4+p_cor2, 2048)
        pred3 = min(TASX[tcheck]*cftr+p_cor3, 2048)
        pred4 = min(TASX[tcheck]*cftr+p_cor4, 2048)
    if pred1 > 1536:
        pred1 = 2048 - pred1
        fold1 = 3
    elif pred1 > 1024:
        pred1 -= 1024
        fold1 = 2
    elif pred1 > 512:
        pred1 = 1024 - pred1
        fold1 = 1
    elif pred1 > 0:
        fold1 = 0
    else:
        fold1 = -1    # TAS is bad
 
    if pred2 > 1536:
        pred2 = 2048 - pred2
        fold2 = 3
    elif pred2 > 1024:
        pred2 -= 1024
        fold2 = 2
    elif pred2 > 512:
        pred2 = 1024 - pred2
        fold2 = 1
    elif pred2 > 0:
        fold2 = 0
    else:
        fold2 = -1    # TAS is bad
        
    if pred3 > 1536:
        pred3 = 2048 - pred3
        fold3 = 3
    elif pred3 > 1024:
        pred3 -= 1024
        fold3 = 2
    elif pred3 > 512:
        pred3 = 1024 - pred3
        fold3 = 1
    elif pred3 > 0:
        fold3 = 0
    else:
        fold3 = -1    # TAS is bad
    if pred4 > 1536:
        pred4 = 2048 - pred4
        fold4 = 3
    elif pred4 > 1024:
        pred4 -= 1024
        fold4 = 2
    elif pred4 > 512:
        pred4 = 1024 - pred4
        fold4 = 1
    elif pred4 > 0:
        fold4 = 0
    else:
        fold4 = -1    # TAS is bad

#         get peak value but require it to be within tolerance of prediction
    pmax1 = -1.e3
    pmax2 = -1.e3
    pmax3 = -1.e3
    pmax4 = -1.e3
    pmx1 = -100
    pmx2 = -100
    pmx3 = -100
    pmx4 = -100

    snr1 = -1.
    snr2 = -1.
    snr3 = -1.
    snr4 = -1.
    snr_tol = 4.0       # 4 looks best as standard for unaveraged 1-Hz
    h_off = svl//2
    Other1Time = time.time()
    ptol = ptol_standard if (abs(p_cor1) <= ptol_standard) else int(abs(p_cor1))
    i1 = max(h_off-1, int(pred1-ptol+h_off))
    i2 = min(int(pred1+ptol+h_off), 512+h_off)
    i = hst1[range (i1, i2)].argmax() + i1
#        print i, i1, i2, pred1, max(i-ptol, 0), max(i-ptol//2,0),  min(i+ptol//2, 512+2*h_off), min(i+ptol, 512+2*h_off), ptol, p_cor1
    rms_region = range(max(i-ptol, h_off), max(i-ptol//2, h_off)) + range(min(i+ptol//2, 512+h_off), min(i+ptol, 512+h_off))
#        print rms_region
    pmax = hst1[i]
    mn = mean (hst1[rms_region])
    snr = (pmax - mn) / (mn+std (hst1[rms_region]))
    if snr > snr_tol:
        pmx1 = i - h_off
        pmax1 = pmax
        snr1 = snr
#        print 'tcheck: ', tcheck, pmx1, pred1, p_cor1, mn, snr
    ptol = ptol_standard if (abs(p_cor2) <= ptol_standard) else int(abs(p_cor2))
    i1 = max(h_off-1, int(pred2-ptol+h_off))
    i2 = min(int(pred2+ptol+h_off), 512+h_off)
    i = hst2[range (i1, i2)].argmax() + i1
#    print i, i1, i2, pred2, max(i-ptol, 0), max(i-ptol//2,0),  min(i+ptol//2, 512+2*h_off), min(i+ptol, 512+2*h_off), ptol, p_cor2
    rms_region = range(max(i-ptol, h_off), max(i-ptol//2, h_off)) + range(min(i+ptol//2, 512+h_off), min(i+ptol, 512+h_off))
    
#    print rms_region
    pmax = hst2[i]
    mn = mean (hst2[rms_region])
    snr = (pmax - mn) / (mn+std (hst2[rms_region]))
    if snr > snr_tol:
        pmx2 = i - h_off
        pmax2 = pmax
        snr2 = snr
#    print 'tcheck: ', tcheck, pmx2, pred2, p_cor2, mn, snr
    ptol = ptol_standard if (abs(p_cor3) <= ptol_standard) else int(abs(p_cor3))
    i1 = max(h_off-1, int(pred3-ptol+h_off))
    i2 = min(int(pred3+ptol+h_off), 512+h_off)
    i = hst3[range (i1, i2)].argmax() + i1
#        print i, i1, i2, pred1, max(i-ptol, 0), max(i-ptol//2,0),  min(i+ptol//2, 512+2*h_off), min(i+ptol, 512+2*h_off), ptol, p_cor1
    rms_region = range(max(i-ptol, h_off), max(i-ptol//2, h_off)) + range(min(i+ptol//2, 512+h_off), min(i+ptol, 512+h_off))
#        print rms_region
    pmax = hst3[i]
    mn = mean (hst3[rms_region])
    snr = (pmax - mn) / (mn+std (hst3[rms_region]))
    if snr > snr_tol:
        pmx3 = i - h_off
        pmax3 = pmax
        snr3 = snr
    ptol = ptol_standard if (abs(p_cor4) <= ptol_standard) else int(abs(p_cor4))
    i1 = max(h_off-1, int(pred4-ptol+h_off))
    i2 = min(int(pred4+ptol+h_off), 512+h_off)
    i = hst4[range (i1, i2)].argmax() + i1
#        print i, i1, i2, pred1, max(i-ptol, 0), max(i-ptol//2,0),  min(i+ptol//2, 512+2*h_off), min(i+ptol, 512+2*h_off), ptol, p_cor1
    rms_region = range(max(i-ptol, h_off), max(i-ptol//2, h_off)) + range(min(i+ptol//2, 512+h_off), min(i+ptol, 512+h_off))
#        print rms_region
    pmax = hst4[i]
    mn = mean (hst4[rms_region])
    snr = (pmax - mn) / (mn+std (hst4[rms_region]))
    if snr > snr_tol:
        pmx4 = i - h_off
        pmax4 = pmax
        snr4 = snr
#        print 'tcheck: ', tcheck, pmx1, pred1, p_cor1, mn, snr


    OtherTime += time.time() - Other1Time

#                 use quadratic with two adjacent points to refine position of the peak
    if pmx1 > -1 and pmx1 < 512:
        x1 = hst1[pmx1-1+h_off]
        x2 = hst1[pmx1+h_off]
        x3 = hst1[pmx1+1+h_off]
        a=x1
        c=((x3-x1)-2*(x2-x1))/2.
        b=x2-x1-c
        pmxa1 = pmx1-b/(2*c)-1
        if abs(pmxa1-pmx1) > 1.:  # something went wrong
            pmxa1 = pmx1
    else:
        pmxa1 = pmx1
    if pmx2 > -1 and pmx2 < 512:
        x1 = hst2[pmx2-1+h_off]
        x2 = hst2[pmx2+h_off]
        x3 = hst2[pmx2+1+h_off]
        a=x1
        c=((x3-x1)-2*(x2-x1))/2.
        b=x2-x1-c
        pmxa2 = pmx2-b/(2*c)-1
        if abs(pmxa2-pmx2) > 1.:  # something went wrong
            pmxa2 = pmx2
    else:
        pmxa2 = pmx2
    if pmx3 > -1 and pmx3 < 512:
        x1 = hst3[pmx3-1+h_off]
        x2 = hst3[pmx3+h_off]
        x3 = hst3[pmx3+1+h_off]
        a=x1
        c=((x3-x1)-2*(x2-x1))/2.
        b=x2-x1-c
        pmxa3 = pmx3-b/(2*c)-1
        if abs(pmxa3-pmx3) > 1.:  # something went wrong
            pmxa3 = pmx3
    else:
        pmxa3 = pmx3
    if pmx4 > -1 and pmx4 < 512:
        x1 = hst4[pmx4-1+h_off]
        x2 = hst4[pmx4+h_off]
        x3 = hst4[pmx4+1+h_off]
        a=x1
        c=((x3-x1)-2*(x2-x1))/2.
        b=x2-x1-c
        pmxa4 = pmx4-b/(2*c)-1
        if abs(pmxa4-pmx4) > 1.:  # something went wrong
            pmxa4 = pmx4
    else:
        pmxa4 = pmx4
    pmxa1 += 0.5                  # adjust to center of bin
    pmxa2 += 0.5
    pmxa3 += 0.5
    pmxa4 += 0.5

    if fold1 == 0:
        BEAM1speed[tcheck] = pmxa1 * speed_per_bin
    elif fold1 == 1:
        BEAM1speed[tcheck] = 78. + (512.-pmxa1) * speed_per_bin
    elif fold1 == 2:
        BEAM1speed[tcheck] = 156. + pmxa1 * speed_per_bin
    elif fold1 == 3:
        BEAM1speed[tcheck] = 234. + (512.-pmxa1) * speed_per_bin
    else:
        BEAM1speed[tcheck] = ma.masked
                # if near fold points, choose fold that gives smallest jump
    if pred1%512 < 40:
        BEAM1pred = BEAM1last + BEAM1slope
        if abs(BEAM1speed[tcheck] - BEAM1pred) > \
           abs(BEAM1speed[tcheck] - 2*pmxa1*speed_per_bin - BEAM1pred):
            BEAM1speed[tcheck] -= 2.*pmxa1*speed_per_bin
        elif abs(BEAM1speed[tcheck] - BEAM1pred) > \
           abs(BEAM1speed[tcheck] + 2*pmxa1*speed_per_bin - BEAM1pred):
            BEAM1speed[tcheck] += 2.*pmxa1*speed_per_bin
    if pred1%512 > 472:
        BEAM1pred = BEAM1last + BEAM1slope
        if abs(BEAM1speed[tcheck] - BEAM1pred) > \
           abs(BEAM1speed[tcheck] - 2*(512-pmxa1)*speed_per_bin - BEAM1pred):
            BEAM1speed[tcheck] -= 2.*(512-pmxa1)*speed_per_bin
        elif abs(BEAM1speed[tcheck] - BEAM1pred) > \
           abs(BEAM1speed[tcheck] + 2*(512-pmxa1)*speed_per_bin - BEAM1pred):
            BEAM1speed[tcheck] += 2.*(512-pmxa1)*speed_per_bin
    if BEAM1speed[tcheck] > 0. and BEAM1speed[tcheck] < 250.:
        BEAM1slope += ((BEAM1speed[tcheck] - BEAM1last) - BEAM1slope) / 5.	# 5-sample exponential updating
        BEAM1last = BEAM1speed[tcheck]
    
    if fold2 == 0:
        BEAM2speed[tcheck] = pmxa2 * speed_per_bin
    elif fold2 == 1:
        BEAM2speed[tcheck] = 78. + (512.-pmxa2) * speed_per_bin
    elif fold2 == 2:
        BEAM2speed[tcheck] = 156. + pmxa2 * speed_per_bin
    elif fold2 == 3:
        BEAM2speed[tcheck] = 234. + (512.-pmxa2) * speed_per_bin
    else:
        BEAM2speed[tcheck] = ma.masked
                # if near fold points, choose fold that gives smallest jump
    if pred2%512 < 40:
        BEAM2pred = BEAM2last + BEAM2slope
        if abs(BEAM2speed[tcheck] - BEAM2pred) > \
           abs(BEAM2speed[tcheck] - 2*pmxa2*speed_per_bin - BEAM2pred):
            BEAM2speed[tcheck] -= 2.*pmxa2*speed_per_bin
        elif abs(BEAM2speed[tcheck] - BEAM2pred) > \
           abs(BEAM2speed[tcheck] + 2*pmxa2*speed_per_bin - BEAM2pred):
            BEAM2speed[tcheck] += 2.*pmxa2*speed_per_bin
    if pred2%512 > 472:
        BEAM2pred = BEAM2last + BEAM2slope
        if abs(BEAM2speed[tcheck] - BEAM2pred) > \
           abs(BEAM2speed[tcheck] - 2*(512-pmxa2)*speed_per_bin - BEAM2pred):
            BEAM2speed[tcheck] -= 2.*(512-pmxa2)*speed_per_bin
        elif abs(BEAM2speed[tcheck] - BEAM2pred) > \
           abs(BEAM2speed[tcheck] + 2*(512-pmxa2)*speed_per_bin - BEAM2pred):
            BEAM2speed[tcheck] += 2.*(512-pmxa2)*speed_per_bin
    if BEAM2speed[tcheck] > 0. and BEAM2speed[tcheck] < 250.:
        BEAM2slope += ((BEAM2speed[tcheck] - BEAM2last) - BEAM2slope) / 5.
        BEAM2last = BEAM2speed[tcheck]

    if fold3 == 0:
        BEAM3speed[tcheck] = pmxa3 * speed_per_bin
    elif fold3 == 1:
        BEAM3speed[tcheck] = 78. + (512.-pmxa3) * speed_per_bin
    elif fold3 == 2:
        BEAM3speed[tcheck] = 156. + pmxa3 * speed_per_bin
    elif fold3 == 3:
        BEAM3speed[tcheck] = 234. + (512.-pmxa3) * speed_per_bin
    else:
        BEAM3speed[tcheck] = ma.masked
                # if near fold points, choose fold that gives smallest jump
    if pred3%512 < 40:
        BEAM3pred = BEAM3last + BEAM3slope
        if abs(BEAM3speed[tcheck] - BEAM3pred) > \
           abs(BEAM3speed[tcheck] - 2*pmxa3*speed_per_bin - BEAM3pred):
            BEAM3speed[tcheck] -= 2.*pmxa3*speed_per_bin
        elif abs(BEAM3speed[tcheck] - BEAM3pred) > \
           abs(BEAM3speed[tcheck] + 2*pmxa3*speed_per_bin - BEAM3pred):
            BEAM3speed[tcheck] += 2.*pmxa3*speed_per_bin
    if pred3%512 > 472:
        BEAM3pred = BEAM3last + BEAM3slope
        if abs(BEAM3speed[tcheck] - BEAM3pred) > \
           abs(BEAM3speed[tcheck] - 2*(512-pmxa3)*speed_per_bin - BEAM3pred):
            BEAM3speed[tcheck] -= 2.*(512-pmxa3)*speed_per_bin
        elif abs(BEAM3speed[tcheck] - BEAM3pred) > \
           abs(BEAM3speed[tcheck] + 2*(512-pmxa3)*speed_per_bin - BEAM3pred):
            BEAM3speed[tcheck] += 2.*(512-pmxa3)*speed_per_bin
    if BEAM3speed[tcheck] > 0. and BEAM3speed[tcheck] < 250.:
        BEAM3slope += ((BEAM3speed[tcheck] - BEAM3last) - BEAM3slope) / 5.
        BEAM3last = BEAM3speed[tcheck]
    if fold4 == 0:
        BEAM4speed[tcheck] = pmxa4 * speed_per_bin
    elif fold4 == 1:
        BEAM4speed[tcheck] = 78. + (512.-pmxa4) * speed_per_bin
    elif fold4 == 2:
        BEAM4speed[tcheck] = 156. + pmxa4 * speed_per_bin
    elif fold4 == 3:
        BEAM4speed[tcheck] = 234. + (512.-pmxa4) * speed_per_bin
    else:
        BEAM4speed[tcheck] = ma.masked
                # if near fold points, choose fold that gives smallest jump
    if pred4%512 < 40:
        BEAM4pred = BEAM4last + BEAM4slope
        if abs(BEAM4speed[tcheck] - BEAM4pred) > \
           abs(BEAM4speed[tcheck] - 2*pmxa4*speed_per_bin - BEAM4pred):
            BEAM4speed[tcheck] -= 2.*pmxa4*speed_per_bin
        elif abs(BEAM4speed[tcheck] - BEAM4pred) > \
           abs(BEAM4speed[tcheck] + 2*pmxa4*speed_per_bin - BEAM4pred):
            BEAM4speed[tcheck] += 2.*pmxa4*speed_per_bin
    if pred4%512 > 472:
        BEAM4pred = BEAM4last + BEAM4slope
        if abs(BEAM4speed[tcheck] - BEAM4pred) > \
           abs(BEAM4speed[tcheck] - 2*(512-pmxa4)*speed_per_bin - BEAM4pred):
            BEAM4speed[tcheck] -= 2.*(512-pmxa4)*speed_per_bin
        elif abs(BEAM4speed[tcheck] - BEAM4pred) > \
           abs(BEAM4speed[tcheck] + 2*(512-pmxa4)*speed_per_bin - BEAM4pred):
            BEAM4speed[tcheck] += 2.*(512-pmxa4)*speed_per_bin
    if BEAM4speed[tcheck] > 0. and BEAM4speed[tcheck] < 250.:
        BEAM4slope += ((BEAM4speed[tcheck] - BEAM4last) - BEAM4slope) / 5.
        BEAM4last = BEAM4speed[tcheck]
    if pmx1 < 0:
        BEAM1speed[tcheck] = ma.masked
#        print tcheck
    if pmx2 < 0:
        BEAM2speed[tcheck] = ma.masked
    if pmx3 < 0:
        BEAM3speed[tcheck] = ma.masked
    if pmx4 < 0:
        BEAM4speed[tcheck] = ma.masked  
#    print tcheck, BEAM2speed[tcheck]


    if tcheck%1000 == 0:
        print 'LAMS histogram analysis: ', tcheck, ' histograms processed ', SGtime, CWTtime, OtherTime
      
    

# <rawcell>

#    
#      
#  
#            
# # make this segment code (and merge with above cell) to see plots, or raw text (and separate from above cell) to skip plots
    if tcheck >= 9265 and tcheck < 9265:
        clf()
        timeh   = np.linspace(-12,523,536)
        a = subplot(211)
        a.plot(timeh,hst2)
        if UseDerivatives:
            a.plot(timeh,100.*hstd2, color='orange', lw=2)
        a.plot(timeh[12:524], hsts2, color='red')
        a.plot(timeh[12:524], lamsB2[tcheck,:], color='green')
        yl = plt.ylim()
    #ttl = format(tcheck, 'd') + ' TAS = ' + format(TASX[tcheck],'.1f') 
        #+ ', pred = ' + format(pred, '.1f') + ', selected peak: ' 
        #+ format (pmxa1,'.1f') + ' SNR ' + format(snr1,'.1f')
        ttl = "%d TAS=%.1f pred=%.1f peak %.1f p_cor %.1f SNR %.1f" % (tcheck, TASX[tcheck], pred2, pmxa2, p_cor2, snr2)    
        title(ttl)
        a.plot([pmxa2,pmxa2],yl,color='red', lw=3)
        a.plot([pred2,pred2],yl,color='black', lw=2)
#     
# #    show()
        if pmxa2 < -99:
#           show(block=False)
            show()
            print tcheck
#           pause(5.)
        else:
            show()
#           pause(0.1)
    if not isnan(TASX[tcheck]):
        if pmxa1 > -h_off:
            p_cor1 += (pmxa1 - pred1)/100.
            p_cor1 *= 0.97
        if pmxa2 > -h_off:
            p_cor2 += (pmxa2 - pred2)/100.
            p_cor2 *= 0.97
        if pmxa3 > -h_off:
            p_cor3 += (pmxa3 - pred3)/100.
            p_cor3 *= 0.97
        if pmxa4 > -h_off:
            p_cor4 += (pmxa4 - pred4)/100.
            p_cor4 *= 0.97
#    p_cor4 += (pmxa4 - pred4)/10.
#    print TASX[tcheck], ' prediction = ', pred1, ', p_cor = ', p_cor1, ', pmxa = ', pmxa1
# print 'end of line-of-sight-airspeed code'
#         

# <rawcell>

# R = np.array(range (0,512))
# Ri = (abs(hstd1) < 0.01e6)
# #print R[Ri==True]
# for i in range (0,511):
#     if hstd1[i] > 0. and hstd1[i+1] < 0. and (hstd1[i]-hstd1[i+1]) > 2.e5:
#         rms_region = range(max(i-ptol, 0), max(i-ptol/2,0)) + range(min(i+ptol/2, 512), min(i+ptol, 512))
#         snr = pmax1 / (mean (hst1[rms_region])+std (hst1[rms_region]))
#         print i,snr, (hst1[i]+hst1[i+1])/2.

# <codecell>

print 'After calculation of beam speeds, elapsed time: ', time.time () - Stime, ' s'            

# <rawcell>

# # Leave this commented (i.e., Raw Text)
# # This was a flawed attempt to use fft filtering to isolate the peak. The flaw is that the 512 bins are not cyclic and the end-point transitions can lead
# # to trouble identifying the peak, although as below the zero padding helps a little. This is saved here, but this was an early step abandoned in favor 
# # of using a polynomial representation of the background.
# 
# from scipy.fftpack import rfft, irfft, fftfreq
# import pylab as plt
# timeh   = np.linspace(0,1023,1024)
# hst = ma.empty(1024)
# hst[0:256] = lamsB1[8100][0]
# hst[256:768] = lamsB1[8100][0:512]
# hst[768:1024] = lamsB1[8100][511]
# print len(hst), len(timeh)
# W = fftfreq(hst.size, d=timeh[1]-timeh[0])
# print hist(W)
# f_signal = rfft(hst)
# # If the original signal time was in seconds, this is now in Hz    
# cut_f_signal = f_signal.copy()
# #cut_f_signal[(W<0.1)] *= W[W<0.1]*10.
# #cut_f_signal[(W>0.1)] *= exp(-(W[W>0.1]+0.1))
# cw = 0.05
# cut_f_signal[W < cw] *= exp((W[W < cw]/cw-1.)*10.)
# #cut_f_signal[W<0.0005] = 0.
# cut_signal = irfft(cut_f_signal)

# <rawcell>

#clf()
#R = range(9270,9500)
#R = range (9100, 9300)
#plot(BEAM1speed[R])
#plot(BEAM2speed[R])
#plot(BEAM3speed[R])
#show()

# <rawcell>

# clf()
# #a = subplot(221)
# a = subplot(211)
# a.plot(timeh,hst)
# a.plot(timeh[256:768], hsts, color='red')
# #b = subplot(222)
# #b.plot(W,f_signal)
# #plt.xlim(0.,0.5)
# #c = subplot(223)
# #c.plot(W,cut_f_signal)
# #plt.xlim(0.,0.5)
# #d = subplot(224)
# d = subplot(212)
# d.plot(timeh,cut_signal)
# #show()

# <codecell>

# advance CROLL, CPITCH, CTHDG by N 25-Hz samples
if DL/DLEN == 25:
    N = 0  # set this to advance (positive) or delay (negative) the sequences
    if N > 0:
        CROLL = np.append (CROLL, N*[0])
        CROLL = np.delete (CROLL, arange(0,N))
        CPITCH = np.append (CPITCH, N*[0])
        CPITCH = np.delete (CPITCH, arange(0,N))
        CTHDG = np.append (CTHDG, N*[0])
        CTHDG = np.delete (CTHDG, arange(0,N))
    if N < 0:
        N *= -1 
        CROLL = np.insert (CROLL, 0, N*[CROLL[0]])
        CROLL = np.delete (CROLL, arange(DL-N,DL))
        CPITCH = np.insert (CPITCH, 0, N*[CPITCH[0]])
        CPITCH = np.delete (CPITCH, arange(DL-N,DL))
        CTHDG = np.insert (CTHDG, 0, N*[CTHDG[0]])
        CTHDG = np.delete (CTHDG, arange(DL-N,DL))

# advance VNSC and VEWC by N samples;
    N = 0  # set this to the advance (+ve) or delay (-ve) needed
#@@    N = 2
    if N > 0:
        VNSC = np.append (VNSC, N*[0])
        VEWC = np.append (VEWC, N*[0])
        VNSC = np.delete (VNSC, arange(0,N))
        VEWC = np.delete (VEWC, arange(0,N))
    if N < 0:
        N *= -1
        VNSC = np.insert (VNSC, 0, N*[VNSC[0]])
        VEWC = np.insert (VEWC, 0, N*[VEWC[0]])
        VNSC = np.delete (VNSC, arange (DL-N,DL))
        VEWC = np.delete (VEWC, arange (DL-N,DL))
  
# arrays used in setting up rotation matrices
Z =   ma.zeros (DL)
One = ma.ones (DL)

print 'Have data in python arrays at', time.time () - Stime, ' s'

# <headingcell level=2>

# Retrieve the conventional relative wind 

# <codecell>

# Construct the relative-wind array, a DL x 3 array with
# first dimension the time index and 2nd components u,v,w. This
# is the relative wind as measured by the radome and pitot-tube system.
# It is included here to be able to compare to the corresponding LAMS results
UR = TASX
VR = TASX * ma.tan (SSLIP * pi/180.)
WR = TASX * ma.tan (AKRD * pi/180.)

# relative-wind array: RWR as measured by radome gust system
RWR = transpose (ma.array([UR,VR,WR], fill_value=-32767.))
print 'Have radome-based relative wind array at ', time.time () - Stime, ' s'

# <headingcell level=2>

# Get the rotation-rate array for use below

# <markdowncell>

# Bulletin 23 gives the expected addition to the relative wind $\vec{v}_a$ caused by rotation: 
# 
# $$\hskip2in \vec{v}_a^\prime=\vec{v}_a+\vec{\Omega}_p\times\vec{R}$$
# 
# where $\vec{\Omega}_{p}$ is the angular rotation rate or angular velocity of the aircraft.
# Here $\vec{R}$ is the vector from the aircraft reference location at the IRS to the LAMS, 
# represented by the location of the CMIGITS.

# <codecell>

# get rotation-rate-vector array:
Cradeg = pi / 180.  # convert to radians from degrees 
Omega = ma.array ([ROLL, PITCH, THDG], fill_value = -32767.) * Cradeg
Omega = np.diff (Omega.T, axis=0)                 # this differentiates step-wise to get the rotation rates
Omega[:,2][Omega[:,2] > pi] -= 2.*pi              # handle where THDG goes through 360 degrees
Omega[:,2][Omega[:,2] < -pi] += 2.*pi
Omega *= SampleRate
Omega = np.append (Omega, ma.zeros((1,3)), axis=0) # to match dimensions of other arrays

# get false contribution to relative wind from rotation:
F = ma.array (np.cross (Omega, LL), fill_value=-32767.)
print 'Rotation-rate correction generated at ', time.time () - Stime, ' s'

# <headingcell level=2>

# Transform relative wind to the Earth-reference system:

# <markdowncell>

# For a description of the procedure that follows, see 'LAMSprocessing3Dwind.pdf' on the RAF science wiki: http://wiki.eol.ucar.edu/rafscience/LAMS%20Processing?action=AttachFile&do=view&target=LAMSprocessing3Dwind.pdf . 

# <headingcell level=3>

# Get the three-beam measurements of airspeed:

# <codecell>

A = transpose (ma.array ([BEAM1speed, BEAM2speed, BEAM3speed], fill_value=-32767.))

# <headingcell level=3>

# Get the u,v,w components at LAMS and correct for rotation rate:

# <codecell>

RW = transpose (ma.dot (Si, A.T)) - F  # F is the rotation-rate correction

# <headingcell level=3>

# Rotate to the Earth-reference system:

# <markdowncell>

# If {$\phi$, $\theta$, $\psi$} are respectively roll, pitch, and heading, the needed transformation matrices are:
# 
# 
# $$\hskip2in T_1=\left(\begin{array}{ccc}1 & 0 & 0\\\0 & \cos\phi & -\sin\phi\\\0 & \sin\phi & \cos\phi\end{array}\right)$$
# 
# $$\hskip2in T_2=\left(\begin{array}{ccc}\cos\theta & 0 & \sin\theta\\\0 & 1 & 0\\\ -\sin\theta & 0 & \cos\theta\end{array}\right)$$
# 
# $$\hskip2in T_3=\left(\begin{array}{ccc}\cos\psi & -\sin\psi & 0\\\sin\psi & \cos\psi & 0\\\0 & 0 & 1\end{array}\right)$$

# <codecell>

# first, correct for an assumed misalignment between the LAMS and CMIGITS reference frames
#@@CR = ma.ones (DL) * (2.) * Cradeg 
CR = ma.ones (DL) * (0.) * Cradeg 
CP = ma.ones (DL) * 0. * Cradeg
#@@CH = ma.ones (DL) * 0.3 * Cradeg
#@@CH = ma.ones (DL) * 0.4 * Cradeg
CH = ma.zeros (DL)
T1 = ma.array ([One,Z,Z, Z,cos(CR),-sin(CR), Z,sin(CR),cos(CR)])  # rotation about x
T2 = ma.array ([cos(CP),Z,sin(CP), Z,One,Z, -sin(CP),Z,cos(CP)])  # rotation about y
T3 = ma.array ([cos(CH),-sin(CH),Z, sin(CH),cos(CH),Z, Z,Z,One])  # rotation about z
T1.shape = (3,3,len(CH))
T2.shape = (3,3,len(CH))
T3.shape = (3,3,len(CH))
for i in range (0,DL):
    RW[i,:] = ma.dot (T3[...,i], ma.dot (T2[...,i], ma.dot (T1[...,i], RW[i,:])))
print 'Rotated relative wind to assumed CMIGITS reference frame at ', time.time () - Stime, ' s'

# <codecell>

#RWT = np.dot (T3, np.dot(T2, np.dot(T1, RW))) -- I wish! Try later to get an array statement to work
RWT = ma.empty (shape (RW))
CR = CROLL * Cradeg
CP = CPITCH * Cradeg
CH = CTHDG * Cradeg
T1 = ma.array ([One,Z,Z, Z,cos(CR),-sin(CR), Z,sin(CR),cos(CR)])  # rotation about x
T2 = ma.array ([cos(CP),Z,sin(CP), Z,One,Z, -sin(CP),Z,cos(CP)])  # rotation about y
T3 = ma.array ([cos(CH),-sin(CH),Z, sin(CH),cos(CH),Z, Z,Z,One])  # rotation about z
T1.shape = (3,3,len(CR))
T2.shape = (3,3,len(CP))
T3.shape = (3,3,len(CH))
for i in range (0,DL):
    RWT[i,:] = ma.dot (T3[...,i], ma.dot (T2[...,i], ma.dot (T1[...,i], RW[i,:])))
print 'Rotated relative wind to Earth reference frame at ', time.time () - Stime, ' s'
 

# <headingcell level=2>

# Find the wind in the Earth reference frame:

# <markdowncell>

# RWT is now the relative wind in the earth reference frame, but still with {x,y,z} {north, east, down}.
# These components can now be used to find the Earth-relative wind by subtracting from them the
# corresponding velocity components of the aircraft, {VNSC, VEWC, VSPD}. Subtraction is needed
# because wind is characterized by the direction ${\bf from\ which}$ the wind blows, while the
# aircraft motion is characterized by motion toward the corresponding direction. Subtracting the
# aircraft velocity then gives
# 
#         (-air motion wrt Earth) = (-air motion wrt aircraft) - (aircraft motion wrt Earth)
# 
#         (Earth-relative wind)   = (LAMS-measured wind)       - (aircraft motion wrt Earth)
# 
# For the vertical wind: positive is upward and LAMS measures positive upward as positive incoming along the downward-pointing z axis. 
# Upward motion of the aircraft should be $\underline{added}$ to this, so the sign of VSPD must be reversed:

# <codecell>

   
RWG = RWT  # relative wind wrt ground; save for use later
GWA = transpose (ma.array([VNSC, VEWC, -1.*VSPD], fill_value=-32767.))
GW = RWG - GWA
print 'Have Earth-relative wind at ', time.time () - Stime, ' s'

# <headingcell level=2>

# Transform LAMS-relative wind to the aircraft reference frame:

# <markdowncell>

# For direct comparison to relative wind in the aircraft (Honeywell) reference frame, transform back to that frame. For this transformation, the signs of the attitude angles are reversed and the order in which the rotations is applied is also reversed. 
# 
# This section is not needed to get Earth-relative wind, so later it can be omitted. This is useful in the development stage, though.

# <codecell>

# 5% of run time -- skip later if not needed to just get wind
CR = -1. * ROLL * Cradeg    # negative sign because now doing the transform in the reverse direction
CP = -1. * PITCH * Cradeg
CH = -1. * THDG * Cradeg
T1 = ma.array ([One,Z,Z, Z,cos(CR),-sin(CR), Z,sin(CR),cos(CR)])  # rotation about x
T2 = ma.array ([cos(CP),Z,sin(CP), Z,One,Z, -sin(CP),Z,cos(CP)])  # rotation about y
T3 = ma.array ([cos(CH),-sin(CH),Z, sin(CH),cos(CH),Z, Z,Z,One])  # rotation about z
T1.shape = (3,3,len(CR))
T2.shape = (3,3,len(CP))
T3.shape = (3,3,len(CH))
for i in range (0,DL):  # note reversal in order of transformations
    RWT[i,:] = ma.dot (T1[...,i], ma.dot (T2[...,i], ma.dot (T3[...,i], RWT[i,:])))
print 'Transformed back to aircraft reference frame at ', time.time () - Stime, ' s'

# <headingcell level=2>

# Create the netCDF variables and write the file

# <codecell>

# create new netCDF variables for {u,v,w} and for {WD, WS, WI}:
if len (Shape) < 2:
    BEAM1CDF = netCDFfile.createVariable ('BEAM1speed', 'f4', ('Time'), fill_value=-32767.)
    BEAM2CDF = netCDFfile.createVariable ('BEAM2speed', 'f4', ('Time'), fill_value=-32767.)
    BEAM3CDF = netCDFfile.createVariable ('BEAM3speed', 'f4', ('Time'), fill_value=-32767.)
    BEAM4CDF = netCDFfile.createVariable ('BEAM4speed', 'f4', ('Time'), fill_value=-32767.)
    ULAMSCDF = netCDFfile.createVariable ('U_LAMS', 'f4', ('Time'), fill_value=-32767.)
    VLAMSCDF = netCDFfile.createVariable ('V_LAMS', 'f4', ('Time'), fill_value=-32767.)
    WLAMSCDF = netCDFfile.createVariable ('W_LAMS', 'f4', ('Time'), fill_value=-32767.)
    WDLAMSCDF = netCDFfile.createVariable ('WD_LAMS', 'f4', ('Time'), fill_value=-32767.)
    WSLAMSCDF = netCDFfile.createVariable ('WS_LAMS', 'f4', ('Time'), fill_value=-32767.)
    WILAMSCDF = netCDFfile.createVariable ('WI_LAMS', 'f4', ('Time'), fill_value=-32767.)
    ATTACKLCDF = netCDFfile.createVariable ('ATTACK_L', 'f4', ('Time'), fill_value = -32767.)
    SSLIPLCDF = netCDFfile.createVariable ('SSLIP_L', 'f4', ('Time'), fill_value = -32767.)
else:  
    HRdim = DL / DLEN
    HRdimName = 'sps' + format (HRdim, 'd')
    BEAM1CDF = netCDFfile.createVariable ('BEAM1speed', 'f4', ('Time', HRdimName), fill_value=-32767.)
    BEAM2CDF = netCDFfile.createVariable ('BEAM2speed', 'f4', ('Time', HRdimName), fill_value=-32767.)
    BEAM3CDF = netCDFfile.createVariable ('BEAM3speed', 'f4', ('Time', HRdimName), fill_value=-32767.)
    BEAM4CDF = netCDFfile.createVariable ('BEAM4speed', 'f4', ('Time', HRdimName), fill_value=-32767.)
    ULAMSCDF = netCDFfile.createVariable ('U_LAMS', 'f4', ('Time', HRdimName), fill_value=-32767.)
    VLAMSCDF = netCDFfile.createVariable ('V_LAMS', 'f4', ('Time', HRdimName), fill_value=-32767.)
    WLAMSCDF = netCDFfile.createVariable ('W_LAMS', 'f4', ('Time', HRdimName), fill_value=-32767.)
    WDLAMSCDF = netCDFfile.createVariable ('WD_LAMS', 'f4', ('Time', HRdimName), fill_value=-32767.)
    WSLAMSCDF = netCDFfile.createVariable ('WS_LAMS', 'f4', ('Time', HRdimName), fill_value=-32767.)
    WILAMSCDF = netCDFfile.createVariable ('WI_LAMS', 'f4', ('Time', HRdimName), fill_value=-32767.)
    ATTACKLCDF = netCDFfile.createVariable ('ATTACK_L', 'f4', ('Time', HRdimName), fill_value = -32767.)
    SSLIPLCDF = netCDFfile.createVariable ('SSLIP_L', 'f4', ('Time', HRdimName), fill_value = -32767.)
print 'Created new netCDF variables at ', time.time () - Stime, ' s'

# <codecell>

# 61% of total run time is spend in this cell, so consider a better way to define these attributes?
# for test runs, change this cell to 'raw text' to suppress execution
#add attributes
BEAM1CDF.units = 'm/s'
BEAM3CDF.units = 'm/s'
BEAM3CDF.units = 'm/s'
BEAM4CDF.units = 'm/s'
ULAMSCDF.units = 'm/s'
WLAMSCDF.units = 'm/s'
WLAMSCDF.units = 'm/s'
WDLAMSCDF.units = 'degrees wrt true north'
WSLAMSCDF.units = 'm/s'
WILAMSCDF.units = 'm/s'
ATTACKLCDF.units = 'degrees'
SSLIPLCDF.units = 'degrees'
BEAM1CDF.long_name = 'LAMS beam-1 line-of-sight airspeed'
BEAM2CDF.long_name = 'LAMS beam-2 line-of-sight airspeed'
BEAM3CDF.long_name = 'LAMS beam-3 line-of-sight airspeed'
BEAM4CDF.long_name = 'LAMS beam-4 line-of-sight airspeed'
ULAMSCDF.long_name = 'LAMS-derived longitudinal component of the relative wind (from ahead)'
VLAMSCDF.long_name = 'LAMS-derived lateral component of the relative wind (from starboard)'
WLAMSCDF.long_name = 'LAMS-derived vertical component of the relative wind (from below)'
WDLAMSCDF.long_name = 'LAMS-derived wind direction from true north'
WSLAMSCDF.long_name = 'LAMS-derived wind speed)'
WILAMSCDF.long_name = 'LAMS-derived vertical wind'
ATTACKLCDF.long_name = 'LAMS-derived angle of attack'
SSLIPLCDF.long_name = 'LAMS-derived sideslip angle'
BEAM1CDF.standard_name = 'BEAM1LOS'
BEAM2CDF.standard_name = 'BEAM2LOS'
BEAM3CDF.standard_name = 'BEAM3LOS'
BEAM4CDF.standard_name = 'BEAM4LOS'
ULAMSCDF.standard_name = 'U_LAMS'
VLAMSCDF.standard_name = 'V_LAMS'
WLAMSCDF.standard_name = 'W_LAMS'
WSLAMSCDF.standard_name = 'WD_LAMS'
WDLAMSCDF.standard_name = 'WS_LAMS'
WILAMSCDF.standard_name = 'WI_LAMS'
ATTACKLCDF.standard_name = 'ATTACK_L'
SSLIPLCDF.standard_name = 'SSLIP_L'
#print (RWT[0,np.unravel_index (RWT[0,:].argmax (), RWT[0,:].shape)])

# is actual_range attribute needed? Present for normal nimbus-generated
# variables, but not supplied here
BEAM1CDF.Category = 'Winds'
BEAM2CDF.Category = 'Winds'
BEAM3CDF.Category = 'Winds'
BEAM4CDF.Category = 'Winds'
ULAMSCDF.Category = 'Winds'
VLAMSCDF.Category = 'Winds'
WLAMSCDF.Category = 'Winds'
WDLAMSCDF.Category = 'Winds'
WSLAMSCDF.Category = 'Winds'
WILAMSCDF.Category = 'Winds'
ATTACKLCDF.Category = 'Winds'
SSLIPLCDF.Category = 'Winds'
BEAM1CDF.DataQuality = 'Preliminary'
BEAM2CDF.DataQuality = 'Preliminary'
BEAM3CDF.DataQuality = 'Preliminary'
BEAM4CDF.DataQuality = 'Preliminary'
ULAMSCDF.DataQuality = 'Preliminary'
VLAMSCDF.DataQuality = 'Preliminary'
WLAMSCDF.DataQuality = 'Preliminary'
WDLAMSCDF.DataQuality = 'Preliminary'
WSLAMSCDF.DataQuality = 'Preliminary'
WILAMSCDF.DataQuality = 'Preliminary'
ATTACKLCDF.DataQuality = 'Preliminary'
SSLIPLCDF.DataQuality = 'Preliminary'
BEAM1CDF.Dependencies = '2 BEAM1hist TASX'
BEAM2CDF.Dependencies = '2 BEAM2hist TASX'
BEAM3CDF.Dependencies = '2 BEAM3hist TASX'
BEAM4CDF.Dependencies = '2 BEAM4hist TASX'
ULAMSCDF.Dependencies = '9 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS'
VLAMSCDF.Dependencies = '9 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS'
WLAMSCDF.Dependencies = '9 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS'
WDLAMSCDF.Dependencies = '11 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS VNSC VEWC'
WSLAMSCDF.Dependencies = '11 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS VNSC VEWC'
WILAMSCDF.Dependencies = '10 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS VSPD'
ATTACKLCDF.Dependencies = '9 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS'
SSLIPLCDF.Dependencies = '9 BEAM1speed BEAM2speed BEAM3speed ROLL PITCH THDG CROLL_LAMS CPITCH_LAMS CTHDG_LAMS'
print 'Created netcdf attributes at ', time.time () - Stime, ' s'

# <codecell>

# put the unmasked data in the netCDF file, with masked values replaced by -32767.
RWT.mask[:,0] |= BEAM1speed.mask
GW.mask[:,0]  |= BEAM1speed.mask
RWT.mask[:,1] |= BEAM2speed.mask
GW.mask[:,1]  |= BEAM2speed.mask
RWT.mask[:,2] |= BEAM3speed.mask
GW.mask[:,2]  |= BEAM3speed.mask
Hdiff = CTHDG - THDG
Hdiff[Hdiff > 180.] -= 360.
Hdiff[Hdiff < -180.] += 360.
Hdiff = abs (Hdiff)
RWT.mask[:,0] |= (Hdiff > 5.)    # avoid bad heading from CMIGITS
GW.mask[:,0]  |= (Hdiff > 5.)
RWT.mask[:,1] |= (Hdiff > 5.)   
GW.mask[:,1]  |= (Hdiff > 5.)
RWT.mask[:,2] |= (Hdiff > 5.)  
GW.mask[:,2]  |= (Hdiff > 5.)
WDL = ma.arctan2 (GW[:,1], GW[:,0]) / Cradeg  # wind direction based on LAMS
WDL[WDL < 0.] += 360.
WDL[WDL > 360.] -= 360.
WSL = ma.array ((GW[:,0]**2 + GW[:,1]**2)**0.5, fill_value=-32767.)
#WSL = ma.masked_where ((GW.mask[:,0] | GW.mask[:,1]), WSL)
WDL.data[WDL.mask] = -32767.
WSL.data[WSL.mask] = -32767.
GW.data[GW.mask[:,2],2] = -32767.
Attack = ma.array ((RWT[:,2] / RWT[:,0]), fill_value=-32767.)
Attack = ma.arctan (Attack) / Cradeg
Sslip = ma.array ((RWT[:,1] / RWT[:,0]), fill_value=-32767.)
Sslip = ma.arctan (Sslip) / Cradeg
BEAM1speed.data[BEAM1speed.mask] = -32767.
BEAM2speed.data[BEAM2speed.mask] = -32767.
BEAM3speed.data[BEAM3speed.mask] = -32767.
BEAM4speed.data[BEAM4speed.mask] = -32767.
if len (Shape) < 2:
    BEAM1CDF[:] = BEAM1speed[:].data
    BEAM2CDF[:] = BEAM2speed[:].data
    BEAM3CDF[:] = BEAM3speed[:].data
    BEAM4CDF[:] = BEAM4speed[:].data
    ULAMSCDF[:] = RWT[:,0].data
    VLAMSCDF[:] = RWT[:,1].data 
    WLAMSCDF[:] = RWT[:,2].data
    WDLAMSCDF[:] = ma.array (WDL, fill_value=-32767.).data
    WSLAMSCDF[:] = ma.array (WSL, fill_value=-32767.).data
    WILAMSCDF[:] = GW[:,2].data
    ATTACKLCDF[:] = Attack[:].data
    SSLIPLCDF[:] = Sslip[:].data
else:
    BEAM1CDF[:] = BEAM1speed[:].data.reshape (DLEN, Shape[1])
    BEAM2CDF[:] = BEAM2speed[:].data.reshape (DLEN, Shape[1])
    BEAM3CDF[:] = BEAM3speed[:].data.reshape (DLEN, Shape[1])
    BEAM4CDF[:] = BEAM4speed[:].data.reshape (DLEN, Shape[1])
    ULAMSCDF[:] = RWT[:,0].data.reshape (DLEN, Shape[1])
    VLAMSCDF[:] = RWT[:,1].data.reshape (DLEN, Shape[1]) 
    WLAMSCDF[:] = RWT[:,2].data.reshape (DLEN, Shape[1])
    WDLAMSCDF[:] = ma.array (WDL, fill_value=-32767.).data.reshape (DLEN, Shape[1])
    WSLAMSCDF[:] = ma.array (WSL, fill_value=-32767.).data.reshape (DLEN, Shape[1])
    WILAMSCDF[:] = GW[:,2].data.reshape (DLEN, Shape[1])
    ATTACKLCDF[:] = Attack[:].data.reshape (DLEN, Shape[1])
    SSLIPLCDF[:] = Sslip[:].data.reshape (DLEN, Shape[1])
    

netCDFfile.close () # writes the netCDF file
print 'Elapsed time: ', time.time () - Stime, ' s'
print 'Reached end of routine at ', time.asctime(time.localtime(time.time()))

# <headingcell level=2>

# Find how the results differ from the starting values

# <codecell>

# now get standard deviations of the differences between the radome-based and LAMS-based wind components:
UDiff = RWT[:,0] - RWR[:,0]
VDiff = RWT[:,1] - RWR[:,1]
WDiff = RWT[:,2] - RWR[:,2]
# not sure if the folowing protection against NAN is still needed with masked arrays... 
print (' u standard deviation is ', np.std(UDiff[np.isnan(UDiff) ==  False]))    # must remove missing values 
print (' v standard deviation is ', np.std(VDiff[np.isnan(VDiff) ==  False]))    # to avoid FP exception
print (' w standard deviation is ', np.std(WDiff[np.isnan(WDiff) ==  False]))
#i = np.unravel_index (UD.argmax (), UD.shape) # find the maximum value (or replace with argmin() for min)
#
# calculate RMS for period of turns, 

# <headingcell level=2>

# The following cells are temporary work space and can be deleted or changed

# <rawcell>

# plot (VR, label='VR-radome')
# plot (RWT[:,1], label='VR-LAMS')
# legend ()
# ylim([-5.,5.])
# show ()

# <rawcell>

# RG = range(124500,154500)
# WSL[4315+125000:4415+125000] = ma.masked
# WSL[9050+125000:9150+125000] = ma.masked
# WSL[19130+125000:19230+125000] = ma.masked
# WSL[24055+125000:24155+125000] = ma.masked
# clf()
# plot(WSL[RG])
# print np.mean(WSL[RG])
# print np.std(WSL[RG])
# #show()

# <rawcell>

# # to use, change from 'raw text' to 'code', or vv to skip
# Fig=figure()
# Fig.subplots_adjust(hspace=0.35)
# Panel1 = Fig.add_subplot (211)
# RG = arange (137125,137625)
# plot (ROLL[RG], label='ROLL')
# plot (CROLL[RG], label='CROLL')
# legend ()
# RG2 = arange (137126,137626)
# Panel2=Fig.add_subplot (212)
# plot (ROLL[RG]-CROLL[RG], label='no adjustment')
# ylabel('ROLL-CROLL')
# plot (ROLL[RG]-CROLL[RG2], color='r', label='CROLL+1')
# ylabel('ROLL-CROLL')
# legend(loc=4)
# show()

# <rawcell>

# # find the delay giving the best correlation:
# start = 124500
# end = start + 30000
# RG = arange(start, end)
# for i in range (-25,25):
#     RG2 = arange(start+i, end+i)
#     correlation = np.corrcoef (ROLL[RG], CROLL[RG2])
#     print i,correlation[0,1]

# <codecell>


