import pandas as pd 
import numpy as np 
from scipy import stats
from tqdm.notebook import tqdm
from scipy import fftpack
from datetime import datetime 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import matplotlib 
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler 
plt.style.use("seaborn")
rcParams['figure.dpi']= 300
rcParams['axes.labelsize']=5
rcParams['axes.labelpad']=2
rcParams['axes.titlepad']=3
rcParams['axes.titlesize']=5
rcParams['axes.xmargin']=0
rcParams['axes.ymargin']=0
rcParams['xtick.labelsize']=4
rcParams['ytick.labelsize']=4
rcParams['grid.linewidth']=0.5
rcParams['legend.fontsize']=4
rcParams['lines.linewidth']=0.5
rcParams['xtick.major.pad']=2
rcParams['xtick.minor.pad']=2
rcParams['ytick.major.pad']=2
rcParams['ytick.minor.pad']=2
rcParams['xtick.color']='grey'
rcParams['ytick.color']='grey'
rcParams['figure.titlesize']='medium'
rcParams['axes.prop_cycle']=cycler('color', ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'])







"""
PUPIL DATA PROCESSING STUFF
"""





"""
Calculates time is seconds from a string "HH:MM:SS:msmsmsmsms....."

Parameters: 
•strTime: time reading of form "HH:MM:SS:msmsmsmsms....."

Returns 
• time in seconds (float) 
"""
def scalarTime(strTime): # strTime of form 'HH:MM:SS:msmsmsmsms.....'
	hours = int(strTime[0:2])*60*60
	minutes = int(strTime[3:5])*60
	seconds = int(strTime[6:8])
	milliseconds = float('0.'+strTime[9:])
	return hours+minutes+seconds+milliseconds #time in s





"""
Gets average time step between readings in a (possibly non-uniform) time series array

Parameters:
•timeArray: the time series array 

Returns:
• dt: the average time step for this array 
"""
def get_dt(timeArray):
	return np.mean((np.roll(timeArray,-1) - timeArray)[1:-1])





"""
Loads pupil data. Defaults to EyeLink which has name format "name_pupillometry.csv" else tries pupilLabs "name_pupillometryPL"
If EyeLink, then data file has Bonsai timesyncs saved within it, these are used to sync time with Bonsai machine.
If PupilLabs, timesync data saved in separate csv "name_timesync.csv". These are loaded and used to sync time.

Parameters: 
• name: name of participant. Will search for datafile "./Data/<name>_pupillometry[PL].csv"

Returns
•pupilDiams: array of pupil diameters, raw
•times: time series array one-to-one correspondence with each pupilDiameter (corresponding to time on the non-pupillometry machine)
•dt: average time gap between readings
"""
def loadAndSyncPupilData(name,defaultMachine='EL',eye='right'): #EL
	loadComplete = False
	if defaultMachine == 'EL':
		fileName = './Data/'+name+'_pupillometry.csv'
		try:
			open(fileName,encoding="ISO-8859-1")
			pupilDiams = [] #create arrays
			times = []

			#open and read file line by line 
			with open(fileName,encoding="ISO-8859-1") as f: 
				lines = f.readlines()
				lagTime = 0 #keeps a running estimate of the lag time between pupillometry computer and the bonsai one 
				errorCount = 0
				try: del firstSyncSignalIdx 
				except NameError: pass 
				print("Loading and synchronising pupillometry data (EyeLink): ", end = "")
				for i in tqdm(np.arange(len(lines))):            
					words = lines[i].split()
					if len(words) <= 2:
						continue
					isSyncLine = (words[0][:3] == 'MSG' and words[2][:6] == 'SYNCSS') #test if it's a time sync line
					isDataLine = (words[0].isnumeric() == True) #test if it's a data line 
					if not isSyncLine and not isDataLine: #else pass 
						continue
					try: 
						if isSyncLine: #then update the lag time 
							pupillometryTime = float(words[1]) / 1000
							syncTime = scalarTime(words[2][6:])
							lagTime = syncTime - pupillometryTime # update lag time 
							try: firstSyncSignalIdx #save index so we can chop all data before this first sync
							except NameError: 
								firstSyncSignalIdx = i
							continue
						elif isDataLine: #else scrape time, convert to computer time and scrape pupil diam and save these
							pupillometryTime = float(words[0]) / 1000
							eventTime = lagTime + pupillometryTime #in reference frame of computer, not pupillometry machine
							pupilDiam = float(words[-2])
							pupilDiams.append(pupilDiam)
							times.append(eventTime)
					except: 
						errorCount += 1
						pass 
				print("%g errors in total" %errorCount)

			#converts to arrays and deletes data before firstSyncSignal since true time is unknown
			pupilDiams = np.array(pupilDiams)[firstSyncSignalIdx:] 
			times = np.array(times)[firstSyncSignalIdx:]
			dt = np.mean((np.roll(times,-1) - times)[1:-1]); #print('dt = %.4fs' %dt)
			#print("Percentage data missing: %.2f%%" %(100*len(np.where(pupilDiams == 0)[0])/len(pupilDiams)))
			loadComplete=True

		except FileNotFoundError: 
			print("No EyeLink (default) datafile found, trying PupilLabs")

	#note this could be much better, the model reports a 'confidence' which we could employ
	if ((defaultMachine == 'PL') or (loadComplete == False)):  #try PupilLabs data file 
		fileName = './Data/'+name+'_pupillometryPL.csv'
		try: open(fileName)
		except FileNotFoundError: 
			print("No PupilLabs (fall back) data file found")
		pupilDiams_pl = []
		times_pl = []
		if eye=='right': eyeID='0'
		elif eye=='left': eyeID='1'

		#loads time sync data files and returns arrays of simultaneous pupilLabTimestamps and computerTimestamps for syncing
		computerTimestamp, pupilLabTimestamp = extractSyncTimes(name)

		with open(fileName) as f: 
			lines = f.readlines()
			print("Loading and synchronising pupillometry data (PupilLabs)")
			for i in tqdm(np.arange(len(lines))): 
				data = lines[i].split(",")
				if i == 0: 
					continue
				if ((data[2] != eyeID) or (data[7][:5] != 'pye3d')): 
					pass
				else: 
					raw_time = float(data[0])
					idx = np.argmin(np.abs(pupilLabTimestamp - raw_time))
					lag = computerTimestamp[idx] - pupilLabTimestamp[idx]
					time = raw_time + lag
					times_pl.append(time)
					pupilDiam = float(data[6])
					pupilDiams_pl.append(pupilDiam)
		#convert to array 
		pupilDiams = np.array(pupilDiams_pl)
		times = np.array(times_pl)
		dt = np.mean((np.roll(times,-1) - times)[1:-1]); #print('dt = %.4fs' %dt)
		#print("Percentage data missing : %.2f %%" %(100*len(np.where(pupilDiams == 0)[0])/len(pupilDiams)))  
		loadComplete = True

	return pupilDiams, times





"""
Loads timesync file made when PupilLabs is recorded.

Parameters: 
• name: the name of the participant. Will search for data "./Data/<name>_timesync.csv"

Returns: 
• array of computer timestamps
• array of pupilLabs timestamps

Returns two arrays: one for timestamps from computer (presumably Bonsai or otherwise is running here)
one for simultaneous pupilLabs timestamps
"""
def extractSyncTimes(name):
	computerTimestamp = []
	pupilLabTimestamp = []

	with open('./Data/'+name+'_timesync.csv') as f: 
		lines = f.readlines()
		for i in range(len(lines)): 
			if i == 0: pass
			else: 
				times = lines[i].split(',')
				computerTimestamp.append(scalarTime(times[0]))
				pupilLabTimestamp.append(float(times[1][:-2])) 
	computerTimestamp = np.array(computerTimestamp)
	pupilLabTimestamp = np.array(pupilLabTimestamp)
	return computerTimestamp, pupilLabTimestamp 





"""
Uniformly samples data array to a new constant frequency.

Params: 
• dataArray - the array to be uniformly sampled
• timeArray - the time series for dataArray, one-to-one correspondence with dataArray
• new_dt - new precise time separation between points 
• aligntimes - if defined, dataArray is aligned presicely to the times in this array 

Returns:
• uniformly sampled dataArray
• correspinding time series array 
• new precise dt  

It  interpolates the time series to be uniformly spaced in time.
For this to be stable, however, it's better to increase the frequency, not decrease. 
Hence, ideally, new_dt < dt
Basically, PupilLabs doesn't have a constant FPS which messes with later filtering etc. so this use linear interpolation to make it constant.
"""
def uniformSample(dataArray, timeArray, new_dt = None, aligntimes = None): 

	dt = get_dt(timeArray)

	if aligntimes is None: 
		print("Uniformly sampling  data to %gHz (current frequency ~%gHz)" %(int(1/new_dt),int(1/dt)))
	else: 
		print("Sampling to match given time series")

	newTimeArray = []
	newDataArray = []
	#instead of interpolation to a new constant dt_new it can instead interpolate to exactly match a currently existing timeseries array
	#useful if you want compare two arrays recorded at the same time but with different precise FPSs
	if aligntimes is None: 
		t = timeArray[0] 
		datum = dataArray[0] 
		while True:
			newTimeArray.append(t)
			newDataArray.append(datum)
			t += new_dt
			if t > timeArray[-1]:
				break
			else: #interpolate 
				idx_r = np.searchsorted(timeArray,t)
				idx_l = idx_r-1
				delta = timeArray[idx_r] - timeArray[idx_l]
				datum = ((t-timeArray[idx_l])/delta)*dataArray[idx_l] + ((timeArray[idx_r] - t)/delta)*dataArray[idx_r]

	elif aligntimes is not None: 
		for t in aligntimes: 
			if t > timeArray[-1] or t < timeArray[0]:
				pass
			else: 
				idx_r = np.searchsorted(timeArray,t)
				idx_l = idx_r-1
				delta = timeArray[idx_r] - timeArray[idx_l]
				datum = ((t-timeArray[idx_l])/delta)*dataArray[idx_l] + ((timeArray[idx_r] - t)/delta)*dataArray[idx_r] #do linear interpolation
				newTimeArray.append(t)
				newDataArray.append(datum)
	new_dt = np.mean((np.roll(newTimeArray,-1) - newTimeArray)[1:-1]); #print('dt = %.4fs' %new_dt)
	return np.array(newDataArray), np.array(newTimeArray)






"""
Removes outliers. Following Leys et al 2013 we use median absolute deviation, not std, to identify outliers.
Upper and lower outliers for absolute speed speed are excluded
Only lower outliers and zero values for data are excluded
Outliers in the data are set to zero. 

Parameters: 
•dataArray: array from which outliers is to be removed
•timeArray: time series array for the idata array 
•n_speed: threshold for data speed, d data / dt (equivalent to how many +- stds to accept)
•n_size: threshold for data (equivalent to how many +- stds to accept)
•plotHist: if True, plots data histograms showing thresholds

Returns: 
•pupilDiams with outliers set to zero
•a boolean array identifying outliers (True) and inliers (False)
"""
def removeOutliers(dataArray,timeArray,n_speed=2.5,n_size=2.5, plotHist=False): #following Leys et al 2013 
	print("Removing speed outliers", end="")
	size = dataArray
	absSpeed = np.zeros(len(dataArray))
	data = dataArray
	for i in range(len(dataArray)):
		absSpeed[i]=max(np.abs((data[i]-data[i-1])/(timeArray[i]-timeArray[i-1])),np.abs((data[(i+1)%len(data)]-data[i])/(timeArray[(i+1)%len(data)]-timeArray[i])))

	MAD_speed = np.median(np.abs(absSpeed - np.median(absSpeed)))
	MAD_size = np.median(np.abs(size - np.median(size)))
	threshold_speed_low = np.median(absSpeed) - n_speed*MAD_speed
	threshold_size_low = np.median(size) - n_size*MAD_size
	threshold_speed_high = np.median(absSpeed) + n_speed*MAD_speed
	threshold_size_high = np.median(size) + n_size*MAD_size
	
	data = data * (absSpeed<threshold_speed_high) * (absSpeed>threshold_speed_low)
	print(" (%.2f%%) " %(100*(1-np.sum((absSpeed<threshold_speed_high) * (absSpeed>threshold_speed_low))/len(data))),end="")
	
	data = data * (size>threshold_size_low) * (size!=0)#only take away low sizes and zero values
	print("and size lowliers/zero (%.2f%%) " %(100*(1-np.sum((size>threshold_size_low)*((size!=0)))/len(data))), end="")
	
	print(" (additional %.2f%% removed vs raw)" %(100*(np.sum(data == 0) - np.sum(dataArray==0))/len(data)))
	
	if plotHist == True: #plots histograms and thresholds
		fig, ax = plt.subplots(1,2)
		ax[0].hist(np.log(absSpeed),bins=30)
		ax[1].hist(size,bins=30)
		ax[0].axvline(x=np.median(absSpeed),c='k')
		ax[0].axvline(x=threshold_speed_low,c='k')
		ax[0].axvline(x=threshold_speed_high,c='k')
		ax[1].axvline(x=np.median(size),c='k')
		ax[1].axvline(x=threshold_size_low,c='k')
		ax[1].axvline(x=threshold_size_high,c='k')
		ax[0].set_title("data")
		ax[1].set_title("d data / dt ")
	isOutlier = np.invert((absSpeed<threshold_speed_high) * (absSpeed>threshold_speed_low) * (dataArray>threshold_size_low) * (dataArray!=0))	
	
	return data, isOutlier





"""
Downsample lowers the frequency of the data by bin averaging. 
Often data may comes off machine at high frequencies which is prohibitive for filtering. This function, therefore not crucial but is helpful and saves you time by down sampling it to a lower frequency. Bin averaging is lousy compared to the filtering done later, but probably okay as long as the frequency (Hz) we bin average to is still much higher than the eventual frequency band. Currently set to 50Hz (bin = 0.02s) this is still far higher than pupil responses so averaging over bins this size shouldn't hurt. If a value in the data array is an outlier this is masked and will not contribute to the bin average. 

Parameters: 
• dataArray: the data array ot be down sampled
• timeArray: the time series array corresponding to the data array
• Hz: frequency to downsample to 
• isOutlier: a boolean array same size as dataArray (True if the value in data array is an outlier/zero, otherwise False) Assumes no outliers if not provided

Returns:
• down sampled dataArray
• correspondingly downSampled timeArray
• downSampled isOutlier array. Now this is no longer a boolean array. It is a float array representing proportion of data in this bin which was an outlier. 
"""
def downsample(dataArray, timeArray, Hz=50, isOutlier=None): 

	dt = get_dt(timeArray)

	downsampledDataArray = []
	downsampledTimes = []
	downsampledIsOutlier = []

	if isOutlier is None: 
		isOutlier = np.zeros(len(dataArray), dtype=bool) #assumes none are outliers otherwise

	binSize = int((1/dt) / Hz) #current no. samples in one second / 50

	for i in range(np.int(np.floor(len(dataArray)/binSize))):
		outlierMask = np.where(isOutlier[i*binSize:(i+1)*binSize] == False)[0] #
		if len(outlierMask) == 0: 
			downsampledDataArray.append(0)
			downsampledIsOutlier.append(1)
		else:
			downsampledDataArray.append(np.mean(dataArray[i*binSize:(i+1)*binSize][outlierMask]))
			downsampledIsOutlier.append(1-len(outlierMask)/binSize)
		downsampledTimes.append(timeArray[int((i+0.5)*binSize)])
	downsampledDataArray = np.array(downsampledDataArray)
	downsampledTimes =  np.array(downsampledTimes)
	downsampledIsOutlier = np.array(downsampledIsOutlier)

	dataArray = downsampledDataArray
	isOutlier = downsampledIsOutlier
	timeArray = downsampledTimes


	return dataArray, timeArray, isOutlier





"""
Performs interpolation to remove zero-values from the data
Wherever a range of the data values are zero (the zero-range) these are replaced with linear interpolation between the interpolation points which are +- gapExtension seconds before and after the zero-range, assuming these are values are non-zero (moving further out if they aren't).
The values between the zero-range and the interpolation points  are themselves also replaced since the blink or whatever maybe cause a smooth drop to zero we also want to remove

Parameters: 
• dataArray: the array to be interpolated
• timeArray: corresponding time series array for the data array 
• gapExtension: how many seconds before/after the gap to extend when doing linear interpolation

Returns: 
• interpolatedDataArray
• isInterpolated: a boolean array same lengths as interpolatedDataArray, True if the value at this point is interpolated, not real


"""
def interpolateArray(dataArray, timeArray, gapExtension = 0.2):
	
	interpolatedDataArray = dataArray.copy()
	jump_dist = int(gapExtension / get_dt(timeArray)) #interpolates between gapExtension seconds before and after the points where it they fell to zero

	print("Interpolating missing values: ", end="")
	isInterpolated = np.zeros(len(dataArray),dtype=bool)
	
	i = 0
	while True:


		if i >= len(dataArray): break 

		if dataArray[i] != 0: #the value exists and there is no problem
			interpolatedDataArray[i] = dataArray[i] 
			i += 1

		elif dataArray[i] == 0:#do some interpolation

			k = jump_dist
			while True: 
				if i-k < 0: #edge case where we fall off array 
					start, startidx = np.mean(dataArray), 0
					break
				elif i-k >= 0:
					if dataArray[i-k] == 0:
						k += 1 #keep extending till you get non-zero val
					elif dataArray[i-k] != 0:
						start, startidx = dataArray[i-k], i-k+1
						break

			j = i 
			while True: 
				while True: #find 'end' of blink
					if j >= len(dataArray):
						j = j-1
						break
					if dataArray[j] == 0:
						j += 1
					elif dataArray[j] !=0:
						j = j-1
						break

				if j+k >= len(dataArray): #edge case where we fall off array 
					end, endidx = np.mean(dataArray), len(dataArray)
					break
				elif j+k < len(dataArray):
					if dataArray[j+k] == 0:
						k += 1 #keep extending till you get non-zero val
					elif dataArray[j+k] != 0:
						end, endidx = dataArray[j+k], j+k-1
						break
			interpolatedDataArray[startidx:endidx] = np.linspace(start,end,endidx-startidx)
			isInterpolated[startidx:endidx] = np.ones(endidx-startidx,dtype=bool)

			i=endidx+1
	print("%.2f%% of values are now interpolated" %(100*np.sum(isInterpolated)/len(isInterpolated)))
	return interpolatedDataArray, isInterpolated





"""
Filters the signal. A smooth logistic filter in frequecy space is fourier transformed into time domain
This is then convolved over the pupil diameter time series. 
Can choose width and whether filter is high or low pass.

Parameters:
• dataArray: the array to be filtered 
• timeArray: time series array for the data array 
• cutoff_freq: point in frequency space where logistic sigmoid = 0.5
• cutoff_width: width of logistic sigmoid
• highpass: If True it does a highpass filter (removing drift), otherwise it does a lowpass filter (removing noise)
• plotStuff: If True, shows some plots 

Returns: 
• the filtered data array 
"""
def frequencyFilter(dataArray,timeArray,cutoff_freq,cutoff_width,highpass=False,plotStuff=False):

	dt = get_dt(timeArray)
	
	if highpass == True: name=('Highpass','below')
	else: name = ('Lowpass','above')

	print("%s filtering out frequencies %s %.2f +- %.2fHz" %(name[0], name[1], cutoff_freq, cutoff_width))
	#derive the filter
	f = fftpack.fftfreq(dataArray.size,dt)
	fil = 1/(1 + np.exp(-(4/cutoff_width)*(np.abs(f) - cutoff_freq))) #logistic curve (abs for positive and negative frequencies)
	if highpass == False: fil = 1 - fil 
	plt.figure(0)
	#plt.plot(f,fil)
	#plt.title("Filter")
	#plt.xlabel("Frequency / Hz")

	# inv fourier transform the filter 
	fil_ifft = fftpack.ifft(fil)
	t = fftpack.fftfreq(f.size,f[1]-f[0])
	fil_ifft = np.real(fil_ifft[t.argsort()])
	t = t[t.argsort()]


	#convolve this with dataArray, then crop 
	filtered_dataArray = np.convolve(dataArray,fil_ifft)
	if dataArray.size%2 == 1: 
		filtered_dataArray = filtered_dataArray[int(np.floor(dataArray.size/2)):-int(np.floor(dataArray.size/2))]
	elif dataArray.size%2 == 0: 
		filtered_dataArray = (0.5*(filtered_dataArray + np.roll(filtered_dataArray,1)))[int(np.floor(dataArray.size/2)):-int(np.floor(dataArray.size/2))+1]        
	if plotStuff == True: 
		fig, ax = plt.subplots(1,2, figsize=(4,2))
		ax[0].plot(timeArray,filtered_dataArray)
		ax[0].set_title("Filtered Signal")
		ax[1].plot(timeArray,dataArray)
		ax[1].set_title("Original Signal")
		ax[0].set_xlabel("Time / s")
		ax[1].set_xlabel("Time / s")

	return filtered_dataArray





"""
z-scores data Array 
norm range allows you to exclude early and late times from zscoring 

Parameters: 
• dataArray: to be zscored
• timeArray: time series array for the data array 
• normrange: in seconds [tstart, tend], which range to use for finding normalisation parameters (can be used to exclude early or late data if required)

Returns: 
• the z scored data array 
"""
def zScore(dataArray, timeArray=None, normrange=None):
	start_idx, end_idx = 0, -1
	if ((timeArray is not None) and (normrange is not None)):
		start_idx, end_idx = np.argmin(np.abs(timeArray-normrange[0])), np.argmin(np.abs(timeArray-normrange[1]))
	print("z scoring")
	zscoreDataArray = (dataArray - np.mean(dataArray[start_idx:end_idx]))/np.std(dataArray[start_idx:end_idx])
	return zscoreDataArray





"""
Plots two arrays and a histogram showing full timeseries and zoomed in time series of pupil diameters
Data is coloured according to whehter it is an outlier (pink), not an outlier but interpolated (yellow) or neither (green)

Paramters: 
• dataArray: the data array to be plotted
• timeArray: corresponding time series array
• title: suptitle for the figure 
• zoomRange: one of the subplots zooms in on the data in this region (seconds)
• hist: False if you don't want a side histogram
• ymin: bottom limit of yrange. Make "-ymax" if you want it to be the negative of ymax
• ymax: top limit of yrange. If undefined chooses mean + 3std
• isInterpolated: boolean array, True where a point is interpolated
• isOutlier: float array between 0 and 1 showing proportion of data points in this bin (recall each data point is the bin average of many) were outliers

Returns: 
• fig (this is also plotted)
• ax  
"""
def plotData(dataArray, timeArray, title=None, zoomRange = [0,60], saveName = None, hist=True, ymin=0, ymax=None, isInterpolated=None, isOutlier=None):
	
	dt = get_dt(timeArray)
	
	if (isInterpolated is not None):
		color = np.array(['C0']*len(dataArray))
		if (isInterpolated is not None):
			color[np.where(isInterpolated == True)[0]] = np.array(['C5']*len(np.where(isInterpolated == True)[0]))
		if (isOutlier is not None):
			color[np.where(isOutlier > 0.9)[0]] = np.array(['C1']*len(np.where(isOutlier > 0.9)[0]))
	if ymax is None: 
		ymax = np.mean(dataArray) + 3*np.std(dataArray)
	if ymin=='-ymax': ymin = -ymax
	fig, ax = plt.subplots(1,2,figsize=(4,2),sharey=True)
	fig.suptitle(title)
	ax[0].scatter((timeArray[int(zoomRange[0]/dt):int(zoomRange[1]/dt)] - timeArray[0]),dataArray[int(zoomRange[0]/dt):int(zoomRange[1]/dt)],c=color[int(zoomRange[0]/dt):int(zoomRange[1]/dt)],s=0.1)
	ax[0].set_ylim([ymin,ymax])
	ax[0].set_ylabel('Pupil diameter')
	ax[0].set_xlabel('Time from start of recording / s')
	ax[0].set_title('Raw data (%gs)'%(zoomRange[1]-zoomRange[0]))

	ax[1].scatter((timeArray - timeArray[0]),dataArray,c=color,s=0.1)
	ax[1].set_ylim([ymin,ymax])
	ax[1].set_ylabel('Pupil diameter')
	ax[1].set_xlabel('Time from start of recording / s')
	ax[1].set_title('Raw data (full)')

	if hist==True: 
		divider = make_axes_locatable(ax[1])
		axHisty = divider.append_axes("right", 0.2, pad=0.02, sharey=ax[1])
		axHisty.yaxis.set_tick_params(labelleft=False)
		axHisty.xaxis.set_tick_params(labelbottom=False)
		binwidth = (ymax-ymin)/80
		ymax = np.max(np.abs(dataArray))
		lim = (int(ymax/binwidth) + 1)*binwidth
		bins = np.arange(-lim, lim + binwidth, binwidth)
		axHisty.hist(dataArray, bins=bins, orientation='horizontal',color='C2',alpha=0.8)

	if saveName is not None: 
		plt.savefig('./figures/' + saveName + '.pdf',tightlayout=True, transparent=False,dpi=100)

	return fig, ax







"""
Groups together all the functions for loading pupil and trial data for a given participant.
The functions in this class basically just interface with the functions defined above. 

Parameters:
• name: the name of the participants. Functions will search for data like "./Data/<name>_... .csv"

Returns: 
Usually nothing, see individual functions. 
"""
class pupilDataClass():

	def __init__(self,name):
		self.name = name

	def loadData(self, defaultMachine='EL',eye='right'):
		self.rawPupilDiams, self.rawTimes = loadAndSyncPupilData(self.name, defaultMachine=defaultMachine,eye=eye)

	def uniformSample(self):
		self.pupilDiams, self.times = uniformSample(self.rawPupilDiams, self.rawTimes, new_dt = 0.5*get_dt(self.rawTimes))

	def removeOutliers(self,n_speed=2.5,n_size=2.5):
		self.pupilDiams, self.isOutlier = removeOutliers(self.pupilDiams,self.times,n_speed=n_speed,n_size=n_size)

	def downSample(self):
		self.pupilDiams, self.times, self.isOutlier = downsample(self.pupilDiams, self.times, isOutlier=self.isOutlier)

	def interpolate(self, gapExtension=0.1):
		self.pupilDiams, self.isInterpolated = interpolateArray(self.pupilDiams, self.times, gapExtension=gapExtension)
		print("Data: %.2f%%  missing, %.2f%% outliers, %.2f%% interpolated" %(100*np.mean((self.rawPupilDiams == 0)), 100*np.mean(self.isOutlier),100*np.mean(self.isInterpolated)))

	def frequencyFilter(self, lowF=0.1, lowFwidth=0.01, highF=4, highFwidth=0.5):
		self.pupilDiams = frequencyFilter(self.pupilDiams,self.times,cutoff_freq=highF, cutoff_width=highFwidth, highpass=False)
		self.pupilDiams = frequencyFilter(self.pupilDiams,self.times,cutoff_freq=lowF, cutoff_width=lowFwidth, highpass=True)

	def zScore(self):
		self.pupilDiams = zScore(self.pupilDiams, normrange=[60,self.times[-1]-60]) 

	def plot(self, title=None, zoomRange = [0,60], saveName = None, hist=True, ymin='-ymax', ymax=None):
		plotData(self.pupilDiams, self.times, title=title, zoomRange = zoomRange, saveName = saveName, ymin=ymin, ymax=ymax, isInterpolated=self.isInterpolated, isOutlier=self.isOutlier)

	def loadAndProcessTrialData(self):
		self.trialData, self.times = loadAndProcessTrialData(self.name, self.times)









"""
TRIAL DATA PROCESSING AND PLOTTING
"""






"""
Loads Bonsai data file and extracts/calculates important data 
Find first trial and aligns pupil times to this (t=0 is now start of first trial). Extracts and stores in a dictionary (different entry for each trial) and series of 'events times' within or 'facts'  about a trial
eg:  times include •Trials_Start, Trial_End, Tone_Time, gapStart...
	 facts include: • whether tone was heard, ToneHeard, whether tone was a violation Pattern_Type, what ype of violation PatternID...

Parameters: 
	• name: participant name, will search trial data file like "./Data/<name>_trial.csv"
	• pupilTimes: time series array corresponding to the pupil diameter data. 
	
Returns: 
	•dictionary with structure: dict[trialID][trialFactOrTimestamp]
	•pupilTimes - since these have been realigned to start of first trial 
"""
def loadAndProcessTrialData(name, pupilTimes): #pupil times also passed as these are shifted relative to start of first trial
	trials = {} #index of event

	with open('Data/'+name+'_trial.csv') as f:
		lines = f.readlines()
		Ntrials = len(lines)-1
		columns = lines[0].split(','); #print(columns)
		columns[-1]=columns[-1][:-1] #remove the \n
		print("%s: %g trials" %(name,Ntrials))
		print("Loading and cleaning trial data")
		for i in tqdm(range(Ntrials)): 




			#MOST RECENT VERSION (conditions for previous versions writen below)
			words = lines[i+1].split(',')

			if i == 0: 
				trialZeroTime = scalarTime(words[columns.index("Trial_Start")]) 
				pupilTimes = pupilTimes - trialZeroTime

			#event index 
			trials[i] = {}

			#trial start 
			try: trialStart = scalarTime(words[columns.index("Trial_Start")]) - trialZeroTime #when they poked in, same as white noise start
			except: trialStart = 'na'
			trials[i]['trialStart'] = trialStart 
			trials[i]['whiteNoiseStart'] = trialStart 

			#trial end 
			try: trialEnd = scalarTime(words[columns.index("Trial_End")]) - trialZeroTime #either by poke out or forced
			except: trialEnd = 'na'
			trials[i]['trialEnd'] = trialEnd

			#pattern type 
			try: patternType = int(words[columns.index("Pattern_Type")]) 
			except: patternType = 'na'
			trials[i]['patternType'] = patternType

			#pattern tones 
			try: 
				tones = [int(i) for i in words[columns.index("PatternID")].split(';')]
				tonesList = ['A']
				alphabet = ['A','B','C','D']
				for j in range(3):
					tonesList.append(alphabet[int((tones[j+1] - tones[0])/2)])
			except: tonesList = 'na'
			trials[i]['tonesList'] = tonesList

			#trial correct
			try:
				if int(words[columns.index("Trial_Outcome")]) == 1: trialCorrect = 'correct'
				elif int(words[columns.index("Trial_Outcome")]) == -1: trialCorrect = 'violation' #withdrew early 
				elif int(words[columns.index("Trial_Outcome")]) == 0: trialCorrect = 'incorrect' #missed response window
			except: trialCorrect = 'na'
			trials[i]['trialCorrect'] = trialCorrect

			#tone after gap 
			try: 
				if int(words[columns.index("Tone_Position")]) == 0: toneAfterGap = False
				elif int(words[columns.index("Tone_Position")]) == 1: toneAfterGap = True
			except: toneAfterGap = 'na'
			trials[i]['toneAfterGap'] = toneAfterGap

			##tone start
			try:
				try: ind = columns.index("ToneTime")
				except ValueError: ind = columns.index("Tone_Time")
				toneStart = scalarTime(words[ind]) - trialZeroTime
			except: trialStart = 'na'
			trials[i]['toneStart'] = toneStart

			#tone heard 
			try: toneHeard = (int(np.floor(scalarTime(words[ind]))) != 0)
			except: toneHeard = 'na'
			trials[i]['toneHeard'] = toneHeard

			#gap starts
			try:
				gapStart = trialStart + float(words[columns.index("Stim1_Duration")])
			except: gapStart = 'na'
			trials[i]['gapStart'] = gapStart

			#csv saved
			try: csvSaved = scalarTime(words[columns.index("Time")]) - trialZeroTime
			except: csvSaved = 'na'
			trials[i]['csvSaved'] = csvSaved

			#white Cross Appears
			try: whiteCrossAppears = scalarTime(words[columns.index("WhiteCross_Time")]) - trialZeroTime
			except: whiteCrossAppears = 'na'
			trials[i]['whiteCrossAppears'] = whiteCrossAppears  #reaction time

			#rewardCrossAppears 
			try: RewardCross_Time = scalarTime(words[columns.index("RewardCross_Time")]) - trialZeroTime
			except: RewardCross_Time = 'na'
			trials[i]['RewardCross_Time'] = RewardCross_Time

			#reaction time 
			try: reactionTime = trialEnd - gapStart
			except: reactionTime = 'na'
			trials[i]['reactionTime'] = reactionTime 



			#code specific to old protocols or specific participants (above should be most up to date)
			if name in ['shanice']: #do specifics for shanice
				#subtract 1.5s from the toneTime (was saving time of start of 4th tone)
				trials[i]['toneStart'] -=1.5


			if name in ['morio','chris']: #do specifics for morio and chris 
				#infer toneStart and toneHeard 
				try:
					toneStart = trialStart + float(words[columns.index("PreTone_Duration")])
					if toneAfterGap == True: 
						toneStart += float(words[columns.index("Stim1_Duration")]) + 4*float(words[columns.index("Tone_Duration")])
				except: toneStart == 'na'
				trials[i]['toneStart'] = toneStart

				try: toneHeard = (trialEnd-1.5 > toneStart)
				except: toneHeard = 'na'
				trials[i]['toneHeard'] = toneHeard

				#exclude trials where thye failed and tone was fter gap 
				if (trialCorrect == 'incorrect' and toneAfterGap == True):
					trials[i]['toneHeard'] = False 

				#if tone comes before gap subtract 1.5 from reaction time
				if toneAfterGap == False:
					trials[i]['reactionTime'] -= 1.5

			if name in ['athena', 'elena']: #do specifics for elena and athena 
				#patternID
				try:
					firstLetter = words[columns.index("PatternID")][0]
					if  firstLetter == '1': patternType = 0
					elif firstLetter == '2': patternType = 1
				except: patternType = 'na'
				trials[i]['patternType'] = patternType

	return trials, pupilTimes







"""
Given an alignEvent (e.g. ToneStart) it goes through the data dict (of form returned by loadAndProcessTrialData) 
and for every trial it aligns pupilDiameters to the start of this event (+- some range)
It then goes through a long list of conditions. If the condition number is in the conditionList 
the trial is checked against that condition and if it fails the trial is excluded.This could be to, e.g., exclude first 50 trials or exclude trials which are heavily interpolated or filter only trials of a certain violation type

Parameters: 
• data: the dataClass for a given participant (instance of pupilDataClass, above)
• alignEvent: str, which event in the trial are we chopping and aligning the pupil data to
• conditionsList: list of integers, corresponding to which conditions you want ot trigger
• tstart: how many seconds after (negative for before) the alignEvent do you want to slice/plot
• tend: until how many seconds after the alignEvent do yuo want to slice/plot
Returns:
	• array with apupilDiams for all statifying trials shape (n_valid_trials,len_pupil_range)
	• a time array of the same size, shape (len_pupil_range) starting at tstart and ending at tend for plotting against 
"""
def sliceAndAlign(data, alignEvent = 'toneStart', conditionsList=[], tstart = -2, tend = 5): 

	trials, pupilDiams, isOutlier, times, dt = data.trialData, data.pupilDiams, data.isOutlier, data.times, get_dt(data.times)
	alignedTime = np.linspace(tstart,tend,int((tend - tstart)/dt))

	interpolationExclusion = 0
	varianceExclusion = 0
	noPupilDataExclusion = 0
	validTrials = 0
	for i in np.arange(len(trials)):

		verdict = True

		tevent = trials[i][alignEvent]
		if type(tevent) is str:
			startidx=0
			endidx = startidx + int((tend - tstart)/dt)                 
			verdict *= False
		else: 
			startidx = np.argmin(np.abs(times - (tevent+tstart)))
			endidx = startidx + int((tend - tstart)/dt)                 

		if 0 in conditionsList: #activate if you ONLY want the first 20 trials
			if i >= 20:
				verdict *= False

		if 1 in conditionsList: #activate if you want to EXCLUDE the first 20 trials
			if i < 20:
				verdict *= False

		if 2 in conditionsList: #activate if you want to EXCLUDE the first 50 trials
			if i < 50:
				verdict *= False

		if 3 in conditionsList: #activate if you want only correct trials only 
			if trials[i]['trialCorrect'] != 'correct':
				verdict *= False

		if 4 in conditionsList: #activate if you want normal (non-violation) patterns only
			if trials[i]['patternType'] != 0:
				verdict *= False

		if 5 in conditionsList: #activate if you want violation patterns only trials only 
			if trials[i]['patternType'] == 0:
				verdict *= False

		if 6 in conditionsList: #look at only patterns of type 1 ABC_
			if trials[i]['patternType'] != 1:
				verdict *= False

		if 7 in conditionsList: #look at only patterns of type 1 AB_D
			if trials[i]['patternType'] != 2:
				verdict *= False

		if 8 in conditionsList: #look at only patterns of type 1 AB__
			if trials[i]['patternType'] != 3:
				verdict *= False

		if 9 in conditionsList: #only trials where tone had a decreasing note e.g. ABDC
			tones = ['A','B','C','D']
			trialTones = trials[i]['tonesList']
			if ((tones.index(trialTones[1]) >= tones.index(trialTones[0])) and 
				(tones.index(trialTones[2]) >= tones.index(trialTones[1])) and
				(tones.index(trialTones[3]) >= tones.index(trialTones[2]))): 
				verdict *= False

		if 10 in conditionsList: #activate if you want only trials when the tone was heard 
			if trials[i]['toneHeard'] == False:
				verdict *= False

		if 12 in conditionsList: #only trials when tone was BEFORE gap
			if trials[i]['toneAfterGap'] == True:
				verdict *= False 

		if 14 in conditionsList: #only trials where tone didn't have a decreasing note e.g. ABDD
			tones = ['A','B','C','D']
			trialTones = trials[i]['tonesList']
			if not ((tones.index(trialTones[1]) >= tones.index(trialTones[0])) and 
				(tones.index(trialTones[2]) >= tones.index(trialTones[1])) and
				(tones.index(trialTones[3]) >= tones.index(trialTones[2]))): 
				verdict *= False

		#Now exclude for non-experimental reasons (e.g. due to high interpolation or variance)
		if verdict == True: 
			validTrials += 1

		if pupilDiams[startidx:endidx].shape != (endidx-startidx,):
			if verdict == True:
				noPupilDataExclusion += 1
				verdict *= False

		if 11 in conditionsList: #activate if you want to EXCLUDE trials where the pupil diameter goes over 4 std:
			if np.max(np.abs(pupilDiams[startidx:endidx])) > 4:
				if verdict == True: 
					varianceExclusion += 1 
				verdict *= False

		if 13 in conditionsList: #exclude trials where over 20% of the pupil data is interpolated
			if np.mean((isOutlier[startidx:endidx]))/(endidx-startidx) >= 0.2:
				if verdict == True: 
					interpolationExclusion += 1 
				verdict *= False


		if verdict==True:
			alignedPupilDiams = pupilDiams[startidx:endidx]
			try: 
				alignedData = np.vstack([alignedData,np.array(alignedPupilDiams)])
			except: 
				alignedData = np.array([np.array(alignedPupilDiams)])

	print("%g valid trials of which %g remain after: \n      %g excluded due to interpolation \n      %g excluded due to high variance \n      %g excluded due to no pupildata in this time range" %(validTrials, len(alignedData),interpolationExclusion,varianceExclusion,noPupilDataExclusion))
	return alignedData, alignedTime







"""
Plots pupil diamters and error bars averaged over all subjects in participant data and averaged over all alike-trials for a given subject.
Parameters: 
• participant data: a dictionary of data for participants (each entry of the dictionary is an instance of the pupilDataClass, containing is essential contains pupil data, times, trial data and more)
• align event: what event each trial are we aligning to 
• tsart and tend for plotting 
• title for graph
• test range: on what time range with significance tests be run
• saveTitle: for saving pngs to file 
• dd: a dictionary of things to plot. each dict contains:
	•color of line
	• conditionsList (passed to sliceAndAlign) this is key as it differentiates each one. e.g. one condition list may allow non-violation trials while the other only violation trials, thus these two cases can be compared on same plot. 
	•range: if ('all') all satifactory trials else if ('early'/'mid'/'late', 15) on the earliest, middlest or latest 15 trials will be kept


"""
def plotAlignedPupilDiams(participantData,  #from particpants
						  alignEvent='toneStart',
						  tstart=-2, tend=5,
						  title='Pupil response',
						  testRange=[0,3],
						  saveTitle=None,
						  dd={
							  '':     {'color':'C2','conditions':[0,4,5],'range':('all'),'plotTrials':True},
							 }):

	now = datetime.strftime(datetime.now(),'%y%m%d%H%M')
	
	fig, ax = plt.subplots(figsize=(3.5,2))

	top, bottom = 0, 0
	for name, details in list(dd.items()):
		print(name)
		for (p,participant) in enumerate(list(participantData.keys())):
			print(participant +": ",end = '')
			d,t = sliceAndAlign(participantData[participant], alignEvent=alignEvent,conditionsList=details['conditions'],tstart=tstart,tend=tend)


			if dd[name]['range'][0] == 'first':
				d = d[:dd[name]['range'][1]]
			elif dd[name]['range'][0] == 'mid':
				d = d[int(len(d)/2)-int(dd[name]['range'][1]/2):int(len(d)/2)+int(dd[name]['range'][1]/2)]
			elif dd[name]['range'][0] == 'last':
				d = d[-dd[name]['range'][1]:]

			if p == 0: 
				dd[name]['d'],dd[name]['t'] = d,t
			else:
				try: dd[name]['d'] = np.append(dd[name]['d'],d,axis=0)
				except ValueError: print("      participant excluded, no compatible trials")
		dd[name]['mean'] = np.mean(dd[name]['d'],axis=0)
		dd[name]['ntrials'] = len(dd[name]['d'])
		dd[name]['ci95'] = 1.96*np.std(d,axis=0)/np.sqrt(dd[name]['ntrials'])

		ax.plot(t, dd[name]['mean'],c=dd[name]['color'],label=r'%s %g trials'%(name,dd[name]['ntrials']))
		ax.fill_between(t,dd[name]['mean']+dd[name]['ci95'],dd[name]['mean']-dd[name]['ci95'],color=dd[name]['color'],alpha=0.2)

		if dd[name]['plotTrials'] == True:
			for i in range(len(d)):
				ax.plot(t,d[i],linewidth=0.2,c=dd[name]['color'],alpha=0.03)

		if np.max(dd[name]['mean']) > top: top = np.max(dd[name]['mean'])
		if np.min(dd[name]['mean']) < bottom: bottom = np.min(dd[name]['mean'])


	if len(dd) >= 2:
		print(r'SIGNIFICANCE TESTING BETWEEN %.2f AND %.2fs:' %(testRange[0],testRange[1]))
		for i in range(len(dd)):
			for j in range(i+1,len(dd)):
				keys = list(dd.keys())
				start, stop = np.argmin(np.abs(dd[keys[i]]['t'] - testRange[0])), np.argmin(np.abs(dd[keys[i]]['t'] - testRange[1]))
				m1, m2 = dd[keys[i]]['mean'][start:stop], dd[keys[j]]['mean'][start:stop]
				std1, std2 = dd[keys[i]]['ci95']/1.96, dd[keys[j]]['ci95']/1.96
				mean = m1 - m2
				std = np.sqrt(std1**2 + std2**2)
				if len(dd) <= 3: ntests = 1000
				else: ntests = 1000
				testResult = funcZeroTest(mean,std,ntests=ntests)
				print("%s vs %s: %.4f" %(keys[i],keys[j],testResult))
	if len(dd) == 2: 
		if testResult <= 0.05: tr = '*'
		if testResult <= 0.01: tr = '**'
		else: tr = 'ns'
		print(ax.get_ylim())
		ax.axhline((bottom-0.5)+0.1*(top + 0.5 - bottom - 0.5),xmin=(testRange[0]-tstart)/(tend-tstart),xmax=(testRange[1]-tstart)/(tend-tstart),c='k',alpha=0.5,linewidth=1.5)
		ax.text(x=np.mean(testRange),y=(bottom-0.5)+0.15*(top + 0.5 - bottom - 0.5),s=tr,fontsize=5)

	if alignEvent == 'toneStart': 
		rect1 = matplotlib.patches.Rectangle((0,-10),0.125,20,linewidth=0,edgecolor='k',facecolor='k',alpha=0.1)
		rect2 = matplotlib.patches.Rectangle((0.25,-10),0.125,20,linewidth=0,edgecolor='k',facecolor='k',alpha=0.1)
		rect3 = matplotlib.patches.Rectangle((0.5,-10),0.125,20,linewidth=0,edgecolor='k',facecolor='k',alpha=0.1)
		rect4 = matplotlib.patches.Rectangle((0.75,-10),0.125,20,linewidth=0,edgecolor='k',facecolor='k',alpha=0.1)
		ax.add_patch(rect1)
		ax.add_patch(rect2)
		ax.add_patch(rect3)
		ax.add_patch(rect4)

	elif alignEvent == 'gapStart':
		rect = matplotlib.patches.Rectangle((0,-10),0.25,20,linewidth=0,edgecolor='k',facecolor='k',alpha=0.1)
		ax.add_patch(rect)

	elif alignEvent == 'whiteCrossAppears':
		ax.axvline(0.2,c='k',alpha=0.2,linestyle="--")

	ax.legend(loc=1)
	ax.set_ylim([bottom-0.5,top+0.5])
	ax.axvline(0,c='k',alpha=0.5)
	ax.set_xlabel('Time / s')
	ax.set_ylabel('Normalised pupil diameter')
	ax.set_title(title)

	if saveTitle is not None: 
		plt.savefig(f"./figures/{saveTitle}_{now}.png", dpi=300,tight_layout=True)

	return fig, ax








"""
Statistical Test for whether two functions are statistically equal. 
Take function 1 over some range with uncertainty on each point {f1,s1}
Take function 2 over some range with uncertainty on each point {f2,s2}
Substract them: {f1-f2,sqrt(s1^2 + s2^2)}
These are the mean and std passed to the function. 
If f1 and f2 are statistically equivalent we'd expect every point of (f1 - f2) to be drawn from gaussian of mean zero and std the value of sqrt(s1^2 + s2^2) at that point. 
That is we'd expect this to be the same as {0,sqrt(s1^2 + s2^2)}
For a random sample of {0,sqrt(s1^2 + s2^2)} we calculate log posterior 
Do ntest of these we get a distribution. 
Then, log posterior of the true {f1-f2,sqrt(s1^2 + s2^2)} can be compared to see if it is an outlier.
The percentile is returned (>0.95 or <0.05 would be typical choices)
"""
def funcZeroTest(mean,std,ntests=1000,plot=False): #a significance test to tell if a function and it's std is zero 
	logPs = []
	for test in range(ntests):
		logP = 0
		for i in range(len(mean)):
			sample = np.random.normal(loc=0,scale=std[i])
			logP += -sample**2/(2*std[i]**2) - np.log(std[i])
		logPs.append(logP)

	logPmean = 0
	for i in range(len(mean)):
		logPmean += -mean[i]**2/(2*std[i]**2) - np.log(std[i])

	if plot == True:
		fig, ax = plt.subplots(figsize=(1,1))
		ax.hist(logPs,bins=20)
		ax.axvline(logPmean,c='r',alpha=0.5)
	percentile = len(np.where(np.sort(np.array(logPs))<logPmean)[0]) / len(logPs)

	return percentile