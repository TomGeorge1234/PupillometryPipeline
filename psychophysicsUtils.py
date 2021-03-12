import pandas as pd 
import numpy as np 
from scipy import stats
from tqdm.notebook import tqdm
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import matplotlib 
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


#converts string times into scalar, units of seconds 
def scalarTime(strTime): # strTime of form 'HH:MM:SS:msmsmsmsms.....'
	hours = int(strTime[0:2])*60*60
	minutes = int(strTime[3:5])*60
	seconds = int(strTime[6:8])
	milliseconds = float('0.'+strTime[9:])
	return hours+minutes+seconds+milliseconds #time in s


"""
Loads pupil data. Defaults to EyeLink which has name format "name_pupillometry.csv" else tries pupilLabs "name_pupillometryPL"
If EyeLink, then data file has Bonsai timesyncs saved within it, these are used to sync time with Bonsai machine.
If PupilLabs, timesync data saved in separate csv "name_timesync.csv". These are loaded and used to sync time.

Returns
•pupilDiams: array of pupil diameters, raw
•times: time (corresponding to time on the Bonsai machine, for each pupil diameter reading)
•dt: average time step
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
			dt = np.mean((np.roll(times,-1) - times)[1:-1]); print('dt = %.4fs' %dt)
			print("Percentage data missing : %.2f %%" %(100*len(np.where(pupilDiams == 0)[0])/len(pupilDiams)))
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
		dt = np.mean((np.roll(times,-1) - times)[1:-1]); print('dt = %.4fs' %dt)
		print("Percentage data missing : %.2f %%" %(100*len(np.where(pupilDiams == 0)[0])/len(pupilDiams)))  
		loadComplete = True

	return pupilDiams, times, dt

"""
Loads timesync file made when pupillabs is recorded.
Returns two array: one for timestamps from computer (presumably Bonsai or otherwise is running here)
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
Plots two arrays and a histogram showing full timeseries and zoomed in time series of pupil diameters
"""
def plotPupilDiams(pupilDiams, times, dt, zoomRange = [0,60], saveName = None, hist=True, ymin=0, ymax=None, color='C0'):
	if ymax == None: 
		ymax = np.max(pupilDiams)
	if ymin=='-ymax': ymin = -ymax
	fig, ax = plt.subplots(1,2,figsize=(4,2),sharey=True)
	ax[0].plot((times[int(zoomRange[0]/dt):int(zoomRange[1]/dt)] - times[0]),pupilDiams[int(zoomRange[0]/dt):int(zoomRange[1]/dt)],c=color)
	ax[0].set_ylim([ymin,ymax])
	ax[0].set_ylabel('Pupil diameter')
	ax[0].set_xlabel('Time from start of recording / s')
	ax[0].set_title('Raw data (%gs)'%(zoomRange[1]-zoomRange[0]))

	ax[1].plot((times - times[0]),pupilDiams,c=color)
	ax[1].set_ylim([ymin,ymax])
	ax[1].set_ylabel('Pupil diameter')
	ax[1].set_xlabel('Time from start of recording / s')
	ax[1].set_title('Raw data (full)')

	if hist==True: 
		divider = make_axes_locatable(ax[1])
		axHisty = divider.append_axes("right", 0.2, pad=0.02, sharey=ax[1])
		axHisty.yaxis.set_tick_params(labelleft=False)
		axHisty.xaxis.set_tick_params(labelbottom=False)
		binwidth = (ymax-ymin)/20
		ymax = np.max(np.abs(pupilDiams))
		lim = (int(ymax/binwidth) + 1)*binwidth
		bins = np.arange(-lim, lim + binwidth, binwidth)
		axHisty.hist(pupilDiams, bins=bins, orientation='horizontal',color='C2',alpha=0.8)

	if saveName is not None: 
		plt.savefig('./figures/' + saveName + '.pdf',tightlayout=True, transparent=False,dpi=100)

	return fig, ax


"""
Performs interpolation to remove zero-values from the data
If wherever a range of pupil diamteres are zero these are replaced with linear interpolation between 'gapExtension' seconds before and after  assuming these are non-zero (moving further out if they aren't).
The values +- gapExtension are themselves also replaced since the blink or whatever maybe cause a smooth drop to zero we also want to remove

"""
def interpolatePupilDiams_new(pupilDiams, times, dt, gapExtension = 0.2):
	interpolatedPupilDiams = pupilDiams.copy()
	i = 0
	jump_dist = int(gapExtension / dt) #interpolates between gapExtension seconds before and after the points where it they fell to zero

	print("Interpolating missing values: ", end="")
	totalInterpolated = 0
	while True:

		
		if i >= len(pupilDiams): break 

		if pupilDiams[i] != 0: #the value exists and there is no problem
			interpolatedPupilDiams[i] = pupilDiams[i] 
			i += 1

		elif pupilDiams[i] == 0:#do some interpolation

			k = jump_dist
			while True: 
				if i-k < 0: #edge case where we fall off array 
					start, startidx = np.mean(pupilDiams), 0
					break
				elif i-k >= 0:
					if pupilDiams[i-k] == 0:
						k += 1 #keep extending till you get non-zero val
					elif pupilDiams[i-k] != 0:
						start, startidx = pupilDiams[i-k], i-k+1
						break
			
			j = i 
			while True: 
				while True: #find 'end' of blink
					if j >= len(pupilDiams):
						j = j-1
						break
					if pupilDiams[j] == 0:
						j += 1
					elif pupilDiams[j] !=0:
						j = j-1
						break

				if j+k >= len(pupilDiams): #edge case where we fall off array 
					end, endidx = np.mean(pupilDiams), len(pupilDiams)
					break
				elif j+k < len(pupilDiams):
					if pupilDiams[j+k] == 0:
						k += 1 #keep extending till you get non-zero val
					elif pupilDiams[j+k] != 0:
						end, endidx = pupilDiams[j+k], j+k-1
						break
			interpolatedPupilDiams[startidx:endidx] = np.linspace(start,end,endidx-startidx)
			totalInterpolated += endidx-startidx
			i=endidx+1
	print("%.2f%% of values are now interpolated" %(100*totalInterpolated/len(interpolatedPupilDiams)))
	return interpolatedPupilDiams

"""
def interpolatePupilDiams(pupilDiams, times, dt, gapExtension = 0.2):
	interpolatedPupilDiams = pupilDiams.copy()
	i = 0
	jump_dist = int(gapExtension / dt) #interpolates missing values between gapExtension seconds before and after the points where it they fell to zero

	print("Interpolating missing values")
	while True:
		if i >= len(pupilDiams): break 

		if pupilDiams[i] != 0:
			interpolatedPupilDiams[i] = pupilDiams[i] 
			i += 1
		else:
			k = jump_dist
			while True:
				if pupilDiams[i-k] != 0:
					start, startidx = pupilDiams[i-k], i-k
					break
				else: 
					k -= 1
			j = i
			while True:
				j += 1
				if j >= len(pupilDiams):
					interpolatedPupilDiams[startidx+1:] = pupilDiams[startidx] * np.ones(shape=len(interpolatedPupilDiams[startidx+1:]))
					endidx = len(pupilDiams)
					break
				elif pupilDiams[j] != 0: 
					k = jump_dist
					while True:
						if pupilDiams[j + k] != 0: 
							end, endidx = pupilDiams[j+k], j+k
							break
						else: k -= 1
					interpolatedPupilDiams[startidx+1:endidx] = np.linspace(start,end,endidx-startidx+1)[1:-1]
					break
			i = endidx

	return interpolatedPupilDiams
"""
def removeSizeAndSpeedOutliers(pupilDiams,times,n_speed=2.5,n_size=2.5, plotHist=False): #following Leys et al 2013 
	print("Removing speed outliers", end="")
	pd = pupilDiams
	absSpeed = np.zeros(len(pd))
	size = pupilDiams
	for i in range(len(pupilDiams)):
		absSpeed[i]=max(np.abs((pd[i]-pd[i-1])/(times[i]-times[i-1])),np.abs((pd[(i+1)%len(pd)]-pd[i])/(times[(i+1)%len(pd)]-times[i])))

	MAD_speed = np.median(np.abs(absSpeed - np.median(absSpeed)))
	MAD_size = np.median(np.abs(size - np.median(size)))
	threshold_speed_low = np.median(absSpeed) - n_speed*MAD_speed
	threshold_size_low = np.median(size) - n_size*MAD_size
	threshold_speed_high = np.median(absSpeed) + n_speed*MAD_speed
	threshold_size_high = np.median(size) + n_size*MAD_size
	pd = pd * (absSpeed<threshold_speed_high) * (absSpeed>threshold_speed_low)
	print(" (%.2f%%) " %(100*(1-np.sum((absSpeed<threshold_speed_high) * (absSpeed>threshold_speed_low))/len(pd))),end="")
	pd = pd * (pupilDiams>threshold_size_low) #only take away low sizes
	print("and size lowliers (%.2f%%) " %(100*(1-np.sum((size>threshold_size_low))/len(pd))), end="")
	print(" (additional %.2f%% removed vs raw)" %(100*(np.sum(pd == 0) - np.sum(pupilDiams==0))/len(pd)))
	if plotHist == True:
		fig, ax = plt.subplots(1,2)
		ax[0].hist(np.log(absSpeed),bins=30)
		ax[1].hist(size,bins=30)
		ax[0].axvline(x=np.median(absSpeed),c='k')
		ax[0].axvline(x=threshold_speed_low,c='k')
		ax[0].axvline(x=threshold_speed_high,c='k')
		ax[1].axvline(x=np.median(size),c='k')
		ax[1].axvline(x=threshold_size_low,c='k')
		ax[1].axvline(x=threshold_size_high,c='k')
		ax[0].set_title("Abs Speed")
		ax[1].set_title("Size")
	
	return pd

#upsample to make uniform time spacing ()
def upsample(pupilDiams, times, dt=None, new_dt = None, aligntimes = None): 
	print("Upsampling pupil data to %gHz: " %int(1/new_dt), end="")
	new_times = []
	new_pupildiams = []
	if aligntimes is None: 
		t = times[0] 
		pd = pupilDiams[0] 
		while True:
			new_times.append(t)
			new_pupildiams.append(pd)
			t += new_dt
			if t > times[-1]:
				break
			else: #interpolate 
				idx_r = np.searchsorted(times,t)
				idx_l = idx_r-1
				delta = times[idx_r] - times[idx_l]
				pd = ((t-times[idx_l])/delta)*pupilDiams[idx_l] + ((times[idx_r] - t)/delta)*pupilDiams[idx_r]
	elif aligntimes is not None: 
		for t in aligntimes: 
			if t > times[-1] or t < times[0]:
				pass
			else: 
				idx_r = np.searchsorted(times,t)
				idx_l = idx_r-1
				delta = times[idx_r] - times[idx_l]
				pd = ((t-times[idx_l])/delta)*pupilDiams[idx_l] + ((times[idx_r] - t)/delta)*pupilDiams[idx_r]
				new_times.append(t)
				new_pupildiams.append(pd)
	new_dt = np.mean((np.roll(new_times,-1) - new_times)[1:-1]); print('dt = %.4fs' %new_dt)

	return np.array(new_pupildiams), np.array(new_times), new_dt 

def downsample(pupilDiams, times, dt, Hz=50):

	print("Downsampling pupil data to %gHz: " %Hz, end="")
	downsampledPupilDiams = []
	downsampledTimes = []
	binSize = int((1/dt) / Hz) #current no. samples in one second / 50 
	for i in range(np.int(np.floor(len(pupilDiams)/binSize))):
		downsampledPupilDiams.append(np.mean(pupilDiams[i*binSize:(i+1)*binSize]))
		downsampledTimes.append(times[int((i+0.5)*binSize)])
	downsampledPupilDiams = np.array(downsampledPupilDiams)
	downsampledTimes =  np.array(downsampledTimes)

	pupilDiams = downsampledPupilDiams
	times = downsampledTimes
	dt = np.mean((np.roll(times,-1) - times)[1:-1]); print('dt = %.4fs' %dt)

	return pupilDiams, times, dt


def frequencyFilter(signal,time,dt,cutoff_freq,cutoff_width,highpass=False):

	if highpass == True: name=('Highpass','below')
	else: name = ('Lowpass','above')

	print("%s filtering out frequencies %s %.2f +- %.2fHz" %(name[0], name[1], cutoff_freq, cutoff_width))
	#derive the filter
	f = fftpack.fftfreq(signal.size,dt)
	fil = 1/(1 + np.exp(-(4/cutoff_width)*(np.abs(f) - cutoff_freq)))
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


	#convolve this with signal, then crop 
	filtered_signal = np.convolve(signal,fil_ifft)
	if signal.size%2 == 1: 
		filtered_signal = filtered_signal[int(np.floor(signal.size/2)):-int(np.floor(signal.size/2))]
	elif signal.size%2 == 0: 
		filtered_signal = (0.5*(filtered_signal + np.roll(filtered_signal,1)))[int(np.floor(signal.size/2)):-int(np.floor(signal.size/2))+1]        
	if False: 
		fig, ax = plt.subplots(1,2, figsize=(4,2))
		ax[0].plot(time,filtered_signal)
		ax[0].set_title("Filtered Signal")
		ax[1].plot(time,signal)
		ax[1].set_title("Original Signal")
		ax[0].set_xlabel("Time / s")
		ax[1].set_xlabel("Time / s")


	return filtered_signal




def zScore(pupilDiams, times=None, normrange=None):
	start_idx, end_idx = 0, -1
	if ((times is not None) and (normrange is not None)):
		start_idx, end_idx = np.argmin(np.abs(times-normrange[0])), np.argmin(np.abs(times-normrange[1]))
	print("z scoring \n \n \n ")
	zscorePupilDiams = (pupilDiams - np.mean(pupilDiams[start_idx:end_idx]))/np.std(pupilDiams[start_idx:end_idx])
	return zscorePupilDiams




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



			#old version specifics
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


def sliceAndAlign(data, alignEvent = 'toneStart', conditionsList=[], tstart = -2, tend = 5): 

	trials, pupilDiams, rawPupilDiams, times, dt = data['trialData'], data['pupilDiams'], data['rawPupilDiams'], data['times'], data['dt']
	alignedData = []
	alignedTime = np.linspace(tstart,tend,int((tend - tstart)/dt))

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

		if 11 in conditionsList: #activate if you want to EXCLUDE trials where the pupil diameter goes over 4 std:
			if np.max(np.abs(pupilDiams[startidx:endidx])) > 4:
				verdict *= False

		if 12 in conditionsList: #only trials when tone was BEFORE gap
			if trials[i]['toneAfterGap'] == True:
				verdict *= False 

		if 13 in conditionsList: #exclude trials where over 20% of the pupil data is interpolated
			if np.sum((rawPupilDiams[startidx:endidx] == 0))/(endidx-startidx) >= 0.2:
				verdict *= False

		if 14 in conditionsList: #only trials where tone didn't have a decreasing note e.g. ABDD
			tones = ['A','B','C','D']
			trialTones = trials[i]['tonesList']
			if not ((tones.index(trialTones[1]) >= tones.index(trialTones[0])) and 
				(tones.index(trialTones[2]) >= tones.index(trialTones[1])) and
				(tones.index(trialTones[3]) >= tones.index(trialTones[2]))): 
				verdict *= False

		if verdict==True:
			alignedPupilDiams = pupilDiams[startidx:endidx]
			alignedData.append(list(alignedPupilDiams))

	alignedData = np.array(alignedData)

	print("%g compatible trials found" %len(alignedData))

	return alignedData, alignedTime


def plotAlignedPupilDiams(participantData,  #from particpants
						  alignEvent='toneStart',
						  tstart=-2, tend=5,
						  title='Pupil response',
						  testRange=[0,3],
						  saveTitle=None,
						  trialpreaverage = 0, #removes mean from  this many seconds BEFORE
						  dd={
							  '':     {'color':'C2','conditions':[0,4,5],'range':('all'),'plotTrials':True},
							 }):

	fig, ax = plt.subplots(figsize=(3.5,2))

	top, bottom = 0, 0
	for name, details in list(dd.items()):
		print(name)
		for (p,participant) in enumerate(list(participantData.keys())):
			print(participant +": ",end = '')
			d,t = sliceAndAlign(participantData[participant], alignEvent=alignEvent,conditionsList=details['conditions'],tstart=tstart,tend=tend)

			if trialpreaverage == True: 
				print(np.mean(np.mean(d[:,:np.argmin(np.abs(t-0))],axis=1)),np.std(np.mean(d[:,:np.argmin(np.abs(t-0))],axis=1)))
				print(np.argmin(np.abs(t-0)))
				d = (d.T - np.mean(d[:,:np.argmin(np.abs(t-0))],axis=1)).T


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
				except ValueError: print("participant excluded, no compatible trials")

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
		plt.savefig("./figures/"+saveTitle+".png", dpi=300,tight_layout=True)

	return fig, ax


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