import os
import sys

import math
import numpy
import scipy
from scipy import signal
from scipy import convolve
from scipy.io import wavfile

import matplotlib
from matplotlib import pyplot, pylab

# Testing

EPS = signal.filter_design.EPSILON

class BadSoundFile(Exception):
	pass

class InvalidWindow(Exception):
	pass


class SpeechFrame(object):
    
	def __init__(self, frame, fs):
		self.frame = frame
		self.fs = fs
	
	def _getLogEnergy(self):
		return 10*numpy.log10(numpy.sum(EPS + numpy.power(self.frame,2)))
	
	logEnergy = property(_getLogEnergy)
	
	def _getZeroCrossings(self):
		return int(numpy.sum(numpy.abs(numpy.diff(numpy.sign(self.frame)))/2))
	
	zeroCrossings = property(_getZeroCrossings)
	
	def cepstrum(self):
		ms1=self.fs/1000.0;           # maximum speech Fx at 1000Hz
		ms20=self.fs/50.0;            # minimum speech Fx at 50Hz
		xfft=scipy.fft(self.frame);
		x_hat=scipy.ifft(numpy.log(numpy.abs(xfft)+EPS));
		x_hat = numpy.real(x_hat[ms1:ms20]);
		q=scipy.r_[ms1:ms20]/float(self.fs);
		return (q,x_hat)
	
	def getCepstrumPeak(self):
		(q,c) = self.cepstrum()
		index = c.argmax()
		return (q[index],c[index])

	def _getPitch(self):
		ms2 = math.floor(self.fs/500) # 2ms
		ms20 = math.floor(self.fs/50) # 20ms 
		t,r = self.autoCorrelate()
		r = r[ms2:ms20]
		index = r.argmax()
		return (self.fs/(ms2+index-1),r[index])
				
	pitch = property(_getPitch)
	
	def autoCorrelate(self):
		r = signal.correlate(self.frame,self.frame,'same')
		r = r[math.floor(len(r)/2):]
		t = scipy.r_[0:len(r)] / float(self.fs)
		return (t,r)

	

class Speech(object):
	
	HAMMING = "HAMMING"
	RECT = "RECTANGLE"
	WINTYPES = (HAMMING,RECT)
	
	
	def __init__(self, path=None):
		if not path is None:
			try:
				self.open(path)
			except IOError:
				raise BadSoundFile("Unable to open wave file '%s'" % path)

		self.windowLength=0
		self.window = numpy.zeros(0)
		self.frame = numpy.array
	
	def open(self, path):
		self.wav = wavfile.read(path)
		self.signal = self.wav[1]
		
	def _setWindowOffset(self, windowOffset):
		assert windowOffset > 0
		self.windowOffset = windowOffset
		
	def setWindow(self, length, offset, wintype):
		if not wintype in self.WINTYPES:
			raise InvalidWindow("Select Valid Window Type")
		if not length > 0:
			raise InvalidWindow("Select Valid Window Size")
			
		if wintype == self.RECT:
			self.window = numpy.ones(length)
		if wintype == self.HAMMING:
			self.window = ( 0.54 - 0.46 * 
			numpy.cos(2*numpy.pi*scipy.r_[0:length]/(length-1)))
		
		assert length > offset
			
		self.windowLength = length
		self._setWindowOffset(offset)
	
	
	def _getSamplingFrequency(self):
		return self.wav[0]
	
	samplingFrequency = property(_getSamplingFrequency)
	
	def _getSampleLength(self):
		return self.signal.size
	
	sampleLength = property(_getSampleLength)
	
	def _getNumberFrames(self):
		return int(math.floor(self.sampleLength/self.windowOffset))
	
	numberOfFrames = property(_getNumberFrames)
	
	def _getFrame(self):
		assert self.windowOffset >  0
		zerolen = math.ceil(len(self.window)/2)
		tempwav = numpy.concatenate((numpy.zeros(zerolen),self.signal,
										numpy.zeros(zerolen)))
		for i in xrange(int(math.floor(self.sampleLength/self.windowOffset))):
			offset = i*self.windowOffset
			yield tempwav[offset:offset+self.windowLength]
	
	def getWindowedFrame(self):
		for frame in self._getFrame():
			yield SoundFrame(frame*self.window,self.samplingFrequency)
	
	def filter(self,lowcutoff, highcutoff):
		self.lowcutoff = lowcutoff
		self.highcutoff = highcutoff
		nyq = self.samplingFrequency / 2
		low = float(lowcutoff)
		high = float(highcutoff)
		#Lowpass filter
		a = signal.firwin(nyq, cutoff = low/nyq, 
									window = 'blackmanharris')
		#Highpass filter with spectral inversion
		b = - signal.firwin(nyq, cutoff = high/nyq, 
									window = 'blackmanharris') 
		b[nyq/2] = b[nyq/2] + 1
		#Combine into a bandpass filter
		self._d = - (a+b) 
		self._d[nyq/2] = self._d[nyq/2] + 1
		self.signal = convolve(self._d, self.signal)
		
	def runAnimation(self):
		
		f = pyplot.figure(1)
		f.subplotpars.update(hspace=1,wspace=1)
		
		for frame in self.getWindowedFrame():
			pylab.subplot(331)
			pyplot.plot(frame.frame)
			pylab.title("ST waveform")
			pylab.ylim([self.signal.min(),self.signal.max()])
			pylab.xlim([0,self.windowLength])
			pylab.subplot(332)
			pyplot.bar(0,frame.logEnergy,2)
			pylab.title("ST Energy %.1f" % frame.logEnergy)
			pylab.ylim([-200,200])
			pylab.subplot(333)
			pyplot.bar(0,frame.zeroCrossings,2)
			pylab.title("ST ZC %.1f" % frame.zeroCrossings)
			pylab.ylim([0,100])
			pylab.subplot(312)
			pyplot.plot(*frame.cepstrum())
			pyplot.hold(True)
			(q,c) = frame.getCepstrumPeak()
			pylab.title("Cepstrum (q peak: %.4f, pitch: %.2f)" % (q, 1/q))
			pyplot.scatter(q,c,s=100,c='r')
			pyplot.hold(False)
			pylab.subplot(313)
			pyplot.plot(*frame.autoCorrelate())
			pyplot.hold(True)
			(pitch,r) = frame.pitch
			pylab.title("Autocorrelation (pitch: %3.1f)" % pitch)
			t = 1/float(pitch)
			pyplot.scatter(t,r,s=100,c='r')
			pyplot.hold(False)
			matplotlib.pylab.draw()
			f.clear()
			

	
def main():
	pass
	
if __name__ == '__main__':
	import time
	main()
	sound = Speech("test_sounds/f1nw000016k.wav")
	sound.setWindow(512,160,Sound.HAMMING)
	sound.filter(100,900)
	sound.runAnimation()
		
		
		
		
