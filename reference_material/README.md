# mids_w207_music_genre/reference_material
A collection of useful reference material for the project.

**2002-07 Music Genre Classification of Audio Signals**

G. Tzanetakis and P. Cook, "Musical genre classification of audio
signals," in IEEE Transactions on Speech and Audio Processing,
vol. 10, no. 5, pp. 293-302, July 2002, doi: 10.1109/TSA.2002.800560.

Abstract—Musical genres are categorical labels created by humans to
characterize pieces of music. A musical genre is characterized by the
common characteristics shared by its members.

These characteristics typically are related to the instrumentation,
rhythmic structure, and harmonic content of the music. Genre
hierarchies are commonly used to structure the large collections of
music available on the Web. Currently musical genre annotation is
performed manually. Automatic musical genre classification can assist
or replace the human user in this process and would be a valuable
addition to music information retrieval systems. In addition,
automatic musical genre classification provides a framework for
developing and evaluating features for any type of content- based
analysis of musical signals.

In this paper, the automatic classification of audio signals into an
hierarchy of musical genres is explored. More specifically, three
feature sets for representing timbral texture, rhythmic content and
pitch content are proposed.

The performance and relative importance of the proposed features is
investigated by training statistical pattern recognition classifiers
using real-world audio collections. Both whole file and real-time
frame-based classification schemes are described. Using the proposed
feature sets, classification of 61% for ten musical genres is
achieved. This result is comparable to results reported for human
musical genre classification.

*Excerpts*

5) Mel-Frequency Cepstral Coefficients: Mel-frequency cepstral
coefficients (MFCC) are perceptually motivated features that are also
based on the STFT. After taking the log-amplitude of the magnitude
spectrum, the FFT bins are grouped and smoothed according to the
perceptually motivated Mel-frequency scaling. Finally, in order to
decorrelate the resulting feature vectors a discrete cosine transform
is performed.  Although typically 13 coefficients are used for speech
representation, we have found that the first five coefficients provide
the best genre classification performance.

6) Analysis and Texture Window: In short-time audio analysis, the
signal is broken into small, possibly overlapping, segments in time
and each segment is processed separately.  These segments are called
analysis windows and have to be small enough so that the frequency
characteristics of the magnitude spectrum are relatively stable (i.e.,
assume that the signal for that short amount of time is
stationary). However, the sensation of a sound “texture” arises as the
result of multiple short-time spectrums with different characteristics
following some pattern in time. For example, speech contains vowel and
consonant sections which have very different spectral characteristics.

Therefore, in order to capture the long term nature of sound
“texture,” the actual features computed in our system are the running
means and variances of the extracted features described in the
previous section over a number of analysis windows. The term texture
window is used in this paper to describe this larger window and
ideally should correspond to the minimum time amount of sound that is
necessary to identify a particular sound or music “texture.”
Essentially, rather than using the feature values directly, the
parameters of a running multidimensional Gaussian distribution are
estimated. More specifically, these parameters (means, variances) are
calculated based on the texture window which consists of the current
feature vector in addition to a specific number of feature vectors
from the past. Another way to think of the texture window is as a
memory of the past.  For efficient implementation a circular buffer
holding previous feature vectors can be used. In our system, an
analysis window of 23 ms (512 samples at 22 050 Hz sampling rate) and
a texture window of 1 s (43 analysis windows) is used.