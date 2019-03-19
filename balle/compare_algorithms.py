AWS = 3

if AWS == 0:
    #Local Trained Model List
    algorithms = ['balle_baseline',  'balle_005_CNN', 'balle_006_CNN', 'li_004']
    labels = ['Balle Baseline Algorithm', 'Balle Encoder Variation 1', 'Balle Encoder Variation 2', 'Balle Li Combination']
    output = 'comb.bin'
elif AWS == 1:
    #AWS Trained Model List
    algorithms = ['li_004_AWS','balle_baseline_AWS', 'balle_001_quality', 'balle_002_compression', 'balle_003_highLearn', 'balle_004_lowLearn']
    labels = ['Balle Li Combination','Balle Baseline Algorithm', 'Balle High Quality', 'Balle High Compression', 'Balle High Learn', 'Balle Low Learn']
    output = 'comb.bin'
elif AWS == 2: #JPEG
    #JPEG Model List
    algorithms = ['JPEG_8', 'JPEG_16', 'JPEG_32']
    labels = ['JPEG 8', 'JPEG 16', 'JPEG 32']
    output = 'comb.jpg'
elif AWS == 3: #Custom
    #Single Custom
    algorithms = ['JPEG_16']
    labels = ['Balle Li Combo Variable Learn']
    output = 'comb.bin'

import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import numpy

testDataPath = '/home/jake/Desktop/CS230/Project/balle/images_BSD500/images_test/'

#List of All Test Files
input = glob.glob(testDataPath + '*.png')

print('Number of Test Files:', len(input))

recon = 'recon.png'

#Determine MSSSIM Comparison for Two Images
def msssim(original, reconstructed):

    original = tf.image.decode_png(tf.read_file(original))
    reconstructed = tf.image.decode_png(tf.read_file(reconstructed))
    ssim = tf.image.ssim_multiscale(original,reconstructed, 255)

    sess = tf.Session()
    print = tf.print(ssim)
    sess.run(print)

    with open('temp.txt', "a") as f:
        f.write(str(sess.run(ssim)) + '\n')

    sess.close()

#Iterate Throught Algorithms
for j in range(0,len(algorithms)):

    #Reset Temp File
    with open('temp.txt', "w") as f:
        f.write('')

    exec('import ' + algorithms[j] + ' as bBaseline')
    Label = labels[j]

    #Iterate Through Test Images
    compRatio = []
    origSize = []
    for i in range(0, len(input)):
    #for i in range(37, 39):

        print(i, input[i].split('/')[-1])

        bBaseline.compress(input[i], output)

        compSize = os.stat(output).st_size
        origSize.append(os.stat(input[i]).st_size)
        compRatio.append(origSize[-1]/compSize)
        print(compRatio[-1])

        bBaseline.decompress(output, recon)

        #Make Sure Image 5096.png gets saved off separately
        if '5096' in input[i]:
            bBaseline.decompress(output, '5096' + Label.replace(' ','_') + '.png')

        msssim(input[i], recon)

    #Compile Compressed Size Statistics from Temporary File
    with open('temp.txt', "r") as f:
        ln = f.readlines()
        msssimAll = [float(q.replace('\n', '')) for q in ln]

    #Report Results
    print('')
    print('Min/Max/Average Compression Ratio', min(compRatio), max(compRatio), sum(compRatio)/len(compRatio))
    print('')
    print('Min/Max/Average MSSSIM', min(msssimAll), max(msssimAll), sum(msssimAll)/len(msssimAll))
    print('')
    print('Min/Max/Average origSize', min(origSize), max(origSize), sum(origSize)/len(origSize))

    with open('TestResults.txt', "a") as f:
        f.write(Label + ',' + str(min(compRatio)) + ',' + str(max(compRatio)) + ',' + str(sum(compRatio)/len(compRatio)) + ',' + str(numpy.std(compRatio))
                + ',' + str(min(msssimAll)) + ',' + str(max(msssimAll)) + ',' + str(sum(msssimAll)/len(msssimAll)) +  ',' + str(numpy.std(msssimAll))
                + ',' + str(min(origSize)) + ',' + str(max(origSize)) + ',' + str(sum(origSize)/len(origSize))  + ',' + str(numpy.std(origSize)) + '\n')

    compRatioNorm = [(q-min(compRatio))/(max(compRatio)-min(compRatio)) for q in compRatio]
    msssimAllNorm = [(q-min(msssimAll))/(max(msssimAll)-min(msssimAll)) for q in msssimAll]
    origSizeNorm = [(q-min(origSize))/(max(origSize)-min(origSize)) for q in origSize]

    plt.scatter(list(compRatioNorm), list(msssimAllNorm), c='b', label='MSSSIM')

    z = numpy.polyfit(compRatioNorm, msssimAllNorm, 1)
    p = numpy.poly1d(z)
    plt.plot(compRatioNorm, p(compRatioNorm), "b--")

    plt.scatter(list(compRatioNorm), list(origSizeNorm), c='r', label='Original Image Size')

    z = numpy.polyfit(compRatioNorm, origSizeNorm, 1)
    p = numpy.poly1d(z)
    plt.plot(compRatioNorm, p(compRatioNorm), "r--")

    plt.legend(loc='lower right')
    plt.xlabel('Compression Ratio')
    plt.ylabel('MSSSIM / Original Image Size')
    plt.title(Label)
    plt.grid()
    mp.savefig('results_' + Label.replace(' ', "_") + '.png')
    plt.clf()