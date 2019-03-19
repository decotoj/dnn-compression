import glob
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import numpy as np

AWS = 1

if AWS == 0:
    testDataPath = '/home/jake/Desktop/CS230/Project/results/trainLogs/Local/'
    StepSize = 10
    labels = ['Encoder1', 'Encoder2', 'Baseline', 'Li Combo']
elif AWS == 1:
    testDataPath = '/home/jake/Desktop/CS230/Project/results/trainLogs/AWS/'
    StepSize = 1000
    labels = ['Li Combo', 'High Lambda', 'Low Lambda', 'High Learn', 'Low Learn', 'Baseline']

#List of All Test Files
input = glob.glob(testDataPath + '*.txt')
input.sort()

for q in range(0,len(input)):

    print('INPUT:', q, input[q])

    with open(input[q], 'r') as f:
        ln = f.readlines()

    step = []
    loss = []
    rateLoss = []
    distortionLoss = []
    cnt = 0
    for n in ln:
        if 'step=' in n and 'last_step' not in n:
            d = n.split(',')
            step.append(cnt*StepSize)
            loss.append(float(d[1].split('=')[-1]))
            loss[-1] = np.log(loss[-1])
            # rateLoss.append(float(d[2].split('=')[-1]))
            # distortionLoss.append(float(d[3].split('=')[-1])*lmbda)
            cnt+=1

    if AWS == 0:
        step = step[0::100]
        loss = loss[0::100]


    plt.plot(step, loss, label=labels[q])
    # plt.plot(step, rateLoss, 'r')
    # plt.plot(step, distortionLoss, 'k')

plt.legend(loc='upper right')
plt.xlabel('Training Step')
plt.ylabel('Loss (Log Scale)')
plt.title('Training Loss')
plt.grid()

plt.show()


