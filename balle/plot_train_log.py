trainFile = '1549059819.0_train_log.txt'
PlotTitle = 'Training Loss: J. Balle Algorithm lambda=0.01, batchSize=1, patchsize=256'

with open(trainFile, 'r') as f:
    ln = f.readlines()

step = []
loss = []
for n in ln:
    if 'step=' in n and 'last_step' not in n:
        print (n)
        d = n.split(',')
        print(d)
        step.append(float(d[0].split('=')[-1]))
        loss.append(float(d[1].split('=')[-1]))

import matplotlib.pyplot as plt

plt.plot(step, loss)
plt.xlabel('step')
plt.ylabel('training loss')
plt.title(PlotTitle)
plt.grid()
plt.show()


