import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

AWS = 5

if AWS == 0:
    input = '/home/jake/Desktop/CS230/Project/results/Local_TestResults.txt'
    plotLabels = ['Baseline', 'Encoder1', 'Encoder2', 'Li Combo']
elif AWS ==1:
    input = '/home/jake/Desktop/CS230/Project/results/AWS_TestResults.txt'
    plotLabels = ['Baseline', 'Quality', 'Comp', 'Learn+', 'Learn-']
elif AWS == 2:
    input = '/home/jake/Desktop/CS230/Project/results/JPEG_TestResults.txt'
    plotLabels = ['JPEG 8', 'JPEG 16', 'JPEG 32']
elif AWS == 3:
    input = '/home/jake/Desktop/CS230/Project/results/ALL_TestResults.txt'
    plotLabels = ['Baseline', 'Encoder1', 'Encoder2', 'Li Combo', 'JPEG 8', 'Baseline AWS', 'Quality', 'Comp', 'Learn+', 'Learn-', 'JPEG 16', 'JPEG 32']
elif AWS == 4:
    input = '/home/jake/Desktop/CS230/Project/results/Select_TestResults.txt'
    plotLabels = ['Baseline AWS','Baseline', 'Encoder1', 'Li Combo', 'JPEG 16']
    offsetX = [     0,     -4,      0,      0, 0.0]
    offsetY = [0.0025, 0.0025, 0.0025, -0.005, 0.0025]
elif AWS == 5:
    input = '/home/jake/Desktop/CS230/Project/results/Select2_TestResults.txt'
    plotLabels = ['Baseline Balle', 'JPEG 16', 'Combo Li', 'Balle Encoder', ]
    offsetX = [     0,     -4,      0, 0.0]
    offsetY = [0.0025, 0.0025, -0.005, 0.0025]

#Pull Summary Results Data
with open(input, 'r') as f:
    ln = f.readlines()
d = [q.replace('\n','').split(',') for q in ln]
label = [q[0] for q in d]
comp = [q[1:5] for q in d]
msssim = [q[5:9] for q in d]
orig =[q[9:13] for q in d]

print(label)
print(comp)
print(msssim)
print(orig)

minMSSSIM = [float(q[0]) for q in msssim]
maxMSSSIM  = [float(q[1]) for q in msssim]
meanMSSSIM = [float(q[2]) for q in msssim]
stdMSSSIM = [float(q[3]) for q in msssim]
minComp = [float(q[0]) for q in comp]
maxComp  = [float(q[1]) for q in comp]
meanComp = [float(q[2]) for q in comp]
stdComp = [float(q[3]) for q in comp]


print('Mean MSSSIM', meanMSSSIM)
print('Mean Comp', meanComp)

# Random test data
np.random.seed(123)
# msssim_data = [np.random.uniform(minMSSSIM[q], maxMSSSIM[q], 1000) for q in range(0, len(msssim))]
# comp_data = [np.random.uniform(minComp[q], maxComp[q], 1000) for q in range(0, len(comp))]

msssim_data = [np.random.normal(meanMSSSIM[q], stdMSSSIM[q], 1000) for q in range(0, len(msssim))]
comp_data = [np.random.normal(meanComp[q], stdComp[q], 1000) for q in range(0, len(comp))]

for r in range(0,len(msssim_data)):
    msssim_data[r] = [min(q,1) for q in msssim_data[r]]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# rectangular box plot
bplot1 = axes[0].boxplot(msssim_data, vert=True,  patch_artist=True, usermedians=meanMSSSIM, showfliers=False)

# notch shape box plot
bplot2 = axes[1].boxplot(comp_data,
                         notch=True,  # notch shape
                         vert=True,   # vertical box aligmnent
                         patch_artist=True, usermedians=meanComp, showfliers=False)   # fill with color

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
cnt = 0
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(msssim_data))], )
    if cnt ==0:
        ax.set_ylabel('MSSSIM')
        cnt +=1
    else:
        ax.set_ylabel('Compression Ratio')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(comp_data))],
         xticklabels=plotLabels)

#Setup Ellipse Plot
NUM = len(msssim)

ells = []
B = []
for i in range(0,len(msssim)):
    B.append([meanComp[i],meanMSSSIM[i]])
    W = maxComp[i]-minComp[i]
    H = (maxMSSSIM[i] - minMSSSIM[i])
    print('W, H', W, H)
    ells.append(Ellipse(xy=B[i],
                width=W, height= H,
                angle=0))



fig, ax = plt.subplots(subplot_kw={'aspect': 'auto'})
for q in range(0,len(ells)):
    e = ells[q]
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(np.random.rand())
    color = np.random.rand(3)


    if AWS <4:
        ax.text(B[q][0], B[q][1], plotLabels[q], style='italic')
    else:
        ax.text(B[q][0]+offsetX[q], B[q][1]+offsetY[q], plotLabels[q], style='italic')
        #color = colors[q]

    e.set_facecolor(color)
    ax.scatter(B[q][0], B[q][1], c='r', zorder=100)

ax.set_xlim(min(minComp), max(maxComp))
ax.set_ylim(min(minMSSSIM), max(maxMSSSIM))
ax.set_xlabel('Compression Ratio')
ax.set_ylabel('MS-SSIM')


plt.grid()
plt.show()
