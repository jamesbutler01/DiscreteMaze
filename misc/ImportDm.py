import numpy as np

areas = 'Area32', 'dACC', 'OFC'

n = {'Area32':92,'dACC':162, 'OFC':143}

epochs = 'optionMade', 'optionsOn'


def getData(area, cell, epoch):

    if area not in areas:
        raise NameError(f'Invalid area, please choose from {areas}')

    if epoch not in epochs:
        raise NameError(f'Invalid epoch, please choose from {epochs}')

    if cell >= n[area]:
        raise IndexError(f'Invalid cell, there are only {n[area]} cells for {area}')

    yy = np.load(f'data/{area}_{cell}_{epoch}.npy')
    distChange = np.load(f'data/{area}_{cell}_distChange.npy')
    currAngle = np.load(f'data/{area}_{cell}_currAngle.npy')
    hd = np.load(f'data/{area}_{cell}_hd.npy')
    numsteps = np.load(f'data/{area}_{cell}_numsteps.npy')
    perfTrials = np.load(f'data/{area}_{cell}_perfTrials.npy')
    startAngle = np.load(f'data/{area}_{cell}_startAngle.npy')
    currDist = np.load(f'data/{area}_{cell}_currDist.npy')
    fromx = np.load(f'data/{area}_{cell}_fromx.npy')
    fromy = np.load(f'data/{area}_{cell}_fromy.npy')
    tox = np.load(f'data/{area}_{cell}_tox.npy')
    toy = np.load(f'data/{area}_{cell}_toy.npy')

    return yy, distChange, currAngle, hd, numsteps, perfTrials, startAngle, currDist, fromx, fromy, tox, toy





