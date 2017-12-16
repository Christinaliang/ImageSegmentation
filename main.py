import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=int, default=3)
parser.add_argument('--thresh', type=float, default=1E-4)
parser.add_argument('--input', type=str, default='image.jpg')
args = parser.parse_args()

K = args.cluster
D = 3
THRESH = args.thresh
useType = np.float64

def findCentroids(feats, centroids):
    N = feats.shape[0]
    idx = np.zeros((N), dtype=np.int)
    for i in xrange(N):
        bestIdx = -1
        bestDist = -1
        for j in xrange(K):
            u = feats[i] - centroids[j]
            dist = np.inner(u, u)
            if (bestIdx == -1 or dist < bestDist):
                bestIdx = j
                bestDist = dist
        idx[i] = bestIdx
    return idx

if __name__ == '__main__':
    img = cv2.imread(args.input)

    size = (img.shape[0] * img.shape[1], img.shape[2])
    feats = np.reshape(img, size).astype(useType) / 256.0

    centroids = np.random.rand(K, D).astype(useType)
    newCentroids = np.zeros((K, D), dtype=useType)

    ITER = 1
    N = size[0]
    print 'Number of Features:', N
    print 'Number of Clusters:', K

    while (True):
        idx = findCentroids(feats, centroids)
        for i in xrange(K):
            newCentroids[i] = np.mean(feats[idx == i, :], axis=0)

        error = 0.0
        for i in xrange(K):
            u = newCentroids[i] - centroids[i]
            error = error + np.sqrt(np.inner(u, u))

        print 'ITERATION:', ITER, 'Error Rate:', error
        ITER = ITER + 1
        if (error < THRESH):
            break
        centroids = newCentroids.copy()

    idx = findCentroids(feats, centroids)
    centroids = centroids * 256.0
    idx = np.reshape(idx, (img.shape[0], img.shape[1]))
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            pix = centroids[idx[i][j]]
            img[i][j] = pix

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


