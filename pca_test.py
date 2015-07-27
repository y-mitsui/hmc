import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")
from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA
from matplotlib import pyplot as plt


iris = datasets.load_iris()
targets = iris.target
kpca = KernelPCA(n_components=2,kernel="linear", fit_inverse_transform=True, gamma=10)
#kpca = PCA(n_components=2,kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(iris.data)
print X_kpca
for c,data in zip(targets,X_kpca):
    plt.scatter(data[0],data[1],c=['r','g','b'][c])
plt.show()