K-means optimizes in-cluster distance of objects

$\sum_{k=1}^{K} \sum_{i=1}^{\ell}\left[a\left(x_{i}\right)=k\right] \rho\left(x_{i}, c_{k}\right)$  (*)

$K$ - number of clusters, $\ell$ - number of objects, $c_k$ - center of $kth$ cluster.

Les us centers of clusters are arbitary. Fix them and define in-cluster distance for each object.

$a\left(x_{i}\right)=\underset{1 \leqslant k \leqslant K}{\arg \min } \rho\left(x_{i}, c_{k}\right)$

Using euclidian distance we can take derivative of (*) with respect to a center $c_k$. Then we express formula for $c_k$

$c_{k}=\frac{1}{\sum_{i=1}^{\ell}\left[a\left(x_{i}\right)=k\right]} \sum_{i=1}^{\ell}\left[a\left(x_{i}\right)=k\right] x_{i}$
