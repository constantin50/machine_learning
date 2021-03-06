<h2>K-means</h2>

The task is to make distribution of some objects over clusters. Each kth cluster is characterized with its center c_k

K-means optimizes in-cluster distance of objects

![formula](http://latex2png.com/pngs/4ab36fa1c925a3054fe37c2042648ec5.png) (*)

Let us centers of clusters are arbitary. Fix them and define in-cluster distance for each object.

![formula](http://latex2png.com/pngs/9fa560b4a7c3cd3d348669a0b2242f62.png)


Using euclidian distance we can take derivative of (*) with respect to a center c_k. Then we express formula for c_k


![formula](http://latex2png.com/pngs/597be5bb3827ed67297e9cd141eeedff.png)

Then use these formulas until convergence

![anim](https://github.com/constantin50/machine_learning/blob/master/algorithms/clustering/k-means-anim.gif)
