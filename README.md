# Fixed simplex coordinates for angular margin loss in CapsNet
Rita Pucci, Christian Micheloni, Gian Luca Foresti, Niki Martinel


A more stationary and discriminative embedding
is necessary for robust classification of images. We focus our
attention on the newel CapsNet model and we propose the
angular margin loss function in composition with margin loss.
We define a fixed classifier implemented with fixed weights
vectors obtained by the vertex coordinates of a simplex polytope.
The advantage of using simplex polytope is that we obtain the
maximal symmetry for stationary features angularly centred.
Each weight vector is to be considered as the centroid of a class in
the dataset. The embedding of an image is obtained through the
capsule network encoding phase, that is identified as digitcaps
matrix. Based on the centroids from the simplex coordinates
and the embedding from the model, we compute the angular
distance between the image embedding and the centroid of the
correspondent class of the image. We take this angular distance
as angular margin loss. We keep the computation proposed for
margin loss in the original architecture of CapsNet. We train
the model to minimise the angular between the embedding and
the centroid of the class and maximise the magnitude of the
embedding for the predicted class. The experiments on different
datasets demonstrate that the angular margin loss improves the
capability of capsule networks with complex datasets.
Index Terms—Machine learning, capsule network, angular loss,
fixed classifier, image analysis


![plot](https://github.com/Riretta/Angle_Loss/blob/master/Caps_Angle_FC10_2.jpg?raw=true)

##################################################################################

if you apply the idea in the project, please cite us:

1. 
Pucci, R., Micheloni, C., Foresti, G. L., & Martinel, N. (2020). 

Deep interactive encoding with capsule networks for image classification. 

Multimedia Tools and Applications, 79(43), 32243-32258. 

https://link.springer.com/article/10.1007/s11042-020-09455-8 

2.

The paper is available ICPR2020 proceedings 

