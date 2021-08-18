
**Pose Estimation on pix3d Using Discretized Masks**

The goal is to train a model for pose estimation on the Pix3D dataset. The pose estimation task is defined as a classification task. The model should classify the azimuth and elevation into defined number of bins.
The models were evaluated in separate phases. 

**Models**

1. In terms of inputs, we have 3 different models. 
   1. boundaries, and normals
   1. gt_mask, boundaries, and normals
   1. gt_D_mask, boundaries, and normals

1. In terms of models
   1. basline net
   1. Upsample net
   
1. In terms of how to use mask
   1. mask as channel
   1. masked features

**Results**: 

*Class agnostic/ evaluated on ground truth masks*:

|<p>Models , </p><p></p>|acc\_azimuth|acc\_azimuth2|acc\_elevation|acc\_elevation2|
| :- | :- | :- | :- | :- |
|baseline_NoMask\_net |58.1|76.5|71.5|94.6|
|baseline_gt_MaskAzChannel\_net |62.4|81.6|71.9|94.7|
|baseline_gt_MaskdOut\_net |60.6|80.8|71.1|93.9|
|baseline_gt_D_MaskAzChannel\_net |70.7|85.9|77.2|96.3|
|baseline_gt_D_MaskdOut\_net |71.7|85.9|77.4|96.4|
|upsample_NoMask\_net |63.1|81.5|74.7|95.4|
|upsample_gt_MaskAzChannel\_net |66.2|84.2|73.6|94.9|
|upsample_gt_MaskedOut\_net |69.8|87.1|77.8|96.2|
|upsample_gt_D_MaskAzChannel\_net |90|95.1|92.4|98.9|
|upsample_gt_D_MaskedOut\_net |88.7|95|92.4|98.9|


*Accuracy for best model per category*:

|<p>Models , </p><p></p>|chair|table|sofa|bed|desk|bookcase|wardrobe|misc|tool|
| :- | :- | :- | :- | :- |:- | :- | :- | :- | :- |
|count|1167|417|415|213|154|79|54|20|11|
|acc\_azimuth |81.1|98.1|90.7|100|54.5|96.2|94.8|81.8|40|


*upsample_gt_D_mask on training. For evaluation, we get top 4 candidate D_masks using upsample_No_mask model. *:

|<p>Models , </p><p></p>|acc\_azimuth|acc\_azimuth2|
| :- | :- | :- |
| upsample_gt_D_mask |50.0|77.5|

problem : model relies to much on mask (60%)
