"""WsnFeatures — WSN dataset feature subset definitions."""
from python.feature_subsets.base import FeatureSubsets

WSN_FULL = list(range(1, 19))
WSN_RCL_GR = list(range(1, 19))

RCL_WSN_IWSSR_NaiveBayes    = [2, 3, 18]
RCL_WSN_IWSSR_RandomTree    = [2, 6, 8, 10, 14, 15, 17, 18]
RCL_WSN_IWSSR_J48            = [2, 4, 7, 8, 9, 10, 14, 15, 18]
RCL_WSN_IWSSR_RandomForest   = [2, 4, 6, 7, 10, 14, 15, 16, 17, 1]
RCL_WSN_IWSSR_RepTree        = [2, 4, 6, 10, 11, 14, 15, 18]
RCL_WSN_IWSSR = [
    RCL_WSN_IWSSR_RandomTree,
    RCL_WSN_IWSSR_J48,
    RCL_WSN_IWSSR_RepTree,
    RCL_WSN_IWSSR_NaiveBayes,
    RCL_WSN_IWSSR_RandomForest,
]
WSN_RCL_I = RCL_WSN_IWSSR

GR_G_BF_NaiveBayes_5    = [18, 8, 7, 13, 6]
GR_G_BF_RandomTree_5    = [9, 15, 3, 2, 6]
GR_G_BF_J48_5           = [9, 14, 6, 7, 15]
GR_G_BF_RandomForest_5  = [6, 9, 10, 18, 15]
GR_G_BF_RepTree_5       = [10, 3, 7, 13, 15]
GR_G_BF = [GR_G_BF_RandomTree_5, GR_G_BF_J48_5, GR_G_BF_RepTree_5, GR_G_BF_NaiveBayes_5, GR_G_BF_RandomForest_5]

GR_G_RVND_RandomTree    = [9, 15, 6, 3, 2]
GR_G_RVND_J48           = [4, 3, 18, 9, 15, 14, 11, 10, 7, 6, 2]
GR_G_RVND_RepTree       = [3, 10, 15, 14, 6, 4, 2]
GR_G_RVND_NaiveBayes    = [6, 13, 7, 8]
GR_G_RVND_RandomForest  = [6, 9, 10, 18, 15]
GR_G_RVND = [GR_G_RVND_RandomTree, GR_G_RVND_J48, GR_G_RVND_RepTree, GR_G_RVND_NaiveBayes, GR_G_RVND_RandomForest]

GR_G_VND_RandomTree    = [9, 15, 6, 3, 2]
GR_G_VND_J48           = [4, 3, 18, 9, 15, 14, 11, 10, 7, 6, 2]
GR_G_VND_RepTree       = [3, 10, 15, 14, 6, 4, 2]
GR_G_VND_NaiveBayes    = [7, 6, 8, 13]
GR_G_VND_RandomForest  = [6, 9, 7, 10, 18]
GR_G_VND = [GR_G_VND_RandomTree, GR_G_VND_J48, GR_G_VND_RepTree, GR_G_VND_NaiveBayes, GR_G_VND_RandomForest]

I_G_VND_RandomTree    = [2, 3, 15, 9, 6]
I_G_VND_J48           = [4, 3, 18, 9, 15, 14, 11, 10, 7, 6, 2]
I_G_VND_RepTree       = [10, 15, 14, 11, 6, 4, 2]
I_G_VND_NaiveBayes    = [6, 13, 7, 8]
I_G_VND_RandomForest  = [18, 10, 7, 9, 6, 16, 15, 2]
I_G_VND = [I_G_VND_RandomTree, I_G_VND_J48, I_G_VND_RepTree, I_G_VND_NaiveBayes, I_G_VND_RandomForest]


class WsnFeatures(FeatureSubsets):
    def __init__(self):
        super().__init__(WSN_FULL, WSN_RCL_GR, WSN_RCL_I)
