#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("../tools/")

def Draw2D(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

def Draw3D(features, poi, mark_poi = False, name = "3Dimage.png", f1_name = "feature 1", f2_name = "feature 2", f3_name = "feature 3"):
    """ some plotting code designed to help you visualize your data """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    
    for ii, pp in zip(features, poi):
    ### if you like, place red stars over points that are POIs (just for funsies)
        if mark_poi and pp:
            ax.scatter(ii[0], ii[1], ii[2], color="r", marker="*") 
        else:
            ax.scatter(ii[0], ii[1], ii[2])
                
    ax.set_xlabel(f1_name)
    ax.set_ylabel(f2_name)
    ax.set_zlabel(f3_name)
    plt.savefig(name)
    plt.show()