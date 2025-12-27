# ################################################################
# Please keep this notification at the beginning of the file
# 
# This code is part of an OpenSees course. 
# Only the person who purchases the course is allowed to 
# use it. Any distribution of this code is forbidden.
#
# developed by:
# 				Hadi Eslamnia
# 				Amirkabir University of Technology
# 
# contact us:
# 				Website: 			eslamnia.com
# 				Instagram: 			@eslamnia.ir
#				Telegram/Eitaa :	@eslamnia
# 				WhatsApp and call: 	+989101858874
#				Email :				opensees.eslamnia@gmail.com
# ################################################################

from openseespy.opensees import *
import numpy as np
import matplotlib.pyplot as plt
import opsvis as opsv

def analyzeAndAnimatePush(nSteps, xlim, ylim, amplifyDeform=10, speedup=1):
    
    el_tags = getEleTags()
    nels = len(el_tags)
    
    if nSteps % speedup == 0:
        n = int(nSteps/speedup)
    else:
        n = int(nSteps/speedup+1)
    
    Eds = np.zeros((n, nels, 6))
    timeV = np.zeros(n)
    
    ii = 0
    for step in range(nSteps):
        analyze(1)
        
        if (step+1) % speedup != 0 and step != nSteps-1:
            continue
        
        timeV[ii] = getTime()
        # collect disp for element nodes
        for el_i, ele_tag in enumerate(el_tags):
            nd1, nd2 = eleNodes(ele_tag)
            Eds[ii, el_i, :] = [nodeDisp(nd1)[0],
                                  nodeDisp(nd1)[1],
                                  nodeDisp(nd1)[2],
                                  nodeDisp(nd2)[0],
                                  nodeDisp(nd2)[1],
                                  nodeDisp(nd2)[2]]
    
        ii += 1

    fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
                'marker': '', 'markersize': 6}
    anim = opsv.anim_defo(Eds, timeV, amplifyDeform, fmt_defo=fmt_defo,
                          xlim=xlim, ylim=ylim, fig_wi_he=(30., 22.))
    
    plt.show()


def analyzeAndAnimateTHA(nSteps, dtAnalysis, xlim, ylim, amplifyDeform=10, speedup=1):
    
    el_tags = getEleTags()
    nels = len(el_tags)
    
    if nSteps % speedup == 0:
        n = int(nSteps/speedup)
    else:
        n = int(nSteps/speedup+1)
    
    Eds = np.zeros((n, nels, 6))
    timeV = np.zeros(n)
    
    ii = 0
    for step in range(nSteps):
        analyze(1, dtAnalysis)
        
        if (step+1) % speedup != 0 and step != nSteps-1:
            continue
        
        timeV[ii] = getTime()
        # collect disp for element nodes
        for el_i, ele_tag in enumerate(el_tags):
            nd1, nd2 = eleNodes(ele_tag)
            Eds[ii, el_i, :] = [nodeDisp(nd1)[0],
                                  nodeDisp(nd1)[1],
                                  nodeDisp(nd1)[2],
                                  nodeDisp(nd2)[0],
                                  nodeDisp(nd2)[1],
                                  nodeDisp(nd2)[2]]
    
        ii += 1

    fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
                'marker': '', 'markersize': 6}
    anim = opsv.anim_defo(Eds, timeV, amplifyDeform, fmt_defo=fmt_defo,
                          xlim=xlim, ylim=ylim, fig_wi_he=(30., 22.))
    
    plt.show()


