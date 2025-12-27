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


# nd1 to nd4 are counter-clockwise starting from bottom
# nd1: bot
# nd2: right
# nd3: top
# nd4: left
from openseespy.opensees import *


def scissorsJoint(firstEleTag, firstNodeTag, nd1, nd2, nd3, nd4, transfTag, rigidMatTag, rotMatTag, E, ATypical, ITypical, amplifyEle=100):
    
    x1 = nodeCoord(nd1, 1)
    y1 = nodeCoord(nd1, 2)
    
    x2 = nodeCoord(nd2, 1)
    y2 = nodeCoord(nd2, 2)
    
    x3 = nodeCoord(nd3, 1)
    y3 = nodeCoord(nd3, 2)
    
    x4 = nodeCoord(nd4, 1)
    y4 = nodeCoord(nd4, 2)
    
    xMid = (x2+x4)/2
    yMid = (y1+y3)/2
    
    midNodeTagHoriz = firstNodeTag+0
    midNodeTagVertical = firstNodeTag+1
    node(midNodeTagHoriz, xMid, yMid)
    node(midNodeTagVertical, xMid, yMid)
    
    eleId = firstEleTag
    # vertical
    element('elasticBeamColumn', eleId+0, midNodeTagVertical, nd1, ATypical, E*amplifyEle, ITypical, transfTag)
    element('elasticBeamColumn', eleId+1, midNodeTagVertical, nd3, ATypical, E*amplifyEle, ITypical, transfTag)
    
    # horizental
    element('elasticBeamColumn', eleId+2, midNodeTagHoriz, nd2, ATypical, E*amplifyEle, ITypical, transfTag)
    element('elasticBeamColumn', eleId+3, midNodeTagHoriz, nd4, ATypical, E*amplifyEle, ITypical, transfTag)
    
    # middle spring
    rotEleTag = eleId+4
    element ('zeroLength', rotEleTag, midNodeTagHoriz, midNodeTagVertical, '-mat', rigidMatTag, rigidMatTag, rotMatTag, '-dir', 1, 2, 3)
    
    
    