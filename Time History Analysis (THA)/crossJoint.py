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


def crossJoint(firstEleTag, nd1, nd2, nd3, nd4, midNodeTag, E, ABeam, IBeam, ACol, ICol, transfTag, amplifyHorizEleJoint=1, amplifyVerticalEleJoint=1):
    
    
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
    
    node(midNodeTag, xMid, yMid)
    
    eleTag = firstEleTag
    # vertical
    element('elasticBeamColumn', eleTag+0, midNodeTag, nd1, ACol, E*amplifyVerticalEleJoint, ICol, transfTag)
    element('elasticBeamColumn', eleTag+1, midNodeTag, nd3, ACol, E*amplifyVerticalEleJoint, ICol, transfTag)
    # horizental
    element('elasticBeamColumn', eleTag+2, midNodeTag, nd2, ABeam, E*amplifyHorizEleJoint, IBeam, transfTag)
    element('elasticBeamColumn', eleTag+3, midNodeTag, nd4, ABeam, E*amplifyHorizEleJoint, IBeam, transfTag)
    
    
    