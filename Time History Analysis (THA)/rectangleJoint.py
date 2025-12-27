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

# nd1 to nd4 are counter-clockwise starting from bottom
# nd1: bot
# nd2: right
# nd3: top
# nd4: left

def rectangleJoint (firstEleTag, firstNodeTag, nd1, nd2, nd3, nd4, ATypical, ITypical, E, transfTag, amplifyEle, rigidMatTag, rotMatTag):
	
    eleId = firstEleTag

    xMin = nodeCoord(nd4)[0]
    xMax = nodeCoord(nd2)[0]
    yMin = nodeCoord(nd1)[1]
    yMax = nodeCoord(nd3)[1]

    nd5  = firstNodeTag + 0
    nd6  = firstNodeTag + 1
    nd7  = firstNodeTag + 2
    nd8  = firstNodeTag + 3
    nd9  = firstNodeTag + 4
    nd10 = firstNodeTag + 5
    nd11 = firstNodeTag + 6
    nd12 = firstNodeTag + 7
    node (nd5 , xMin, yMin)
    node (nd6 , xMin, yMin)
    node (nd7 , xMax, yMin)
    node (nd8 , xMax, yMin)
    node (nd9 , xMax, yMax)
    node (nd10, xMax, yMax)
    node (nd11, xMin, yMax)
    node (nd12, xMin, yMax)

    element ('elasticBeamColumn', eleId+0, nd6 ,  nd1 , ATypical, amplifyEle*E, ITypical, transfTag)
    element ('elasticBeamColumn', eleId+1, nd1 ,  nd7 , ATypical, amplifyEle*E, ITypical, transfTag)
    element ('elasticBeamColumn', eleId+2, nd8 ,  nd2 , ATypical, amplifyEle*E, ITypical, transfTag)
    element ('elasticBeamColumn', eleId+3, nd2 ,  nd9 , ATypical, amplifyEle*E, ITypical, transfTag)
    element ('elasticBeamColumn', eleId+4, nd3 ,  nd10, ATypical, amplifyEle*E, ITypical, transfTag)
    element ('elasticBeamColumn', eleId+5, nd11,  nd3 , ATypical, amplifyEle*E, ITypical, transfTag)
    element ('elasticBeamColumn', eleId+6, nd4 ,  nd12, ATypical, amplifyEle*E, ITypical, transfTag)
    element ('elasticBeamColumn', eleId+7, nd5 ,  nd4 , ATypical, amplifyEle*E, ITypical, transfTag)

    rotEleTag = eleId+11

    element ('zeroLength', eleId+8,   nd5, nd6,  '-mat', rigidMatTag, rigidMatTag, '-dir', 1, 2)
    element ('zeroLength', eleId+9,   nd7, nd8,  '-mat', rigidMatTag, rigidMatTag, '-dir', 1, 2)
    element ('zeroLength', eleId+10,  nd11, nd12, '-mat', rigidMatTag, rigidMatTag, '-dir', 1, 2)
    element ('zeroLength', rotEleTag, nd9,nd10, '-mat', rigidMatTag, rigidMatTag, rotMatTag, '-dir', 1, 2, 3)

    return rotEleTag

