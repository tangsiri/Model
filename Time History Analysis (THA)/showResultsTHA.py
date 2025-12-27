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

from drawPlot import drawPlot, drawMultiplePlot

iRec = 3

dataDir = f"outputs/THA/{iRec}"
xFileNameList = ["disp.txt"] 
yFileNameList = ["disp.txt"] 
iColX, iColY = 1, 2
xLabel, yLabel = "Time (sec)", "disp (mm)"
title= ""
xCoeff, yCoeff = 1, 1000
legendList = []
drawMultiplePlot(dataDir, xFileNameList, yFileNameList, legendList, iColX, iColY, xLabel, yLabel, title, xCoeff, yCoeff, legendLoc="best")


dataDir = f"outputs/THA/{iRec}"
xFileNameList = ["disp.txt"] 
# yFileNameList = ["Vx.txt"] 
iColX, iColY = 2, 0
xLabel, yLabel = "disp (mm)", "bease shear (KN)"
title= ""
xCoeff, yCoeff = 1000, 0.001
legendList = []
drawMultiplePlot(dataDir, xFileNameList, yFileNameList, legendList, iColX, iColY, xLabel, yLabel, title, xCoeff, yCoeff, legendLoc="best")







