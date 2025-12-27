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

import numpy as np
import matplotlib.pyplot as plt

def drawPlot(dataDir, xFileName, yFileName, iColX, iColY, xLabel, yLabel, title="", xCoeff=1, yCoeff=1, separateFig=1):
    
    # ------ extract x values -------
    data = np.loadtxt(f"{dataDir}/{xFileName}")
    if iColX == 0:
        xValues = xCoeff*np.sum(data,1)
    else:
        iColX = iColX-1
        xValues = xCoeff*data[:,iColX]
    
    
    # ------ extract y values -------
    myPath = f"{dataDir}/{yFileName}"
    data = np.loadtxt(myPath)
    if iColY == 0:
        yValues = yCoeff*np.sum(data,1)
    else:
        iColY = iColY-1
        yValues = yCoeff*data[:,iColY]
    
    
    # ------ draw plot -------
    if separateFig:
        fig1 = plt.figure()
    plt.plot(xValues, yValues)
    
    plt.xlabel(xLabel) 
    plt.ylabel(yLabel) 
    plt.title(title) 
    
    if separateFig:
        plt.show()


def drawMultiplePlot(dataDir, xFileNameList, yFileNameList, legendList, iColX, iColY, xLabel, yLabel, title="", xCoeff=1, yCoeff=1, legendLoc="best"):
    
    nPlots = len(xFileNameList)
    
    fig1 = plt.figure()
    
    for i in range(nPlots):
        xFileName = xFileNameList[i]
        yFileName = yFileNameList[i]
        drawPlot(dataDir, xFileName, yFileName, iColX, iColY, xLabel, yLabel, title, xCoeff, yCoeff, 0)
    
    plt.legend(legendList, loc =legendLoc)
    plt.show()
