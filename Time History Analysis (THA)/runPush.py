
from openseespy.opensees import *
import os
import numpy as np
from analyzeAndAnimate import analyzeAndAnimatePush
import vfo.vfo as vfo
# import BraineryWiz as bz

dataDir = "outputs/pushover-rigid"
if not os.path.exists(dataDir):
    os.makedirs(dataDir)


exec(open("model.py").read())
exec(open("defineRecorders.py").read())



linearSeriesTag = 2
timeSeries('Linear', linearSeriesTag)
pattern('Plain', 2, linearSeriesTag)
load(7, 0.5*seismicWeight, 0, 0)
load(11, 0.5*seismicWeight, 0, 0)


# modelName = "steelModel"
# loadCaseName = "pushover"
# Nmodes = 2
# vfo.createODB(model= modelName, loadcase= loadCaseName, Nmodes= Nmodes)

# eleNumber = 1
# sectionNumber = 1
# vfo.saveFiberData2D(modelName, loadCaseName, eleNumber, sectionNumber)

controlNodeTag = 7
incr = 0.001
targetDisp = 0.3
dof = 1
nSteps = int(targetDisp/incr)


wipeAnalysis()
constraints('Transformation')
numberer('RCM')
system('BandGen')
test('NormDispIncr', 1e-6, 100)
algorithm('Newton')
integrator('DisplacementControl', controlNodeTag, dof, incr)
analysis('Static')
analyze(nSteps)

# for i in range(nSteps):
#     analyze(1)
#     bz.Record()
    
# bz.PlotAnime (plotmode=3, dt=0.01, scale_factor=10) 


# if showAnimationDeform == 0:
#     analyze(nSteps)
# else:
#     xlim=[-1, L+1]; ylim=[-1, H+1]
#     analyzeAndAnimate(nSteps, xlim, ylim, amplifyDeform=5, speedup=10)


wipe()


# vfo.plot_deformedshape(model=modelName, loadcase=loadCaseName, scale=10)
# vfo.animate_deformedshape(model=modelName, loadcase=loadCaseName, scale=10, speedup=10, gifname="pushAnimation")
# vfo.plot_fiberResponse2D(modelName, loadCaseName, eleNumber, sectionNumber, LocalAxis='y', InputType='stress')    
    
# anim = vfo.animate_fiberResponse2D(modelName, loadCaseName, eleNumber, sectionNumber,
# LocalAxis='y', InputType='stress')
    

# bz.PlotDefo(plotmode=3, scale_factor=2, draw_wire_shadow=False)





