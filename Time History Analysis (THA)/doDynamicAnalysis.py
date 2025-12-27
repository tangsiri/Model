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
import os

def doDynamicAnalysis(Tmax, dtInput, logFileDir = 0, tolInitial = 1e-6, tolMax = 1e-6, systemName = 'BandGeneral'):
    
    cwd = os.getcwd()
    logFileDir = cwd + '/logTHA.txt'
    
    if logFileDir != 0:
        fileId = open(logFileDir, 'w')
    
    
    # settings
    minStepRatio = 1e-3
    maxNumIter = 100
    smallProtionTime = 0.1
    
    # tests and algorithms
    testList = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 3:'EnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algorithmList = {1:'KrylovNewton', 2: 'SecantNewton' , 3:'ModifiedNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}
    
    # analysis parameters
    wipeAnalysis()
    constraints('Transformation')
    numberer('RCM')
    system(systemName)

    # parametrs for first analysis
    testName = testList[1]
    algorithmName = algorithmList[1]
    
    # convergence code
    failureFlag = 0
    firstTry = 1
    TCurrent = getTime()
    remainedTime = Tmax - TCurrent
    
    # outer loop: this is intended to perform major of analysis
    while remainedTime > dtInput and failureFlag == 0:
        
        # set default parameters
        tol = tolInitial
        dtAnalysis = dtInput
        
        # perform analysis
        if logFileDir != 0:
            fileId.write("\n\nRunning: test: %s   algorithm: %s   dtAnalysis: %.6f   tol: %.10f   TCurrent: %.2f   remainedTime: %.2f\n" %(testName, algorithmName, dtAnalysis, tol, TCurrent, remainedTime))
        
        nSteps = int(remainedTime/dtAnalysis)+1
        test(testName, tol, maxNumIter)
        algorithm(algorithmName)
        integrator('Newmark', 0.5, 0.25) 
        analysis('Transient')
        analyze(nSteps,dtAnalysis)
        
        # check remainedTime
        TCurrent = getTime()
        remainedTime = Tmax - TCurrent

        while remainedTime > dtInput and failureFlag == 0:
            firstTry = 0
            
            for i in testList:
                for j in algorithmList: 
                    algorithmName = algorithmList[j]
                    
                    if j < 4:
                        algorithm(algorithmName, '-initial')
                    else:
                        algorithm(algorithmName)
                    
                    testName = testList[i]
                    test(testName, tol, maxNumIter)        
                    integrator('Newmark', 0.5, 0.25)
                    analysis('Transient')
                    nSteps = int(smallProtionTime/dtAnalysis)+1
                    
                    if logFileDir != 0:
                        fileId.write("Running: test: %s   algorithm: %s   dtAnalysis: %.6f   tol: %.10f   TCurrent: %.2f   remainedTime: %.2f\n" %(testName, algorithmName, dtAnalysis, tol, TCurrent, remainedTime))
                    
                    ok = analyze(nSteps,dtAnalysis)
                    
                    if ok == 0:
                        TCurrent = getTime()
                        remainedTime = Tmax - TCurrent
                        break
                
                if ok == 0:
                    break
            if ok == 0:
                break
            else:
                dtAnalysis = dtAnalysis/2
                tol = min(tol*3,tolMax)
                
                if dtAnalysis/dtInput < minStepRatio:
                    failureFlag = 1
                    break
    
    if failureFlag == 0:
        print("---------- Analysis successful ----------")
        
        if logFileDir != 0:
            fileId.write("---------- Analysis successful ----------\n")
    else:
        print("!!!!!!!!!!!! Analysis Interrupted !!!!!!!!!!!!")
        if logFileDir != 0:
            fileId.write("!!!!!!!!!!!! Analysis Interrupted !!!!!!!!!!!!\n")
    fileId.close()
        