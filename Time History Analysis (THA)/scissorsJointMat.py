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

# tp: column web thickness + doubler palte thickness
def scissorsJointMat (matTag, db, tf_b, dc, tp, bf_c, tf_c, ACol, Es, nus, Fy, Pg, HStory, LBay):

    # calculate Vy and Vp
    Py = ACol*Fy
    Pr = Pg

    # calculate Vy
    Vy = 0.6*Fy*dc*tp
    if Pr <= 0.4*Py:
        Vy = Vy
    else:
        Vy = Vy*(1.4-Pr/Py)


    # calculate Vp
    Vp = 0.6*Fy*dc*tp*(1.+3.*bf_c*tf_c**2/(db*dc*tp))
    if Pr <= 0.75*Py:
        Vp = Vp

    else:
        Vp = Vp*(1.9-1.2*Pr/Py)

    # calculate theta and M
    G = Es/(2.*(1.+nus))
    Apz = tp*dc
    Ke = G*Apz

    gammaY = Vy/Ke
    My = Vy*(db-tf_b)/(1.-db/HStory-dc/LBay)
    KRot = Ke*(db-tf_b)/(1.-db/HStory-dc/LBay)**2
    thetaY = My/KRot

    gammaP = 4*gammaY
    thetaP = 4.*thetaY
    Mp = Vp*(db-tf_b)/(1.-db/HStory-dc/LBay)

    gammaU = 100.*gammaY
    thetaU = 100.*thetaY
    as0 = 0.02
    Vu = Vp + as0*Ke*(gammaU-gammaP)
    Mu = Vu*(db-tf_b)/(1.-db/HStory-dc/LBay)

    uniaxialMaterial('Hysteretic',matTag, My, thetaY, Mp, thetaP, Mu, thetaU, -My, -thetaY, -Mp, -thetaP, -Mu, -thetaU, 1, 1, 0., 0., 0.)

