# =============================================================================
# 
# 
# 
# # units: N,m
# 
# from openseespy.opensees import *
# import opsvis as opsv
# import os
# import math
# from ISection import ISection
# from HssSection import HssSection
# import matplotlib.pyplot as plt
# import vfo.vfo as vfo
# # import BraineryWiz as bz
# from crossJoint import crossJoint
# from rectangleJoint import rectangleJoint
# from krawinklerJointMat import krawinklerJointMat
# from scissorsJointMat import scissorsJointMat
# from scissorsJoint import scissorsJoint
# 
# wipe()
# 
# # model('basic', '-ndm', 2, '-ndf', 3)
# 
# # H = 3    # for train
# # # H = 6  # for predict
# # L = 3
# 
# model('basic', '-ndm', 2, '-ndf', 3)
# 
# # ---------------------------------
# # H و L را از متغیرهای محیطی بگیر؛
# # اگر تنظیم نشده بودند، مقدار پیش‌فرض بگذار
# # ---------------------------------
# import os
# 
# H_env = os.environ.get("H_COL")
# L_env = os.environ.get("L_BEAM")
# 
# if H_env is not None:
#     H = float(H_env)
# else:
#     H = 3.0   # مقدار پیش‌فرض، اگر از بیرون ندادیم
# 
# if L_env is not None:
#     L = float(L_env)
# else:
#     L = 3.0   # مقدار پیش‌فرض
# 
# 
# 
# 
# 
# db = 0.3023
# dc = 0.3556
# 
# # ---------------------------------
# # define nodes
# # ---------------------------------
# node(1, 0, 0)
# node(3, L, 0)
# 
# # left joint
# node(5, 0, H - db / 2.0)
# node(7, 0, H + db / 2.0)
# node(6, 0 + dc / 2.0, H)
# node(8, 0 - dc / 2.0, H)
# 
# # right joint
# node(9,  L, H - db / 2.0)
# node(11, L, H + db / 2.0)
# node(10, L + dc / 2.0, H)
# node(12, L - dc / 2.0, H)
# 
# # ---------------------------------
# # geometric transformation (ONLY Linear for linear model)
# # ---------------------------------
# linearTransfTag = 1
# geomTransf('Linear', linearTransfTag)
# 
# # no PDelta transformation in linear model
# # pDeltaTransfTag = 2
# # geomTransf('PDelta', pDeltaTransfTag)
# 
# # ---------------------------------
# # material (LINEAR ELASTIC)
# # ---------------------------------
# steelMatTag = 1
# Es = 2e11
# nus = 0.3
# Fy = 1.1 * 345e6
# hardeningRatio = 0.01
# uniaxialMaterial('Elastic', steelMatTag, Es)
# 
# # ---------------------------------
# # sections (fiber sections with elastic material)
# # ---------------------------------
# numSubdivL = 10
# numSubdivT = 3
# numIntgrPts = 5
# 
# # W12x40 - beam
# beamSecTag = 1
# d = 0.3023
# bf = 0.2035
# tf = 0.0131
# tw = 0.0075
# ABeam = 7548.4e-6
# IBeam = 127783047.7e-12
# fib_sec_beam = ISection(
#     beamSecTag, steelMatTag, d, tw, bf, tf,
#     numSubdivL, numSubdivT, numIntgrPts
# )
# # fib_sec_beam = HssSection(beamSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)
# 
# db = d
# tf_b = tf
# 
# # W14x68 - column
# colSecTag = 2
# d = 0.3556
# bf = 0.254
# tf = 0.0183
# tw = 0.0105
# ACol = 12903.2e-4
# ICol = 300519089.3e-12
# fib_sec_col = ISection(
#     colSecTag, steelMatTag, d, tw, bf, tf,
#     numSubdivL, numSubdivT, numIntgrPts
# )
# # fib_sec_col = HssSection(colSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)
# 
# dc = d
# tp = tw
# bf_c = bf
# tf_c = tf
# 
# # ---------------------------------
# # joint definition (RECTANGLE + RIGID behavior)
# # ---------------------------------
# jointShape = "rectangle"   # rectangle, cross, scissors
# jointBehavior = "rigid"    # for linear model: rigid
# 
# amplifyEle = 100
# amplifyHorizEleJoint = 100
# amplifyVerticalEleJoint = 100
# 
# ATypical = ACol
# ITypical = ICol
# KAxialTypical = ATypical * Es
# KRotTypical = ITypical * Es
# KRigid = 100 * max(KAxialTypical, KRotTypical)
# rigidMatTag = 10
# uniaxialMaterial('Elastic', rigidMatTag, KRigid)
# 
# if jointShape == "cross":
#     crossJoint(
#         10, 5, 6, 7, 8, 2, Es,
#         ABeam, IBeam, ACol, ICol,
#         linearTransfTag, amplifyHorizEleJoint, amplifyVerticalEleJoint
#     )
#     crossJoint(
#         20, 9, 10, 11, 12, 4, Es,
#         ABeam, IBeam, ACol, ICol,
#         linearTransfTag, amplifyHorizEleJoint, amplifyVerticalEleJoint
#     )
# 
# elif jointShape == "rectangle":
# 
#     if jointBehavior == "rigid":
#         # no nonlinear joint material (Krawinkler) in linear model
#         rotMatTag = rigidMatTag
# 
#     # rigid/elastic joints
#     rectangleJoint(
#         20, 20, 5, 6, 7, 8,
#         ATypical, ITypical, Es,
#         linearTransfTag, amplifyEle,
#         rigidMatTag, rotMatTag
#     )
#     rectangleJoint(
#         40, 40, 9, 10, 11, 12,
#         ATypical, ITypical, Es,
#         linearTransfTag, amplifyEle,
#         rigidMatTag, rotMatTag
#     )
# 
# elif jointShape == "scissors":
# 
#     if jointBehavior == "rigid":
#         rotMatTag = rigidMatTag
# 
#     scissorsJoint(
#         20, 20, 5, 6, 7, 8,
#         linearTransfTag, rigidMatTag, rotMatTag,
#         Es, ATypical, ITypical, amplifyEle
#     )
#     scissorsJoint(
#         40, 40, 9, 10, 11, 12,
#         linearTransfTag, rigidMatTag, rotMatTag,
#         Es, ATypical, ITypical, amplifyEle
#     )
# 
# rhoSteel = 7810.0
# 
# # ---------------------------------
# # beam and columns (elasticBeamColumn, LINEAR)
# # ---------------------------------
# element(
#     'elasticBeamColumn', 1, 1, 5,
#     ACol, Es, ICol, linearTransfTag,
#     '-mass', ACol * rhoSteel
# )
# element(
#     'elasticBeamColumn', 2, 3, 9,
#     ACol, Es, ICol, linearTransfTag,
#     '-mass', ACol * rhoSteel
# )
# element(
#     'elasticBeamColumn', 3, 6, 12,
#     ABeam, Es, IBeam, linearTransfTag,
#     '-mass', ABeam * rhoSteel
# )
# 
# # previous nonlinear elements (commented)
# # element('forceBeamColumn', 1, 1, 5, pDeltaTransfTag, colSecTag, '-mass', ACol*rhoSteel)
# # element('forceBeamColumn', 2, 3, 9, pDeltaTransfTag, colSecTag, '-mass', ACol*rhoSteel)
# # element('forceBeamColumn', 3, 6, 12, pDeltaTransfTag, beamSecTag, '-mass', ABeam*rhoSteel)
# 
# # ---------------------------------
# # constraints
# # ---------------------------------
# fix(1, 1, 1, 1)
# fix(3, 1, 1, 1)
# 
# # ---------------------------------
# # masses
# # ---------------------------------
# mass(7, 9174.3, 9174.3, 1e-12)
# mass(11, 9174.3, 9174.3, 1e-12)
# 
# g = 9.81
# seismicWeight = 2 * 9174.3 * g
# 
# # ---------------------------------
# # eigen analysis (LINEAR)
# # ---------------------------------
# omega2List = eigen(2)
# 
# i = 0
# for omega2 in omega2List:
#     omega = math.sqrt(omega2)
#     T = 2.0 * math.pi / omega
#     print(f'T({i+1}) = {T}')
#     if i == 0:
#         T1 = T
#     i += 1
# 
# # ---------------------------------
# # gravity loads
# # ---------------------------------
# linearSeriesTag = 1
# timeSeries('Linear', linearSeriesTag)
# pattern('Plain', 1, linearSeriesTag)
# load(7, 0.0, -30e3, 0.0)
# load(11, 0.0, -30e3, 0.0)
# 
# loadPerLength = 20e3   # N/m
# eleLoad('-ele', 3, '-type', '-beamUniform', -loadPerLength)
# 
# # ---------------------------------
# # gravity analysis (LINEAR STATIC)
# # ---------------------------------
# wipeAnalysis()
# constraints('Transformation')
# numberer('RCM')
# system('BandGen')
# test('NormDispIncr', 1e-6, 100)
# algorithm('Linear')
# integrator('LoadControl', 0.1)
# analysis('Static')
# analyze(10)
# loadConst('-time', 0.0)
# 
# # ---------------------------------
# # optional plotting
# # ---------------------------------
# displayModel = 0
# displayModeShape = 0
# 
# if displayModel:
#     opsv.plot_model(
#         node_labels=1,
#         element_labels=1,
#         local_axes=True,
#         gauss_points=False
#     )
#     vfo.plot_model(
#         show_nodes='yes',
#         show_nodetags='yes',
#         show_eletags='yes',
#         font_size=10
#     )
# 
# if displayModeShape:
#     modeNumber = 1
#     opsv.plot_mode_shape(modeNumber, sfac=False, unDefoFlag=1)
#     vfo.plot_modeshape(modenumber=modeNumber, scale=100)
# 
# 
# =============================================================================








# units: N,m

from openseespy.opensees import *
import opsvis as opsv
import os
import math
from ISection import ISection
from HssSection import HssSection
import matplotlib.pyplot as plt
import vfo.vfo as vfo
# import BraineryWiz as bz
from crossJoint import crossJoint
from rectangleJoint import rectangleJoint
from krawinklerJointMat import krawinklerJointMat
from scissorsJointMat import scissorsJointMat
from scissorsJoint import scissorsJoint

wipe()

model('basic', '-ndm', 2, '-ndf', 3)

# ---------------------------------
# H و L را از متغیرهای محیطی بگیر؛
# اگر تنظیم نشده باشند، مقدار پیش‌فرض بگذار
# ---------------------------------
H_env = os.environ.get("H_COL")
L_env = os.environ.get("L_BEAM")

if H_env is not None:
    H = float(H_env)
else:
    H = 3.0   # مقدار پیش‌فرض، اگر از بیرون ندادیم

if L_env is not None:
    L = float(L_env)
else:
    L = 3.0   # مقدار پیش‌فرض


db = 0.3023
dc = 0.3556

# ---------------------------------
# define nodes
# ---------------------------------
node(1, 0, 0)
node(3, L, 0)

# left joint
node(5, 0, H - db / 2.0)
node(7, 0, H + db / 2.0)
node(6, 0 + dc / 2.0, H)
node(8, 0 - dc / 2.0, H)

# right joint
node(9,  L, H - db / 2.0)
node(11, L, H + db / 2.0)
node(10, L + dc / 2.0, H)
node(12, L - dc / 2.0, H)

# ---------------------------------
# geometric transformation (ONLY Linear for linear model)
# ---------------------------------
linearTransfTag = 1
geomTransf('Linear', linearTransfTag)

# no PDelta transformation in linear model
# pDeltaTransfTag = 2
# geomTransf('PDelta', pDeltaTransfTag)

# ---------------------------------
# material (LINEAR ELASTIC)
# ---------------------------------
steelMatTag = 1
Es = 2e11
nus = 0.3
Fy = 1.1 * 345e6
hardeningRatio = 0.01
uniaxialMaterial('Elastic', steelMatTag, Es)

# ---------------------------------
# sections (fiber sections with elastic material)
# ---------------------------------
numSubdivL = 10
numSubdivT = 3
numIntgrPts = 5

# W12x40 - beam
beamSecTag = 1
d = 0.3023
bf = 0.2035
tf = 0.0131
tw = 0.0075
ABeam = 7548.4e-6
IBeam = 127783047.7e-12
fib_sec_beam = ISection(
    beamSecTag, steelMatTag, d, tw, bf, tf,
    numSubdivL, numSubdivT, numIntgrPts
)
# fib_sec_beam = HssSection(beamSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)

db = d
tf_b = tf

# W14x68 - column
colSecTag = 2
d = 0.3556
bf = 0.254
tf = 0.0183
tw = 0.0105
ACol = 12903.2e-4
ICol = 300519089.3e-12
fib_sec_col = ISection(
    colSecTag, steelMatTag, d, tw, bf, tf,
    numSubdivL, numSubdivT, numIntgrPts
)
# fib_sec_col = HssSection(colSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)

dc = d
tp = tw
bf_c = bf
tf_c = tf

# ---------------------------------
# joint definition (RECTANGLE + RIGID behavior)
# ---------------------------------
jointShape = "rectangle"   # rectangle, cross, scissors
jointBehavior = "rigid"    # for linear model: rigid

amplifyEle = 100
amplifyHorizEleJoint = 100
amplifyVerticalEleJoint = 100

ATypical = ACol
ITypical = ICol
KAxialTypical = ATypical * Es
KRotTypical = ITypical * Es
KRigid = 100 * max(KAxialTypical, KRotTypical)
rigidMatTag = 10
uniaxialMaterial('Elastic', rigidMatTag, KRigid)

if jointShape == "cross":
    crossJoint(
        10, 5, 6, 7, 8, 2, Es,
        ABeam, IBeam, ACol, ICol,
        linearTransfTag, amplifyHorizEleJoint, amplifyVerticalEleJoint
    )
    crossJoint(
        20, 9, 10, 11, 12, 4, Es,
        ABeam, IBeam, ACol, ICol,
        linearTransfTag, amplifyHorizEleJoint, amplifyVerticalEleJoint
    )

elif jointShape == "rectangle":

    if jointBehavior == "rigid":
        # no nonlinear joint material (Krawinkler) in linear model
        rotMatTag = rigidMatTag

    # rigid/elastic joints
    rectangleJoint(
        20, 20, 5, 6, 7, 8,
        ATypical, ITypical, Es,
        linearTransfTag, amplifyEle,
        rigidMatTag, rotMatTag
    )
    rectangleJoint(
        40, 40, 9, 10, 11, 12,
        ATypical, ITypical, Es,
        linearTransfTag, amplifyEle,
        rigidMatTag, rotMatTag
    )

elif jointShape == "scissors":

    if jointBehavior == "rigid":
        rotMatTag = rigidMatTag

    scissorsJoint(
        20, 20, 5, 6, 7, 8,
        linearTransfTag, rigidMatTag, rotMatTag,
        Es, ATypical, ITypical, amplifyEle
    )
    scissorsJoint(
        40, 40, 9, 10, 11, 12,
        linearTransfTag, rigidMatTag, rotMatTag,
        Es, ATypical, ITypical, amplifyEle
    )

rhoSteel = 7810.0

# ---------------------------------
# beam and columns (elasticBeamColumn, LINEAR)
# ---------------------------------
element(
    'elasticBeamColumn', 1, 1, 5,
    ACol, Es, ICol, linearTransfTag,
    '-mass', ACol * rhoSteel
)
element(
    'elasticBeamColumn', 2, 3, 9,
    ACol, Es, ICol, linearTransfTag,
    '-mass', ACol * rhoSteel
)
element(
    'elasticBeamColumn', 3, 6, 12,
    ABeam, Es, IBeam, linearTransfTag,
    '-mass', ABeam * rhoSteel
)

# previous nonlinear elements (commented)
# element('forceBeamColumn', 1, 1, 5, pDeltaTransfTag, colSecTag, '-mass', ACol*rhoSteel)
# element('forceBeamColumn', 2, 3, 9, pDeltaTransfTag, colSecTag, '-mass', ACol*rhoSteel)
# element('forceBeamColumn', 3, 6, 12, pDeltaTransfTag, beamSecTag, '-mass', ABeam*rhoSteel)

# ---------------------------------
# constraints
# ---------------------------------
fix(1, 1, 1, 1)
fix(3, 1, 1, 1)

# ---------------------------------
# masses
# ---------------------------------
mass(7, 9174.3, 9174.3, 1e-12)
mass(11, 9174.3, 9174.3, 1e-12)

g = 9.81
seismicWeight = 2 * 9174.3 * g

# ---------------------------------
# eigen analysis (LINEAR)
# ---------------------------------
omega2List = eigen(2)

i = 0
for omega2 in omega2List:
    omega = math.sqrt(omega2)
    T = 2.0 * math.pi / omega
    print(f'T({i+1}) = {T}')
    if i == 0:
        T1 = T
    i += 1

# ---------------------------------
# gravity loads
# ---------------------------------
linearSeriesTag = 1
timeSeries('Linear', linearSeriesTag)
pattern('Plain', 1, linearSeriesTag)
load(7, 0.0, -30e3, 0.0)
load(11, 0.0, -30e3, 0.0)

loadPerLength = 20e3   # N/m
eleLoad('-ele', 3, '-type', '-beamUniform', -loadPerLength)

# ---------------------------------
# gravity analysis (LINEAR STATIC)
# ---------------------------------
wipeAnalysis()
constraints('Transformation')
numberer('RCM')
system('BandGen')
test('NormDispIncr', 1e-6, 100)
algorithm('Linear')
integrator('LoadControl', 0.1)
analysis('Static')
analyze(10)
loadConst('-time', 0.0)

# ---------------------------------
# optional plotting
# ---------------------------------
displayModel = 0
displayModeShape = 0

if displayModel:
    opsv.plot_model(
        node_labels=1,
        element_labels=1,
        local_axes=True,
        gauss_points=False
    )
    vfo.plot_model(
        show_nodes='yes',
        show_nodetags='yes',
        show_eletags='yes',
        font_size=10
    )

if displayModeShape:
    modeNumber = 1
    opsv.plot_mode_shape(modeNumber, sfac=False, unDefoFlag=1)
    vfo.plot_modeshape(modenumber=modeNumber, scale=100)











