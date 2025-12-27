
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

# model('basic', '-ndm', 2, '-ndf', 3)

# H = 3    #for train
# # H = 6  #for predict
# L = 3

model('basic', '-ndm', 2, '-ndf', 3)

# ---------------------------------
# H و L از متغیرهای محیطی (اگر تنظیم شده باشند)
# ---------------------------------
import os

H_env = os.environ.get("H_COL")
L_env = os.environ.get("L_BEAM")

if H_env is not None:
    H = float(H_env)
else:
    H = 3.0   # مقدار پیش‌فرض

if L_env is not None:
    L = float(L_env)
else:
    L = 3.0   # مقدار پیش‌فرض





db = 0.3023
dc = 0.3556


# define nodes
node(1, 0, 0)
node(3, L, 0)

# left joint
node(5, 0, H-db/2)
node(7, 0, H+db/2)
node(6, 0+dc/2, H)
node(8, 0-dc/2, H)

# right joint
node(9,  L, H-db/2)
node(11, L, H+db/2)
node(10, L+dc/2, H)
node(12, L-dc/2, H)



# define geometric transformation
linearTransfTag = 1
geomTransf('Linear', linearTransfTag)

pDeltaTransfTag = 2
geomTransf('PDelta', pDeltaTransfTag)


# define material
steelMatTag = 1
Fy = 1.1*345e6
Es = 2e11
hardeningRatio = 0.01
nus = 0.3
# R0 = 18; cR1 = 0.925; R2 = 0.15
parameters = [18, 0.925, 0.15]
uniaxialMaterial('Steel02', steelMatTag, Fy, Es, hardeningRatio, *parameters)

# define section
numSubdivL = 10
numSubdivT = 3
numIntgrPts = 5

# W12x40
beamSecTag = 1
d = 0.3023
bf = 0.2035
tf = 0.0131
tw = 0.0075
ABeam = 7548.4e-6
IBeam = 127783047.7e-12
fib_sec_beam = ISection(beamSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)
# fib_sec_beam = HssSection(beamSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)

db = d 
tf_b = tf 


# W14x68
colSecTag = 2
d = 0.3556
bf = 0.254
tf = 0.0183
tw = 0.0105
ACol = 12903.2e-4
ICol = 300519089.3e-12
fib_sec_col = ISection(colSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)
# fib_sec_col = HssSection(colSecTag, steelMatTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts)

dc = d 
tp = tw
bf_c = bf
tf_c = tf



# -------------------------
# define elements
# -------------------------

# define joint
jointShape = "rectangle"  # rectangle, cross, scissors
jointBehavior = "nonlinear" # nonlinear, rigid (rectangle, scissors)

amplifyEle = 100

amplifyHorizEleJoint = 100
amplifyVerticalEleJoint = 100


ATypical = ACol
ITypical = ICol
KAxialTypical = ATypical*Es
KRotTypical = ITypical*Es
KRigid = 100*max(KAxialTypical,KRotTypical)
rigidMatTag = 10
uniaxialMaterial('Elastic', rigidMatTag, KRigid)

if jointShape == "cross":
    crossJoint (10, 5, 6, 7, 8, 2, Es, ABeam, IBeam, ACol, ICol, linearTransfTag, amplifyHorizEleJoint, amplifyVerticalEleJoint)
    crossJoint (20, 9, 10, 11, 12, 4, Es, ABeam, IBeam, ACol, ICol, linearTransfTag, amplifyHorizEleJoint, amplifyVerticalEleJoint)

elif jointShape == "rectangle":
    
    if jointBehavior == "nonlinear":
        rotMatTag = 11
        Pg = 90e3
        krawinklerJointMat(rotMatTag, db, tf_b, dc, tp, bf_c, tf_c, ACol, Es, nus, Fy, Pg)
    elif jointBehavior == "rigid":
        rotMatTag = rigidMatTag
    
    rectangleJoint (20, 20, 5, 6, 7, 8, ATypical, ITypical, Es, linearTransfTag, amplifyEle, rigidMatTag, rotMatTag)
    rectangleJoint (40, 40, 9, 10, 11, 12, ATypical, ITypical, Es, linearTransfTag, amplifyEle, rigidMatTag, rotMatTag)
    
    # element('Joint2D', 20, 5, 6, 7, 8, 20, rotMatTag, 0)
    # element('Joint2D', 40, 9, 10, 11, 12, 40, rotMatTag, 0)

elif jointShape == "scissors":
    
    if jointBehavior == "nonlinear":
        rotMatTag = 11
        Pg = 90e3
        scissorsJointMat (rotMatTag, db, tf_b, dc, tp, bf_c, tf_c, ACol, Es, nus, Fy, Pg, H, L)
    
    elif jointBehavior == "rigid":
        rotMatTag = rigidMatTag
    
    scissorsJoint(20, 20, 5, 6, 7, 8, linearTransfTag, rigidMatTag, rotMatTag, Es, ATypical, ITypical, amplifyEle)
    scissorsJoint(40, 40, 9, 10, 11, 12, linearTransfTag, rigidMatTag, rotMatTag, Es, ATypical, ITypical, amplifyEle)

rhoSteel = 7810.

# define beam and columns
element('forceBeamColumn', 1, 1, 5, pDeltaTransfTag, colSecTag, '-mass', ACol*rhoSteel)
element('forceBeamColumn', 2, 3, 9, pDeltaTransfTag, colSecTag, '-mass', ACol*rhoSteel)
element('forceBeamColumn', 3, 6, 12, pDeltaTransfTag, beamSecTag, '-mass', ABeam*rhoSteel)

# element('elasticBeamColumn', 1, 1, 2, ACol, Es, ICol, transfTag) 
# element('elasticBeamColumn', 2, 3, 4, ACol, Es, ICol, transfTag) 
# element('elasticBeamColumn', 3, 2, 4, ABeam, Es, IBeam, transfTag) 

# define constraints
fix(1, 1, 1, 1)
fix(3, 1, 1, 1)


# define mass
mass(7, 9174.3, 9174.3, 1e-12)
mass(11, 9174.3, 9174.3, 1e-12)

g = 9.81
seismicWeight = 2*9174.3*g

# eigen analysis
omega2List = eigen(2)

i = 0
for omega2 in omega2List:
    omega = math.sqrt(omega2)
    T = 2*math.pi/omega
    print(f'T({i+1}) = {T}')
    
    if i == 0:
        T1 = T
    i += 1


# define gravity loads
linearSeriesTag = 1
timeSeries('Linear', linearSeriesTag)
pattern('Plain', 1, linearSeriesTag)
load(7, 0, -30e3, 0)
load(11, 0, -30e3, 0)

loadPerLength = 20e3   # N/m
eleLoad('-ele', 3, '-type', '-beamUniform', -loadPerLength)


# gravity analysis
wipeAnalysis()
constraints('Transformation')
numberer('RCM')
system('BandGen')
test('NormDispIncr', 1e-6, 100)
algorithm('Newton')
integrator('LoadControl', 0.1)
analysis('Static')
analyze(10)
loadConst('-time', 0)

# # display model # به دلیل اجرای زیاد نیازی به نمایش ندارم
displayModel = 0
displayModeShape = 0

if displayModel:
    opsv.plot_model(node_labels=1, element_labels=1, local_axes=True,  gauss_points=False)
    
    vfo.plot_model(show_nodes='yes', show_nodetags='yes', show_eletags='yes', font_size=10)
    
    # bz.PlotModel(plotmode=3, draw_nodes=True, show_nodes_tag=True, 
                  # show_elemens_tag=True, onhover_message=True,
                  # plot_integration_points=False,
                  # quivers_size=0.002, plot_fibers=True)
    
if displayModeShape:
    modeNumber = 1
    
    opsv.plot_mode_shape(modeNumber, sfac=False, unDefoFlag=1)
    vfo.plot_modeshape(modenumber=modeNumber, scale=100)
    # bz.PlotModeShape(plotmode=3, mode_number=modeNumber, draw_wire_shadow=False, scale_factor=100)
    anim = opsv.anim_mode(modeNumber, sfac=False, unDefoFlag=1, fig_wi_he=(30,20), xlim=[-1, L+1], ylim=[-1, H+1])

# # به دلیل اجرای زیاد نیازی به نمایش ندارم

# ------------------------------------
# opsvis
# -----------------------------------

# opsv.plot_defo(sfac=False, unDefoFlag=1)

# opsv.plot_loads_2d(sfac=False)

# secShapeDir = "outputs/secShape"
# if not os.path.exists(secShapeDir):
#     os.makedirs(secShapeDir)

# opsv.plot_fiber_section(fib_sec_beam)
# plt.axis('equal')
# plt.savefig(f'{secShapeDir}/beamSec.jpeg')

# opsv.plot_fiber_section(fib_sec_col)
# plt.axis('equal')
# plt.savefig(f'{secShapeDir}/colSec.jpeg')


# sfacN, sfacV, sfacM = 1.e-5, 1.e-5, 1.e-5

# opsv.section_force_diagram_2d('N', sfacN)
# plt.title('Axial force distribution')

# opsv.section_force_diagram_2d('T', sfacV)
# plt.title('Shear force distribution')

# opsv.section_force_diagram_2d('M', sfacM)
# plt.title('Bending moment distribution')

# plt.show()




