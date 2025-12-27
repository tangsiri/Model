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

def HssSection (secTag, matTag, d, tw, bf, tf, numSubdivL, numSubdivT, numIntgrPts):
	
    y1 = -d/2.
    y2 = -d/2. + tf
    y3 = d/2. - tf
    y4 = d/2.

    z1 = -bf/2.
    z2 = -bf/2+tw
    z3 = bf/2-tw
    z4 = bf/2.

    section('Fiber', secTag)
    patch ('quad', matTag, numSubdivL, numSubdivT, y1, z4, y1, z1, y2, z1, y2, z4)
    patch ('quad', matTag, numSubdivT, numSubdivL, y2, z2, y2, z1, y3, z1, y3, z2)
    patch ('quad', matTag, numSubdivL, numSubdivT, y3, z4, y3, z1, y4, z1, y4, z4)
    patch ('quad', matTag, numSubdivT, numSubdivL, y2, z4, y2, z3, y3, z3, y3, z4)
    
    beamIntegration('Lobatto', secTag, secTag, numIntgrPts)

    # to display section
    fib_sec_list = [['section', 'Fiber', secTag],
             ['patch', 'quad', matTag, numSubdivL, numSubdivT, y1, z4, y1, z1, y2, z1, y2, z4],
             ['patch', 'quad', matTag, numSubdivT, numSubdivL, y2, z2, y2, z1, y3, z1, y3, z2],
             ['patch', 'quad', matTag, numSubdivL, numSubdivT, y3, z4, y3, z1, y4, z1, y4, z4],
             ['patch', 'quad', matTag, numSubdivT, numSubdivL, y2, z4, y2, z3, y3, z3, y3, z4],
             ]
    
    return fib_sec_list
