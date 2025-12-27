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
import math

nModeDamp = 2
zetaDamp = 0.05

omegaI = math.sqrt(omega2List[0])
omegaJ = math.sqrt(omega2List[nModeDamp-1])
alphaM = 2*zetaDamp*omegaI*omegaJ/(omegaI+omegaJ)
betaK = 2*zetaDamp/(omegaI+omegaJ)
rayleigh(alphaM, 0., 0., betaK)




