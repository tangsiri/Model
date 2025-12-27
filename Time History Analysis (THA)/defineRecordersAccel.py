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

recorder('Node', '-file', f'{dataDir}/accel.txt', '-time',  '-timeSeries', seriesTag, '-node', 7, '-dof', 1, 'accel')
recorder('EnvelopeNode', '-file', f'{dataDir}/maxAccel.txt', '-timeSeries', seriesTag, '-node', 7, '-dof', 1, 'accel')








