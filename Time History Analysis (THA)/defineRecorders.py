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

recorder('Node', '-file', f'{dataDir}/disp.txt', '-time', '-node', 7, '-dof', 1, 'disp')
# recorder('Node', '-file', f'{dataDir}/vel.txt', '-time', '-node', 7, '-dof', 1, 'vel')

# recorder('Node', '-file', f'{dataDir}/Vx.txt',  '-node', 1, 3, '-dof', 1, 'reaction')


# recorder('EnvelopeNode', '-file', f'{dataDir}/maxDisp.txt', '-node', 7, '-dof', 1, 'disp')
# recorder('EnvelopeNode', '-file', f'{dataDir}/maxVel.txt', '-node', 7, '-dof', 1, 'vel')


# recorder('EnvelopeNode', '-file', f'{dataDir}/modeShape1.txt', '-node', 7, '-dof', 1, 'eigen 1')


# recorder('EnvelopeElement', '-file', f'{dataDir}/col1_globalForce.txt', '-ele', 1, 'force')
# recorder('EnvelopeElement', '-file', f'{dataDir}/integrationWeights.txt', '-ele', 1, 'integrationWeights')
# recorder('EnvelopeElement', '-file', f'{dataDir}/col1_sec1_deformation.txt', '-ele', 1, 'section', 1, 'deformations')
# recorder('EnvelopeElement', '-file', f'{dataDir}/fiberStrain.txt', '-ele', 1, 'section', 1, 'fiber', dc/2, 0, steelMatTag, 'strain')




