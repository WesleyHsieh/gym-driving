from xboxController import *

class Controller:
	def __init__(self, acc_action, steer_action, mode='keyboard'):
		self.mode = mode
		self.acc_action = acc_action
		self.steer_action = steer_action
		if mode == 'keyboard':
			pass
		elif mode == 'xbox':
			self.xbox_controller = XboxController()
		else:
			pass

	def process_input(self):
		if self.mode == 'keyboard':
			action_dict = self.process_keys()
		elif self.mode == 'xbox':
			action_dict = self.process_xbox_controller()
		return action_dict

	def process_keys(self):
	    action_dict = {'steer': 0.0, 'acc': 0.0}
	    pygame.event.pump()
	    keys = pygame.key.get_pressed()
	    if keys[pygame.K_UP]:
	        action_dict['acc'] = self.acc_action
	    elif keys[pygame.K_DOWN]:
	        action_dict['acc'] = -self.acc_action
	    if keys[pygame.K_LEFT]:
	        action_dict['steer'] = -self.steer_action
	    elif keys[pygame.K_RIGHT]:
	        action_dict['steer'] = self.steer_action
	    return action_dict

	def process_xbox_controller(self):
		action_dict = {'steer': 0.0, 'acc': 0.0}
		# TODO: Xbox Controller
		left_stick_horizontal, left_stick_vertical, \
		right_stick_horizontal, right_stick_vertical = \
						self.xbox_controller.getUpdates()
		action_dict['steer'] = self.steer_action * right_stick_horizontal
		action_dict['acc'] = -self.acc_action * left_stick_vertical
		return action_dict
