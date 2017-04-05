from builtins import input
import numpy as np
import pickle
import IPython 
import os 
import pprint
import json

def get_filtered_input(key, description, default_args, num_args, num_inputs, generator_func):
	"""
	Obtains raw input and splits on whitespace.
	Calls generator_func on the input list if it is valid, 
	otherwise returns default_args.
	"""
	output = []
	input_prompt = 'Key: {} \n Description: {} \n Default Arguments: {} \n Number of Arguments: {} \n Input Arguments: '.format(key, description, default_args, num_args)
	input_token_lists = []

	for i in range(num_inputs):
		print("-----")
		print("Collecting input {} out of {}".format(i + 1, num_inputs))
		input_tokens = input(input_prompt).split(' ')
		if is_empty(input_tokens):
			input_token_lists = default_args if num_inputs > 1 else [default_args]
			break
		elif input_tokens[0] == 'quit':
			break
		else:
			input_token_lists.append(input_tokens)
	for token_list in input_token_lists:
		output_val = apply_tokens(token_list, generator_func)
		output.append(output_val)
		print("Output Value: {}".format(output_val))
	if num_inputs == 1:
		output = output[0]
	return output

def apply_tokens(tokens, generator_func):
	output_val = generator_func(*tokens)
	return output_val

def is_empty(tokens):
	return len(tokens) == 1 and tokens[0] == ''

class ParamaterWrapper:
	def __init__(self):
		pass
	def get_num_cpu_cars(self, num):
		return int(num)
	def get_main_car_starting_angles(self, low, high, step):
		low, high, step = map(float, [low, high, step])
		return np.linspace(low, high, step).tolist()
	def get_cpu_cars_bounding_box(self, x_low, x_high, y_low, y_high):
		x_low, x_high, y_low, y_high = map(float, [x_low, x_high, y_low, y_high])
		return [[x_low, x_high], [y_low, y_high]]
	def get_screen_size(self, x, y):
		return (int(x), int(y))
	def get_screenshot_dir(self, screenshot_dir):
		if screenshot_dir is not None:
			screenshot_dir = str(screenshot_dir)
		return screenshot_dir
	def get_screenshot_rate(self, screenshot_rate):
		return int(screenshot_rate)
	def get_time_horizon(self, time_horizon):
		return int(time_horizon)
	def get_terrain(self, x, y, width, length, texture):
		return int(x), int(y), int(width), int(length), str(texture)
	def get_main_car_params(self, x, y, vel, max_vel):
		return int(x), int(y), float(vel), float(max_vel)
	def get_steer_action(self, steer):
		return float(steer)
	def get_acc_action(self, acc):
		return float(acc)

class GenerateConfig:
	def __init__(self):
		self.paramater_wrapper = ParamaterWrapper()
		# parameter: (description, default_args, num_args, num_inputs, generator_func)
		self.config_dict_defaults = {
			'num_cpu_cars': ('Number of CPU cars (int)', [10], 1, 1, self.paramater_wrapper.get_num_cpu_cars), 
			'main_car_starting_angles': ('Low, High, Number of Angles (float, float, float)', [-30, 30, 5], 3, 1, self.paramater_wrapper.get_main_car_starting_angles),
			'cpu_cars_bounding_box': ('x_low, x_high, y_low, y_high (float, float, float, float)', [100.0, 1000.0, -90.0, 90.0], 4, 1, self.paramater_wrapper.get_cpu_cars_bounding_box),
			'screen_size': ('Width, Height of Screen (int, int)', [512, 512], 2, 1, self.paramater_wrapper.get_screen_size), 
			'screenshot_dir': ('Directory of screenshots (str)', [None], 1, 1, self.paramater_wrapper.get_screenshot_dir), 
			'screenshot_rate': ('Screenshot every n frames (int)', [10], 1, 1, self.paramater_wrapper.get_screenshot_rate), 
			'time_horizon': ('Time horizon for a rollout (int)', [100], 1, 1, self.paramater_wrapper.get_time_horizon), 
			'terrain_params': ('x, y, width, length, texture (int, int, int, int, str)', \
				[[0, -2000, 20000, 38000, 'grass'], [0, 0, 20000, 200, 'road'], [0, 2000, 20000, 3800, 'grass']], \
				5, 100, self.paramater_wrapper.get_terrain),
			'main_car_params': ('x, y, starting_vel, max_vel (int, int, float, float)', [0, 0, 0.0, 20.0], \
				4, 1, self.paramater_wrapper.get_main_car_params),
			'steer_action': ('Angle change in degrees for a steer action (float)', [15.0], 1, 1, self.paramater_wrapper.get_steer_action),
			'acc_action': ('Velocity change for an acceleration action (float)', [5.0], 1, 1, self.paramater_wrapper.get_acc_action),
		}
		self.command_dict = {
			'help': self.help,
			'edit': self.edit_config_dict,
			'show keys': self.show_keys,
			'show config': self.show_config,
			'save': self.save_config_dict,
			'load': self.load_config_dict,
		}
		self.config_dict = self.generate_default_config_dict()
		self.printer = pprint.PrettyPrinter(indent=4)
		self.default_filepath = 'configs/config.json'
	def generate_default_config_dict(self):
		config_dict = {}
		for key in self.config_dict_defaults.keys():
			description, default_args, num_args, num_inputs, generator_func = self.config_dict_defaults[key]
			if num_inputs == 1:
				config_dict[key] = apply_tokens(default_args, generator_func)
			else:
				config_dict[key] = [apply_tokens(default_arg, generator_func) for default_arg in default_args]
		return config_dict

	def help(self):
		print('Generate the config_dict used in the driving environment. \
				A config_dict with the default arguments has already been generated.')

	def update_keys(self, keys=None):
		if keys is None:
			keys = self.config_dict_defaults.keys()
		for key in keys:
			if key in self.config_dict_defaults:
				description, default_args, num_args, num_inputs, generator_func = self.config_dict_defaults[key]
				self.config_dict[key] = get_filtered_input(key, description, default_args, num_args, num_inputs, generator_func)
			else:
				print('Key not found, skipping: {}'.format(key))

	def edit_config_dict(self):
		self.show_keys()
		prompt = 'Input the desired keys, separated by spaces:\nInput: '
		input_tokens = input(prompt).split(' ')
		if is_empty(input_tokens):
			keys = None
		else:
			keys = input_tokens
		self.update_keys(keys)

	def show_keys(self):
		print("Keys")
		print(self.config_dict_defaults.keys())

	def show_config(self):
		print("Config Dict")
		self.printer.pprint(self.config_dict)

	def save_config_dict(self):
		prompt = 'Input the desired save location: (default: {})\nInput: '.format(self.default_filepath)
		input_str = input(prompt)
		try:
			if len(input_str) == 0:
				input_str = self.default_filepath
			json.dump(self.config_dict, open(input_str, 'w'), indent=4)
			print('Saved at {}'.format(input_str))
		except IOError as e:
			print(e)

	def load_config_dict(self):
		prompt = 'Input the desired load location: (default: {})\nInput: '.format(self.default_filepath)
		input_str = input(prompt)
		try:
			if len(input_str) == 0:
				input_str = self.default_filepath
			self.config_dict = json.load(open(input_str, 'r'))
			print('Loaded from {}'.format(input_str))
		except IOError as e:
			print(e)

	def config_loop(self):
		command_options = ', '.join(sorted(self.command_dict.keys()))
		prompt = '=========\nOptions: quit, {}\n Input: '.format(command_options)
		while True:
			input_str = input(prompt)
			input_tokens = input_str.split(' ')
			if input_str == 'quit':
				break
			elif input_str in self.command_dict:
				self.command_dict[input_str]()
			else:
				print("Command not recognized")

if __name__ == '__main__':
	config = GenerateConfig()
	config.config_loop()
	config_dict = config.config_dict
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(config_dict)