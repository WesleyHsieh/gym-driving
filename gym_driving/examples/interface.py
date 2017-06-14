from tkinter import *
from tkinter import ttk
from generate_config import *


class ConfigGUI():
	def __init__(self):
		# Creates the window
		self.root = Tk()
		self.root.title('Configuration Settings')

		# The frame to which everything is aligned
		self.mainframe = ttk.Frame(self.root, padding='3 3 12 12')
		self.mainframe.grid(column=0, row=0, sticky=(N, S, E, W))

		##-----------------------------------------------Default Values---------------------------------------------------##
		self.config_dict = {
			'num_cpu_cars': [10], 
			'main_car_starting_angles': [-30, 30, 5], #'Low, High, Num (float, float, float/None)'
			'cpu_cars_bounding_box': [200.0, 1000.0, -90.0, 90.0], # ('x_low, x_high, y_low, y_high (float, float, float, float)', 
			'screen_size': [512, 512], # 'Width, Height of Screen (int, int)'
			'logging_dir': [None], # 'Directory for logging (str)'
			'logging_rate': [10], # 'Logging every n steps (int)'
			'time_horizon': [100], # 'Time horizon for a rollout (int)'
			'terrain_params': [[0, -2000, 30000, 3800, 'grass'], [0, 0, 30000, 200, 'road'], [0, 2000, 30000, 3800, 'grass']], # 'x, y, width, length, texture (int, int, int, int, str)'
			'main_car_params': [0, 0, 0.0, 20.0], # 'x, y, starting_vel, max_vel (int, int, float, float)'
			'steer_action': [-15.0, 15.0, 3.0], # 'Low, High, Num (float, float, float/None)'
			'acc_action': [-5.0, 5.0, 3.0], # 'Low, High, Num (float, float, float/None)'
			'control_space': ['discrete'], # 'Control space (str, {discrete, continuous})'
			'downsampled_size': [None], # 'Downsampled size (int)'
			'state_space': ['positions'], # 'State space (str, {positions, image})'
			'main_car_dynamics': ['kinematic'], # 'Main car dynamics model (str, {point, kinematic, dynamic})'

		}

		self.selection_options = ['Current Values', 'Action Space', 'Dynamics', 'Starting Setup', 'State Space', 'Logging and Time Horizon', 'Terrain']
		self.selection = StringVar()
		self.selector = ttk.Combobox(self.mainframe,
		                             values=self.selection_options,
		                             textvariable=self.selection,
		                             state='readonly'
		                             )
		self.selector.grid(row=0, column=0, sticky=W)
		self.selector.bind('<<ComboboxSelected>>', self.display_option_selected)
		self.currently_open = [] #Stores the open instances so they can be closed later

		self.set_values(None)

	def set_values(self, config_filepath):
		# Number of cars
		self.num_cpu_cars = StringVar()
		self.num_cpu_cars.set(self.config_dict['num_cpu_cars'])
		# Starting angle bounds and step size when discretized
		self.mc_start_angle_low, self.mc_start_angle_high, self.mc_start_angle_step = StringVar(), StringVar(), StringVar()
		self.mc_start_angle_low.set(self.config_dict['main_car_starting_angles'][0])
		self.mc_start_angle_high.set(self.config_dict['main_car_starting_angles'][1])
		self.mc_start_angle_step.set(self.config_dict['main_car_starting_angles'][2])
		# CPU Car Starting Area
		self.cpu_start_xlow, self.cpu_start_xhigh, self.cpu_start_ylow, self.cpu_start_yhigh = StringVar(), StringVar(), StringVar(), StringVar()
		self.cpu_start_xlow.set(self.config_dict['cpu_cars_bounding_box'][0])
		self.cpu_start_xhigh.set(self.config_dict['cpu_cars_bounding_box'][1])
		self.cpu_start_ylow.set(self.config_dict['cpu_cars_bounding_box'][2])
		self.cpu_start_yhigh.set(self.config_dict['cpu_cars_bounding_box'][3])
		# Screen size
		self.screen_size_x, self.screen_size_y = StringVar(), StringVar()
		self.screen_size_x.set(self.config_dict['screen_size'][0])
		self.screen_size_y.set(self.config_dict['screen_size'][1])
		# Logging stats
		self.logging_dir, self.logging_rate = StringVar(), StringVar()
		self.logging_dir.set(self.config_dict['logging_dir'][0])
		self.logging_rate.set(self.config_dict['logging_rate'][0])
		# Time Horizon
		self.time_horizon = StringVar()
		self.time_horizon.set(self.config_dict['time_horizon'][0])
		# Terrain Parameters
		# Come back to this
		# Main Car Parameters
		self.mc_x, self.mc_y, self.mc_startv, self.mc_maxv = StringVar(), StringVar(), StringVar(), StringVar()
		self.mc_x.set(self.config_dict['main_car_params'][0])
		self.mc_y.set(self.config_dict['main_car_params'][1])
		self.mc_startv.set(self.config_dict['main_car_params'][2])
		self.mc_maxv.set(self.config_dict['main_car_params'][3])
		# Steering Action
		self.ang_low, self.ang_high, self.ang_step = StringVar(), StringVar(), StringVar()
		self.ang_low.set(self.config_dict['steer_action'][0])
		self.ang_high.set(self.config_dict['steer_action'][1])
		self.ang_step.set(self.config_dict['steer_action'][2])
		# Acceleration Action
		self.acc_low, self.acc_high, self.acc_step = StringVar(), StringVar(), StringVar()
		self.acc_low.set(self.config_dict['acc_action'][0])
		self.acc_high.set(self.config_dict['acc_action'][1])
		self.acc_step.set(self.config_dict['acc_action'][2])
		# Control Space
		self.control_space = StringVar()
		self.control_space.set(self.config_dict['control_space'][0])
		# Downsampled Size
		self.downsampled_size = StringVar()
		self.downsampled_size.set(self.config_dict['downsampled_size'][0])
		# State Space
		self.state_space = StringVar()
		self.state_space.set(self.config_dict['state_space'][0])
		# Main Car Dynamics
		self.main_car_dynamics = StringVar()
		self.main_car_dynamics.set(self.config_dict['main_car_dynamics'][0])
		##---------------------------------------------End Setting Values-------------------------------------------------##





		

		ttk.Button(self.mainframe, command=test, text='Save').grid(column=15, row=15)


	def display_option_selected(self, event=None):
		"""This function is called whenever the selection in the selection box
		changes, and displays only the relevant options for that selection. If
		the user chose to write something not listedinto the combobox, it will
		not do anything.
		"""
		[widget.destroy() for widget in self.currently_open]
		self.currently_open = []
		if self.selection.get() == 'Current Values':
			action_grid = ttk.Frame(self.mainframe, padding='3 3 3 3')
			action_grid.grid(column=0, row=1)
			#action_label = ttk.Label(action_grid, text='Action Space Settings')
			#action_label.grid(column=0, row=0)

			cs_left = ttk.Label(action_grid, text='Control Space Type:')
			cs_left.grid(column=0, row=1)
			cs_right = ttk.Label(action_grid, text=self.control_space.get())
			cs_right.grid(column=1, row=1)


			noise_left = ttk.Label(action_grid, text='Noise Type:')
			noise_left.grid(column=0, row=6)
			#noise_right = ttk.Label(action_grid, text=self.noise.get())
			#noise_right.grid(column=1, row=1)

			acc_left = ttk.Label(action_grid, text='Acceleration Magnitude:')
			acc_right = ttk.Label(action_grid, text=self.acc_high.get())
			acc_left.grid(column=0, row=2)
			acc_right.grid(column=1, row=2)
			ang_left = ttk.Label(action_grid, text='Steering Magnitude:')
			ang_right = ttk.Label(action_grid, text=self.ang_high.get())
			ang_left.grid(column=0, row=4)
			ang_right.grid(column=1, row=4)
			#self.currently_open.extend([acc_left, acc_right, ang_left, ang_right])
			if self.control_space.get() == 'discrete':
			    angs_left = ttk.Label(action_grid, text='Steering Step Size:')
			    angs_right = ttk.Label(action_grid, text=self.ang_step.get())
			    angs_left.grid(column=0, row=5)
			    angs_right.grid(column=1, row=5)
			    accs_left = ttk.Label(action_grid, text='Acceleration Step Size:')
			    accs_right = ttk.Label(action_grid, text=self.acc_step.get())
			    accs_left.grid(column=0, row=3)
			    accs_right.grid(column=1, row=3)
			    #self.currently_open.extend([angs_left, angs_right, accs_left, accs_right])
			self.currently_open.append(action_grid)

			dyn_label = ttk.Label(action_grid, text='Dynamics Model:')
			dyn_label.grid(column=0, row=7)
			dyn_val = ttk.Label(action_grid, text=self.main_car_dynamics.get())
			dyn_val.grid(column=1, row=7)




		elif self.selection.get() == 'Action Space':
			# The action space contains steering action, acceleration action,
			# the type of control space (continuous/discrete), noise, and the
			# main car's min/max accel and velocity
			# ang_low, ang_high, ang_step, acc_low, acc_high, acc_step
			control_label = ttk.Label(self.mainframe, text='Control Space')
			control_label.grid(column=0, row=1, sticky=W)
			self.currently_open.append(control_label)

			control_space_discrete = ttk.Radiobutton(self.mainframe, variable=self.control_space, value='discrete', text='Discrete', command=self.display_option_selected)
			control_space_cont = ttk.Radiobutton(self.mainframe, variable=self.control_space, value='continuous', text='Continuous', command=self.display_option_selected)
			control_space_discrete.grid(column=2, row=1)
			control_space_cont.grid(column=4, row=1)
			self.currently_open.extend([control_space_discrete, control_space_cont])

			noise_label = ttk.Label(self.mainframe, text='Noise')
			noise_entry = ttk.Entry(self.mainframe, )
			noise_label.grid(column=0, row=2, sticky=W)
			self.currently_open.append(noise_label)
			# Finish noise after implementing it

			acc_label = ttk.Label(self.mainframe, text='Acceleration')
			acc_label.grid(column=0, row=3, sticky=W)
			self.currently_open.append(acc_label)
			acc_mlabel = ttk.Label(self.mainframe, text='Max')
			acc_mmlabel = ttk.Label(self.mainframe, text='Min')
			acc_slabel = ttk.Label(self.mainframe, text='Step Size')
			acc_mentry = ttk.Entry(self.mainframe, textvariable=self.acc_high)
			acc_mmentry = ttk.Entry(self.mainframe, textvariable=self.acc_low)
			acc_sentry = ttk.Entry(self.mainframe, textvariable=self.acc_step)
			self.currently_open.extend([acc_mlabel, acc_slabel, acc_mentry, acc_sentry, acc_mmlabel, acc_mmentry])
			if self.control_space.get() != 'discrete':
				acc_sentry['state'] = 'disabled'
			acc_slabel.grid(column=5, row=3, sticky=W)
			acc_sentry.grid(column=6, row=3, sticky=E)
			acc_mmlabel.grid(column=1, row=3, sticky=W)
			acc_mmentry.grid(column=2, row=3, sticky=E)
			acc_mlabel.grid(column=3, row=3)
			acc_mentry.grid(column=4, row=3)

			ang_label = ttk.Label(self.mainframe, text='Wheel Angle')
			ang_label.grid(column=0, row=4, sticky=W)
			self.currently_open.append(ang_label)
			ang_mlabel = ttk.Label(self.mainframe, text='Max')
			ang_slabel = ttk.Label(self.mainframe, text='Step Size')
			ang_mentry = ttk.Entry(self.mainframe, textvariable=self.ang_high)
			ang_sentry = ttk.Entry(self.mainframe, textvariable=self.ang_step)
			self.currently_open.extend([ang_mlabel, ang_slabel, ang_mentry, ang_sentry])
			if self.control_space.get() != 'discrete':
				ang_sentry['state'] = 'disabled'
			ang_slabel.grid(column=3, row=4, sticky=W)
			ang_sentry.grid(column=4, row=4, sticky=E)
			ang_mlabel.grid(column=1, row=4, sticky=W)
			ang_mentry.grid(column=2, row=4, sticky=E)

		elif self.selection.get() == 'Dynamics':
			dynamics_label = ttk.Label(self.mainframe, text='Dynamics Model')
			dynamics_label.grid(column=0, row=1, sticky=W)
			opt_point = ttk.Radiobutton(self.mainframe, variable=self.main_car_dynamics, value='point', text='Point')
			opt_kinematic = ttk.Radiobutton(self.mainframe, variable=self.main_car_dynamics, value='kinematic', text='Kinematic')
			opt_dynamic = ttk.Radiobutton(self.mainframe, variable=self.main_car_dynamics, value='dynamic', text='Dynamic')
			opt_point.grid(column=1, row=1)
			opt_kinematic.grid(column=3, row=1)
			opt_dynamic.grid(column=5, row=1)
			self.currently_open.extend([opt_dynamic, opt_point, opt_kinematic, dynamics_label])

		elif self.selection.get() == 'State Space':
			pass

		elif self.selection.get() == 'Starting Setup':
			pass

		else:
			[b.destroy() for b in self.currently_open]


	def save(self, filepath):
		new_config_dict = {
		    'num_cpu_cars': [self.num_cpu_cars], 
			'main_car_starting_angles': [self.mc_start_angle_low, self.mc_start_angle_high, self.mc_start_angle_step], #'Low, High, Num (float, float, float/None)'
			'cpu_cars_bounding_box': [self.cpu_start_xlow, self.cpu_start_xhigh, self.cpu_start_ylow, self.cpu_start_yhigh], # ('x_low, x_high, y_low, y_high (float, float, float, float)', 
			'screen_size': [self.screen_size_x, self.screen_size_y], # 'Width, Height of Screen (int, int)'
			'logging_dir': [self.logging_dir], # 'Directory for logging (str)'
			'logging_rate': [self.logging_rate], # 'Logging every n steps (int)'
			'time_horizon': [self.time_horizon], # 'Time horizon for a rollout (int)'
			############# TERRAIN UNFINISHED
			'terrain_params': self.config_dict['terrain_params'], # 'x, y, width, length, texture (int, int, int, int, str)'
			############# TERRAIN UNFINISHED
			'main_car_params': [self.mc_x, self.mc_y, self.mc_startv, self.mc_maxv], # 'x, y, starting_vel, max_vel (int, int, float, float)'
			'steer_action': [-self.ang_high, self.ang_high, self.ang_step], # 'Low, High, Num (float, float, float/None)'
			'acc_action': [self.acc_low, self.acc_high, 3.0], # 'Low, High, Num (float, float, float/None)'
			'control_space': [self.control_space], # 'Control space (str, {discrete, continuous})'
			'downsampled_size': [self.downsampled_size], # 'Downsampled size (int)'
			'state_space': [self.state_space], # 'State space (str, {positions, image})'
			'main_car_dynamics': [self.main_car_dynamics], # 'Main car dynamics model (str, {point, kinematic, dynamic})'
		}
		



def test():
	print config.control_space.get()


config = ConfigGUI()
config.root.mainloop()