from tkinter import *
from tkinter import ttk
from generate_config import *

import argparse

class ConfigGUI():
    def __init__(self, filepath=None):
        # Creates the window
        self.root = Tk()
        self.root.title('Configuration Settings')

        # The frame to which everything is aligned
        self.mainframe = ttk.Frame(self.root, padding='3 3 12 12')
        self.mainframe.grid(column=0, row=0, sticky=(N, S, E, W))

        # Initializes the parameter wrapper class (defined in generate_config.py)
        self.par_wrap = ParameterWrapper()


        ##-----------------------------------------------Default Values---------------------------------------------------##
        if not filepath:
            self.config_dict = {
                'num_cpu_cars': 10,
                'main_car_starting_angles': [-30, 30, 5], #'Low, High, Num (float, float, float/None)'
                'cpu_cars_bounding_box': [[200.0, 1000.0], [-90.0, 90.0]], # ('x_low, x_high, y_low, y_high (float, float, float, float)',
                'screen_size': [512, 512], # 'Width, Height of Screen (int, int)'
                'logging_dir': None, # 'Directory for logging (str)'
                'logging_rate': 10, # 'Logging every n steps (int)'
                'time_horizon': 100, # 'Time horizon for a rollout (int)'
                'terrain_params': [[0, -2000, 30000, 3800, 0, 'grass'], [0, 0, 30000, 200, 0, 'road'], [0, 2000, 30000, 3800, 0, 'grass']], # 'x, y, width, length, texture (int, int, int, int, str)'
                'main_car_params': [0, 0, 0.0, 20.0], # 'x, y, starting_vel, max_vel (int, int, float, float)'
                'steer_action': [-15.0, 15.0, 3.0], # 'Low, High, Num (float, float, float/None)'
                'acc_action': [-5.0, 5.0, 3.0], # 'Low, High, Num (float, float, float/None)'
                'control_space': 'discrete', # 'Control space (str, {discrete, continuous})'
                'downsampled_size': None, # 'Downsampled size (int)'
                'state_space': 'positions', # 'State space (str, {positions, image})'
                'main_car_dynamics': 'kinematic', # 'Main car dynamics model (str, {point, kinematic, dynamic})'
                'noise': ['gaussian', .1],

            }
        else:
            try:
                self.config_dict = json.load(open(filepath, 'r'))
            except IOError as e:
                print(e)

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

        self.set_values()

        # Save button
        self.save_frame = ttk.Frame(self.mainframe)
        self.save_frame.grid(column=0, row=30, columnspan=5, sticky=W)
        ttk.Label(self.save_frame, text='Save location').grid(column=0, row=0)
        ttk.Entry(self.save_frame, textvariable=self.save_filepath).grid(column=1, row=0)
        ttk.Button(self.save_frame, command=self.save, text='Save').grid(column=2, row=0)

    def set_values(self):
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
        self.cpu_start_xlow.set(self.config_dict['cpu_cars_bounding_box'][0][0])
        self.cpu_start_xhigh.set(self.config_dict['cpu_cars_bounding_box'][0][1])
        self.cpu_start_ylow.set(self.config_dict['cpu_cars_bounding_box'][1][0])
        self.cpu_start_yhigh.set(self.config_dict['cpu_cars_bounding_box'][1][1])
        # Screen size
        self.screen_size_x, self.screen_size_y = StringVar(), StringVar()
        self.screen_size_x.set(self.config_dict['screen_size'][0])
        self.screen_size_y.set(self.config_dict['screen_size'][1])
        # Logging stats
        self.logging_dir, self.logging_rate = StringVar(), StringVar()
        self.logging_dir.set(self.config_dict['logging_dir'])
        self.logging_rate.set(self.config_dict['logging_rate'])
        # Time Horizon
        self.time_horizon = StringVar()
        self.time_horizon.set(self.config_dict['time_horizon'])
        # Terrain Parameters
        self.terrain_elements = []
        for elem in self.config_dict['terrain_params']:
            terrain_element = []
            for param in elem:
                x = StringVar()
                x.set(param)
                terrain_element.append(x)
            self.terrain_elements.append(terrain_element)

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
        self.control_space.set(self.config_dict['control_space'])
        # Downsampled Size
        self.downsampled_size = StringVar()
        self.downsampled_size.set(self.config_dict['downsampled_size'])
        # State Space
        self.state_space = StringVar()
        self.state_space.set(self.config_dict['state_space'])
        # Main Car Dynamics
        self.main_car_dynamics = StringVar()
        self.main_car_dynamics.set(self.config_dict['main_car_dynamics'])
        # Noise
        self.noise_type, self.noise_magnitude = StringVar(), StringVar()
        self.noise_type.set(self.config_dict['noise'][0])
        self.noise_magnitude.set(self.config_dict['noise'][1])
        # Save Location
        self.save_filepath = StringVar()
        self.save_filepath.set('../configs/config.json')
        ##---------------------------------------------End Setting Values-------------------------------------------------##





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
            noise_right = ttk.Label(action_grid, text=self.noise_type.get())
            noise_right.grid(column=1, row=6)


            acc_left = ttk.Label(action_grid, text='Acceleration Magnitude:')
            acc_right = ttk.Label(action_grid, text=self.acc_high.get())
            acc_left.grid(column=0, row=2)
            acc_right.grid(column=1, row=2)
            ang_left = ttk.Label(action_grid, text='Max Steering Angle:')
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
            # Variables:
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
            noise_type_label = ttk.Label(self.mainframe, text= 'Type')
            noise_type_entry = ttk.Combobox(self.mainframe, textvariable=self.noise_type, values=['gaussian', 'discrete'], state='readonly')
            noise_mag_label = ttk.Label(self.mainframe, text='Magnitude')
            noise_mag_entry = ttk.Entry(self.mainframe, textvariable=self.noise_magnitude)
            self.currently_open.extend([noise_label, noise_type_label, noise_type_entry, noise_mag_label, noise_mag_entry])
            i = 0
            for elem in [noise_label, noise_type_label, noise_type_entry, noise_mag_label, noise_mag_entry]:
                elem.grid(column=i, row=2, sticky=W)
                i += 1


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
            opt_point = ttk.Radiobutton(self.mainframe, variable=self.main_car_dynamics, value='point', text='Point', command=self.display_option_selected)
            opt_kinematic = ttk.Radiobutton(self.mainframe, variable=self.main_car_dynamics, value='kinematic', text='Kinematic', command=self.display_option_selected)
            opt_dynamic = ttk.Radiobutton(self.mainframe, variable=self.main_car_dynamics, value='dynamic', text='Dynamic', command=self.display_option_selected)
            opt_point.grid(column=1, row=1)
            opt_kinematic.grid(column=2, row=1)
            opt_dynamic.grid(column=3, row=1)
            self.currently_open.extend([opt_dynamic, opt_point, opt_kinematic, dynamics_label])

        elif self.selection.get() == 'State Space':
            #The state space contains the state space type, the number of CPU cars, screen size, and downsampled size
            # Variables:
            # num_cpu_cars, state_space, downsampled_size

            space_label = ttk.Label(self.mainframe, text='State Space')
            space_label.grid(column=0, row=1, sticky=W)
            self.currently_open.append(space_label)

            state_space_discrete = ttk.Radiobutton(self.mainframe, variable=self.state_space, value='positions', text='Positions', command=self.display_option_selected)
            state_space_cont = ttk.Radiobutton(self.mainframe, variable=self.state_space, value='image', text='Image', command=self.display_option_selected)
            state_space_discrete.grid(column=2, row=1)
            state_space_cont.grid(column=4, row=1)
            self.currently_open.extend([state_space_discrete, state_space_cont])

            num_label = ttk.Label(self.mainframe, text='Number of CPU Cars')
            num_entry = ttk.Entry(self.mainframe, textvariable=self.num_cpu_cars)
            num_label.grid(column=0, row=2, sticky=W)
            num_entry.grid(column=1, columnspan=2, row=2)
            self.currently_open.extend([num_label, num_entry])

            downsize_label = ttk.Label(self.mainframe, text='Downsampled Size')
            downsize_entry = ttk.Entry(self.mainframe, textvariable=self.downsampled_size)
            downsize_label.grid(column=0, row=3, sticky=W)
            downsize_entry.grid(column=1, columnspan=2, row=3)
            self.currently_open.extend([downsize_label, downsize_entry])

        elif self.selection.get() == 'Starting Setup':
            # The starting setup contains the CPU car bounding box, main car starting angle range,
            # and the main car starting velocity
            # Variables:
            # mc_start_angle_low, mc_start_angle_high, mc_start_angle_step, cpu_start_xlow, cpu_start_xhigh,
            # cpu_start_ylow, cpu_start_yhigh, mc_startv

            # CPU Bounding Box
            cpu_start_label = ttk.Label(self.mainframe, text='CPU Car Start Box')
            cpu_x_label = ttk.Label(self.mainframe, text='X Range')
            cpu_y_label = ttk.Label(self.mainframe, text='Y Range')
            cpu_xlow = ttk.Entry(self.mainframe, textvariable=self.cpu_start_xlow)
            cpu_xhigh = ttk.Entry(self.mainframe, textvariable=self.cpu_start_xhigh)
            cpu_ylow = ttk.Entry(self.mainframe, textvariable=self.cpu_start_ylow)
            cpu_yhigh = ttk.Entry(self.mainframe, textvariable=self.cpu_start_yhigh)
            cpu_xdash = ttk.Label(self.mainframe, text='-')
            cpu_ydash = ttk.Label(self.mainframe, text='-')
            cpu_start_label.grid(column=0, row=1)
            cpu_x_label.grid(column=0, row=2)
            cpu_xlow.grid(column=1, row=2), cpu_xdash.grid(column=2, row=2), cpu_xhigh.grid(column=3, row=2)
            cpu_y_label.grid(column=0, row=3)
            cpu_ylow.grid(column=1, row=3), cpu_ydash.grid(column=2, row=3), cpu_yhigh.grid(column=3, row=3)
            self.currently_open.extend([cpu_start_label, cpu_x_label, cpu_y_label, cpu_xlow, cpu_xhigh, cpu_ylow, cpu_yhigh, cpu_xdash, cpu_ydash])

            # Main Car Starting Range
            mc_start_label = ttk.Label(self.mainframe, text='Main Car Starting Position:')
            mc_xy_label = ttk.Label(self.mainframe, text= 'Position (x, y)')
            mc_x_entry = ttk.Entry(self.mainframe, textvariable=self.mc_x)
            mc_y_entry = ttk.Entry(self.mainframe, textvariable=self.mc_y)
            mc_start_degrees = ttk.Label(self.mainframe, text='Angle Range (degrees)')
            mc_lstart = ttk.Entry(self.mainframe, textvariable=self.mc_start_angle_low)
            mc_dash = ttk.Label(self.mainframe, text='-')
            mc_hstart = ttk.Entry(self.mainframe, textvariable=self.mc_start_angle_high)
            mc_start_stepl = ttk.Label(self.mainframe, text='Step Size')
            mc_start_stepe = ttk.Entry(self.mainframe, textvariable=self.mc_start_angle_step)
            mc_xy_label.grid(row=5, sticky=W)
            mc_x_entry.grid(column=1, row=5)
            mc_y_entry.grid(column=3, row=5)
            mc_start_label.grid(column=0, row=4, sticky=W)
            mc_start_degrees.grid(column=0, row=6)
            mc_lstart.grid(column=1, row=6)
            mc_dash.grid(column=2, row=6)
            mc_hstart.grid(column=3, row=6)
            if self.control_space.get() == 'discrete':
                mc_start_stepl.grid(column=4, row=6)
                mc_start_stepe.grid(column=5, row=6)
                self.currently_open.extend([mc_start_stepl, mc_start_stepe])
            self.currently_open.extend([mc_start_label, mc_lstart, mc_dash, mc_hstart, mc_start_degrees, mc_xy_label, mc_x_entry, mc_y_entry])


        elif self.selection.get() == 'Logging and Time Horizon':
            # Contains logging rate, logging directory, and time horizon
            # Variables:
            # logging_rate, logging_dir, time_horizon

            # Logging
            log_label = ttk.Label(self.mainframe, text='Logging Rate and Directory')
            log_rate_entry = ttk.Entry(self.mainframe, textvariable=self.logging_rate)
            log_dir_entry = ttk.Entry(self.mainframe, textvariable=self.logging_dir)
            log_label.grid(row=1, sticky=W)
            log_rate_entry.grid(column=0, row=2)
            log_dir_entry.grid(column=1, row=2)
            self.currently_open.extend([log_label, log_rate_entry, log_dir_entry])

            # Time Horizon
            horizon_label = ttk.Label(self.mainframe, text='Time Horizon')
            horizon_entry = ttk.Entry(self.mainframe, textvariable=self.time_horizon)
            horizon_steps = ttk.Label(self.mainframe, text='timesteps')
            horizon_label.grid(column=0, row=3, sticky=W)
            horizon_entry.grid(column=0, row=4)
            horizon_steps.grid(column=1, row=4, sticky=W)
            self.currently_open.extend([horizon_steps, horizon_entry, horizon_label])

        elif self.selection.get() == 'Terrain':
            # Displays and makes editable all the current terrain elements
            terrain_label = ttk.Label(self.mainframe, text='Terrain Elements')
            terrain_label.grid(column=0, row=1)
            terrain_xy = ttk.Label(self.mainframe, text='Center (x, y)')
            terrain_width = ttk.Label(self.mainframe, text='Width')
            terrain_length = ttk.Label(self.mainframe, text='Height')
            terrain_angle = ttk.Label(self.mainframe, text='Angle (deg)')
            terrain_type = ttk.Label(self.mainframe, text='Type')
            terrain_xy.grid(row=2, columnspan=2)
            terrain_width.grid(column=2, row=2)
            terrain_length.grid(column=3, row=2)
            terrain_angle.grid(column=4, row=2)
            terrain_type.grid(column=5, row=2)
            row_index = [3] # Used a 1-element list because python 2 doesn't have nonlocal
            # was useful for new_element
            for elem in self.terrain_elements:
                column_index = 0
                for param in elem:
                    entry = ttk.Entry(self.mainframe, textvariable=param)
                    entry.grid(column=column_index, row=row_index[0])
                    self.currently_open.append(entry)
                    column_index += 1
                row_index[0] = row_index[0] + 1

            def new_element():
                element_list = []
                for i in range(6):
                    param = StringVar()
                    element_list.append(param)
                    if i < 5:
                        param.set(0)
                    else:
                        param.set('grass')
                    entry = ttk.Entry(self.mainframe, textvariable=param)
                    entry.grid(column=i, row=row_index[0])
                    self.currently_open.append(entry)
                self.terrain_elements.append(element_list)
                row_index[0] = row_index[0] + 1

                print [[param.get() for param in elem] for elem in self.terrain_elements]
            # Button that lets you add a new terrain element
            new_terrain_button = ttk.Button(self.save_frame, text='Create New Terrain Element', command=new_element)
            new_terrain_button.grid(column=3, row=0)
        else:
            [widget.destroy() for widget in self.currently_open]


    def save(self):
        # The new config dictionary converts all StringVars needed to interact with TKinter into standard python variables.
        # It applies the necessary functions to them (str, int, float, etc).
        new_config_dict = {
            'num_cpu_cars': int(self.num_cpu_cars.get()),
            'main_car_starting_angles': [float(self.mc_start_angle_low.get()), float(self.mc_start_angle_high.get()), float(self.mc_start_angle_step.get())], #'Low, High, Num (float, float, float/None)'
            'cpu_cars_bounding_box': [[float(self.cpu_start_xlow.get()), float(self.cpu_start_xhigh.get())], [float(self.cpu_start_ylow.get()), float(self.cpu_start_yhigh.get())]], # ('x_low, x_high, y_low, y_high (float, float, float, float)',
            'screen_size': [int(self.screen_size_x.get()), int(self.screen_size_y.get())], # 'Width, Height of Screen (int, int)'
            'logging_dir': self.par_wrap.get_logging_dir(self.logging_dir.get()), # 'Directory for logging (str)'
            'logging_rate': int(self.logging_rate.get()), # 'Logging every n steps (int)'
            'time_horizon': int(self.time_horizon.get()), # 'Time horizon for a rollout (int)'
            ############# TERRAIN UNFINISHED
            'terrain_params': self.par_wrap.get_terrain_from_list(self.terrain_elements), # 'x, y, width, length, texture (int, int, int, int, str)'
            ############# TERRAIN UNFINISHED
            'main_car_params': [int(self.mc_x.get()), int(self.mc_y.get()), float(self.mc_startv.get()), float(self.mc_maxv.get())], # 'x, y, starting_vel, max_vel (int, int, float, float)'
            'steer_action': [-float(self.ang_high.get()), float(self.ang_high.get()), float(self.ang_step.get())], # 'Low, High, Num (float, float, float/None)'
            'acc_action': [-float(self.acc_low.get()), float(self.acc_high.get()), float(self.acc_step.get())], # 'Low, High, Num (float, float, float/None)'
            'control_space': str(self.control_space.get()), # 'Control space (str, {discrete, continuous})'
            'downsampled_size': self.par_wrap.get_downsampled_size(self.downsampled_size.get()), # 'Downsampled size (int)'
            'state_space': str(self.state_space.get()), # 'State space (str, {positions, image})'
            'main_car_dynamics': str(self.main_car_dynamics.get()), # 'Main car dynamics model (str, {point, kinematic, dynamic})'
            'noise': [str(self.noise_type.get()), float(self.noise_magnitude.get())],
        }

        try:
            json.dump(new_config_dict, open(self.save_filepath.get(), 'w'), indent=4)
            print('Saved at {}'.format(self.save_filepath.get()))
        except IOError as e:
            print e




def test():
    print config.control_space.get()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    args = parser.parse_args()
    config_filepath = args.config
    config = ConfigGUI(config_filepath)
    config.root.mainloop()
