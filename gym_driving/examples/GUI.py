from tkinter import *
from tkinter import ttk
from generate_config import GenerateConfig
#class Configuration(Frame):
#	def __init__(self, master=None):
#		super().__init__(master)
#		self.pack()
#		self.create_widgets()
#
#	def create_widgets(self):
#		self.hi_there = Button(self)
#		self.hi_there['text'] = "Hello World \n (click me)"
#		self.hi_there['command'] = self.say_hi
#		self.hi_there.pack(side='top', expand=1, fill='both')
#
#		
#
#		self.quit = Button(self, text='QUIT', fg='red', bg='blue',command=root.destroy)
#		self.quit.pack(side='bottom')
#
#	def say_hi(self):
#		print('hi there, everyone!')


def calculate(*args):
	try:
		value = float(feet.get())
		meters.set((.3048*value*10000.0+.5)/10000)
		print(meters.get())
	except ValueError:
		pass

root = Tk()
root.title('Feet to Meters')

mainframe = ttk.Frame(root, padding='3 3 12 12')
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(1, weight=1)
mainframe.rowconfigure(1, weight=1)

feet = StringVar()
meters = StringVar()

feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
feet_entry.grid(column=2, row=1, sticky=(W,E))

ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
ttk.Button(mainframe, text='Calculate', command=calculate).grid(column=3, row=3, sticky=(W))

ttk.Label(mainframe, text='feet').grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text='is equivalent to').grid(column=1, row=2, sticky=E)
ttk.Label(mainframe, text='meters').grid(column=3, row=2, sticky=E)


#Testing
config_dict = GenerateConfig()
ttk.Combobox(mainframe, values=config_dict.config_dict_defaults.keys()).grid(column=1, row=3, stick=(S, W))

for child in mainframe.winfo_children():
	child.grid_configure(padx=5, pady=5)

feet_entry.focus()
root.bind('<Return>', calculate)

root.mainloop()