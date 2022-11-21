#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/09/26
# ---------------------------------------------------------------------------
""" Environment to simulate a wind turbine """
# ---------------------------------------------------------------------------

import numpy as np
from math_utils import wrap_to_m180_p180
from filters import butter_lowpass, manual_filter
from scipy.signal import lfilter_zi
from queue import Queue

class Wind_turbine:
	# Mechanical Data
	name = 'V80/2000'
	manufacturer = 'Vestas'
	rated_power = 2 											# MW
	rotor_diameter = 80 										# m 
	nb_blades = 3
	power_control = 'pitch'
	min_rotor_speed = 9 										# rd/min
	max_rotor_speed = 19 										# rd/min
	cut_in_wind_speed = 3.5 									# m/s
	cut_off_wind_speed = 25 									# m/s
	min_hub_height = 60 										# m
	max_hub_height = 100 										# m
	# other public details here : https://www.thewindpower.net/turbine_en_30_vestas_v80-2000.php

	# Control data
	__power_curve = [[0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5,  5., 5.5,  6., 6.5,  7., 7.5,  8., 8.5,  9.,  9.5, 10. , 10.5, 11. , 11.5, 12. , 12.5, 13. , 13.5, 14. , 14.5, 15. , 15.5, 16. , 16.5, 17. , 17.5, 18. , 18.5, 19. , 19.5, 20. , 20.5, 21. , 21.5, 22. , 22.5, 23. , 23.5, 24. , 24.5, 25. ], \
					 [0,   0, 0,    0,  0,   0,  0,  35, 70, 117, 165, 225, 285, 372, 459, 580, 701, 832, 964, 1127, 1289, 1428, 1567, 1678, 1788, 1865, 1941, 1966, 1990, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]]
	__yaw_cut_off = 40 											# deg, the maximum yaw relative to the wind that the structure can handle
	__yaw_control_step = 1 										# deg
	__yaw_control_cost = 1e-2 * rated_power 					# MW
	__control_on = False
	__heading_sensor_bias = -3									# deg

	# Dynamics data
	__rotor_cutoff = 1/60										# Hz
	__filter_order = 1
	__b, __a = butter_lowpass(__rotor_cutoff, 1.0, __filter_order)
	__zi = lfilter_zi(__b, __a)
	__power_hist_filt = []  									# MW
	__power_hist = []  											# MW

	def __init__(self, initial_estimated_heading=None, has_inertia=None):
		''' 
		Inputs :
			initial_estimated_heading 		- [deg] The estimated heading = true heading + sensor bias wrt North in degree
			has_inertia 					- [bool] Determines whether the output power will be filtered

		Outputs :
			power_output 					- [MW]
		'''
		self._heading = -self.__heading_sensor_bias if initial_estimated_heading is None \
			else initial_estimated_heading - self.__heading_sensor_bias
		self._has_inertia = False if has_inertia is None else has_inertia

	def __power_output(self, wind_speed:float, wind_heading:float) -> float :
		'''
		The output power of the wind turbine in MW
		'''
		# Linear interpolation of the given power curve to get the power output (output in MW)
		facing_wind_power_output = np.interp(wind_speed, self.__power_curve[0], self.__power_curve[1])/1e3
		wraped_wt_heading = wrap_to_m180_p180(self._heading)
		wraped_wind_heading = wrap_to_m180_p180(wind_heading)
		rel_wind_angle = wraped_wind_heading - wraped_wt_heading
		# Get power output without filtering
		if np.abs(rel_wind_angle) > self.__yaw_cut_off:
			power_output = 0
		else:
			power_output = np.max([np.cos(rel_wind_angle * np.pi/180) * facing_wind_power_output, 0])


		# If filtering is enabled, process to low-pass filter
		if self._has_inertia:
			# Initialize the filter
			if len(self.__power_hist_filt) <= self.__filter_order:
				self.__power_hist.append(power_output)
				self.__power_hist_filt = len(self.__power_hist) * [np.mean(self.__power_hist)]
			# Filter
			else:
				self.__power_hist.pop(0)
				self.__power_hist.append(power_output)
				power_hist_filt = manual_filter(self.__b, self.__a, \
					self.__power_hist, self.__power_hist_filt[1:])
				self.__power_hist_filt.pop(0)
				self.__power_hist_filt.append(power_hist_filt[-1])
			return self.__power_hist_filt[-1]
		else:
			return power_output

	def __rotate(self, direction):
		if direction == -1 :
			# rotate trigo
			self._heading -= self.__yaw_control_step
			self.__control_on = True
		if direction == +1 :
			# rotate clockwise
			self._heading += self.__yaw_control_step
			self.__control_on = True
		if direction == 0 :
			# stays in place
			self.__control_on = False
		if direction != -1 and direction != +1 and direction != 0 :
			print('wind turbine command ', direction, ' not valid')
			self.__control_on = False

	def step(self, wind_speed:float, wind_heading:float, action:int) -> float:
		'''
		Takes an action and then returns the output power
		Inputs :
			wind_speed 		- [m/s] The wind speed at the wind turbine
			wind_heading 	- [deg] The wind heading wrt North in degree
			actions - {0, 1, 2}, 0 is rotate trigo, 1 is 'do nothing', 2 is rotate clockwise
		Outputs :
			power_output 	- [MW]
		'''
		self.__rotate(action-1)
		power_output = self.__power_output(wind_speed, wind_heading)
		if self.__control_on:
			power_output -= self.__yaw_control_cost
		return power_output

	@property
	def heading(self): 	# corresponds to the estimated heading
		return np.mod(self._heading + self.__heading_sensor_bias, 360)

	@property
	def true_heading(self): 	# corresponds to the true heading
		return np.mod(self._heading, 360)

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)



# Model of the wind. It simulates the wind with a short term gaussian noise and a long term (24h period)
# variation of the wind and the heading due to diurnal cycles
class Wind:
	model = 'OU' 					# model used for wind simulation
	__c_speed_factor = 0.2 			# variable used in the OU model
	__speed_noise = 0.1 			# variable used in the OU model
	__c_heading_factor = 0.005 		# variable used in the OU model
	__heading_noise = 0.3 			# variable used in the OU model
	__speed_target = 0.0 			# m/s, represents the average wind speed on some minutes
	__heading_target = 0.0 			# deg, represents the average wind angle on some minutes
	__diurnal_period = 24*3600 		# s, represents the period of the diurnal cycle
	__diurnal_factor = 0.2 			# represents the wind speed elliptical coefficient on a diurnal cycle
	__time = 0 						# s, represent the elapsed time in s due to the cumulative steps
	__revolution = 0 				# represents the number of revolutions
	__unwrap_threshold = 180 		# deg, is the threshold used to unwrap the wind angle
	__dt = 1 						# s, represents the time step

	def __init__(self, initial_speed=None, initial_heading=None, step_duration=None, model_type='OU'):
		''' 
		Inputs :
			heading 		- [deg] The wind angle wrt Northin degree
			speed 			- [m/s] A negative speed would result in a 180° shift in heading
			step_duration 	- [s] 	The duration of a time step. It should be higher or equal to 1s
			time_of_the_day - [s]	For instance 7H30AM is 7*3600 + 30*60. By default it is set to midnight
			model_type		- [] 	The model used to simulate the wind, the 'OU'
									model is the one selected by default and corresponds to the 
									Ornstein-Uhlenbeck model. It could be better represented by the Kailman,
									or the Mann model as described here : https://wes.copernicus.org/preprints/wes-2021-51/wes-2021-51.pdf 
		Outputs :
			heading 		- [deg]
			speed 			- [m/s]
		'''
		self._speed = 0 if initial_speed is None else initial_speed
		self._heading = 0 if initial_heading is None else initial_heading
		self.step_duration = self.__dt if step_duration is None else step_duration
		self.model_type = model_type

		# Initialise hidden variables. The heading and speed target corresponds to the
		# average on which the OU process must tend. They vary slowly whereas the OU
		# process accounts for fast variations such as gusts
		self.__speed_init = self._speed
		self.__heading_init = self._heading
		self.__heading_target = self.__heading_init

	def step(self):
		'''
		step_duration must be an int
		'''
		# Increment time and compute long term speed and heading duration
		self.__time += self.step_duration
		self.__diurnal_cycle()

		# Compute short term variations
		if self.model == 'OU':
			mean_sp = self._speed
			mean_hd = self._heading
			steps = 1
			for i in range(int(np.ceil(self.step_duration))):
				# Compute fast chaning wind at 1/dt frequency, typically 1Hz
				self.__ou()
				steps += 1
				mean_sp += (self._speed - mean_sp)/steps
				mean_hd += (self._heading - mean_hd)/steps
				if i % self.step_duration == 1:
					# Flush new value
					self._speed = mean_sp
					self._heading = mean_hd
					# Reset incremental mean
					mean_sp = self._speed
					mean_hd = self._heading
					steps = 1
		else:
			print('Wind model not found in class ', str(self.__class__))

	def __ou(self):
		'''
		This is the Ornstein-Uhlenbeck process, a stationary Gauss-Markov model,
		a math definition is available at page 7 here : https://www.merl.com/publications/docs/TR2022-102.pdf
		'''
		c_speed = self.__c_speed_factor * self.__speed_target
		# The random distribution variance is chosen such that wind gust can reach 40% of the average wind. Support for random walk and normal distribution
		# is available here : https://en.wikipedia.org/wiki/Random_walk#:~:text=A%20random%20walk%20having%20a,walk%20as%20an%20underlying%20assumption
		self._speed = c_speed + (1 - self.__c_speed_factor) * self._speed + np.random.normal(0.0, self.__speed_noise * np.abs(self.__speed_target))

		c_heading = self.__c_heading_factor * self.__heading_target
		self._heading = c_heading + (1 - self.__c_heading_factor) * self._heading + np.random.normal(0.0, self.__heading_noise)

	def __diurnal_cycle(self):
		'''
		The diurnal cycles is modelised as an elliptic day-night shift of the wind : https://en.wikipedia.org/wiki/Ellipse 
		'''
		u = np.tan((2 * np.pi * self.__time / self.__diurnal_period) / 2)
		x = (1 - u**2)/(1 + u**2)
		y = self.__diurnal_factor * (2*u) / (1 + u**2)
		self.__speed_target = np.sqrt(x**2 + y**2) * self.__speed_init
		# unwrap heading target
		prev_heading_target = self.__heading_target - self.__revolution * 360
		heading_target = np.arctan2(y, x) * 180/np.pi + self.__heading_init
		self.__heading_target = self.__unwrap_360(prev_heading_target, heading_target)

	def __unwrap_360(self, prev_value:float, value:float) -> float:
		"""
		Unwrap the value to the previous value. This is used to avoid discontinuities in the wind angle
		"""
		if np.abs(value - prev_value) > self.__unwrap_threshold:
			self.__revolution -= np.sign(value - prev_value)
		return value + self.__revolution * 360
		
	@property
	def heading(self):
		if self._speed < 0:
			return np.mod(self._heading + 180, 360)
		else:
			return np.mod(self._heading, 360)

	@property
	def speed(self):
		return np.abs(self._speed)
	
	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)
		
