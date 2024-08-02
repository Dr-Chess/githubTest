# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:18:46 2023

@author: chad.w.hess2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import timeit
import pandas as pd
from scipy.optimize import curve_fit
# from matplotlib.ticker import MultipleLocator
# from Ct_disk_plotter import get_lambda_and_Ct

ti = timeit.default_timer()  # initial time
#%% parameters

rho = 1.225  # kg/m^3
R = 8.1778  # m
m = 7375  # kgs
T = m * 9.81  # N
N = 4  # number of blades
c = 0.5334  # chord [m]
# RPM = 300
# omega = RPM * 2 * np.pi / 60  # rad/s
omega = 27  # rad/s
k = 1.15  # induced power factor (usually ~1.15 (1 is ideal))
Cd = 0.01  # coefficient of drag
Cd0 = Cd  # mean drag coefficient -> approximation but is good enough
Cla = 5.73  # make this in rad of the angles are in rad, same with deg

# derrived parameters
A = np.pi * R**2  # m^2
sigma = N * c / (np.pi * R)
r = np.linspace(0.001, R, 1000, endpoint=True)
r_bar = r/R
v_tip = omega * R
theta_0 = np.deg2rad(15)  # rad
theta_tw = np.deg2rad(-5)  # rad
theta = theta_0 + theta_tw * r_bar  # total theta (potentially as a function of r if there is twist)
v_h = np.sqrt(T / (2 * rho * A))  # in hover
psi = np.radians(np.linspace(0, 360, 1000, endpoint=True))  # converts from deg to rad
alpha_fp = np.deg2rad(4)  # converted into rad from deg
r_real = np.linspace(0.2*R, R, 1000, endpoint=True)
theta_real = theta_0 + theta_tw * r_real/R
phi = np.arctan(v_h / (r_real * omega))
theta_75 = theta_0 + theta_tw * 0.75

''' Select airfoil '''
df = pd.read_csv('xf-naca0008-il-1000000.csv')  # NACA 0008
# df = pd.read_csv('xf-n0012-il-1000000.csv')  # NACA 0012
# df = pd.read_csv('xf-naca2412-il-1000000.csv')  # NACA 2412
# df = pd.read_csv('NASA(6)-8 polars.csv')  # NASA(6)-8 (from PSP rotor)

def Cl_func(x, A, B, C, D, E, F, G):
    '''used in the get_Cl(r) function'''
    y = A*x**6 + B*x**5 + C*x**4 + D*x**3 + E*x**2 + F*x + G
    return y

def get_Cla(df):
    alpha = theta_real - phi
    alpha = np.linspace(min(alpha), max(alpha), 1000)
    parameters, covariance = curve_fit(Cl_func, df.loc[:, 'Alpha'], df.loc[:, 'Cl'])
    fit_A = parameters[0]
    fit_B = parameters[1]
    fit_C = parameters[2]
    fit_D = parameters[3]
    fit_E = parameters[4]
    fit_F = parameters[5]
    fit_G = parameters[6]
    Cl = Cl_func(alpha, fit_A, fit_B, fit_C, fit_D, fit_E, fit_F, fit_G)
    Cla = (max(Cl) - min(Cl)) / (max(alpha) - min(alpha))
    print(max(alpha), min(alpha))
    plt.plot(alpha, Cl)
    
    return Cla
Cla = get_Cla(df)
# print(Cla)

#%% get lambda

def get_lambda_and_Ct(mu):
    Ct = 0.01  # guess
    if mu >= 0.1:
        Lambda = Ct / (2 * mu)  # guess -> for high speed (mu >= 0.1)
        # lambda_i = Lambda
    else:
        Lambda = np.sqrt(Ct / 2)  # guess -> for low speed (mu < 0.1)
        # lambda_i = Lambda
        
    error = 1  # initialize
    tol = 0.0001
    
    func = lambda r_bar: sigma * Cla / 2 * ((theta_0 + theta_tw * r_bar) * r_bar**2 - Lambda * r_bar)
    # func = lambda r_bar: (theta_0 + theta_tw * r_bar) * r_bar**2 - Lambda * r_bar
    
    while error > tol:
        Ct = quad(func, 0, 1)[0] # this gives the integral of "func" for r_bar from 0-1
        # T = 1/2 * Cla * c * N * quad(func, 0, 1)[0]
        # Ct = T / (rho * v_tip**2 * np.pi * R**2)
        # Ct = sigma * Cla / 2 * (theta_75/3 - Lambda/2)
        # lambda_i = Ct / (2 * np.sqrt(mu**2 + Lambda**2))
        Lambda_new = mu * np.tan(alpha_fp) + Ct / (2 * np.sqrt(mu**2 + Lambda**2))
        # if abs(Lambda_new - Lambda) < tol and abs(Ct_new - Ct) < tol:
        #     break
        error = abs(Lambda_new - Lambda) / Lambda
        Lambda = (Lambda_new + Lambda) / 2
        # Ct = Ct_new
     
    return Lambda, Ct

def get_lambda(Ct, mu):
    if mu >= 0.1:
        Lambda = Ct / (2 * mu)  # guess -> for high speed (mu >= 0.1)
    else:
        Lambda = np.sqrt(Ct / 2)  # guess -> for low speed (mu < 0.1)
        
    error = 1  # initialize
    tol = 0.0001
    while error > tol:
        Lambda_new = mu * np.tan(alpha_fp) + Ct / (2 * np.sqrt(mu**2 + Lambda**2))
        error = abs(Lambda_new - Lambda) / Lambda
        Lambda = (Lambda_new + Lambda) / 2
    return Lambda

#%% Cp

mu_range = np.linspace(0.001, 0.35, 1000)  # v / v_tip
# v = mu * v_tip
# P = Cp * rho * A * v_tip**3

# three types of power: induced power (Pi), profile power (P0), and parasite power (Pp)
# P = Pi + P0 + Pp
# Cp = Cpi + Cp0 + Cpp

# Cpi: induced power is power required to produce rotor thrust
k = 1.15  # this factor accounts for nonuniform flow and tip loss
# f = 0.015 * A  # flat plate drag area (old helis: 0.025, modern: 0.01-0.015, clean: 0.004-0.008)
f = 3.78  # m^2 (for the baseline blackhawk as discussed in my paper)
# f = 4
# Lambda = np.sqrt(Ct / 2)  # for hover ONLY
Cp = np.zeros_like(mu_range)
Cpi = np.zeros_like(mu_range)
Cp0 = np.zeros_like(mu_range)
Cpp = np.zeros_like(mu_range)
lambda_matrix = np.zeros_like(mu_range)
# Ct_matrix = np.zeros_like(mu_range)
for i, mu in enumerate(mu_range):
    Ct = 0.008 # set this for now
    # Lambda = .065
    # print(mu * np.tan(alpha_fp) + Ct / (2 * np.sqrt(mu**2 + Lambda**2)))
    # mu_x = mu * np.cos(alpha_fp)
    # mu_y = mu * np.sin(alpha_fp)
    # Lambda, Ct = get_lambda_and_Ct(mu) 
    # Ct_matrix[i] = Ct
    Lambda = get_lambda(Ct, mu)
    # lambda_matrix[i] = Lambda
    # Ct = sigma*Cla/2 * (theta_0 * (1/3 + mu_x**2 / 2) + theta_tw * (1/4 + mu_x**2 / 2) - Lambda/2)   
    lambda_i = Ct / (2 * np.sqrt(mu**2 + Lambda**2))
    # print('mu:', mu)
    # print('Ct:', Ct)
    # print('Lambda:', Lambda)
    # print('lambda_i:', lambda_i)
    # print('Ct =', Ct, ' lambda =', Lambda)
    Cpi[i] = k * lambda_i * Ct

    # Cp0: profile power is a power required to turn the rotor in the air (due to viscous drag)
    Cp0[i] = sigma*Cd0/8 * (1 + 4.6 * mu**2)  # based on approximations from the APG paper

    # Cpp: parasite power is a power required to overcome the drag of the helicopter
    # D = 1/2 * rho * v**2 * f  # drag force acting on the helicopter
    # Pp = D * v
    # Cpp = 1/2 * v**3/(v_tip**3) * f/A  # same as the Cpp equation below
    # f_A = 0.018
    # f = A * f_A
    Cpp[i] = 1/2 * f/A * mu**3
    
    Cp[i] = Cpi[i] + Cp0[i] + Cpp[i]
    
#%% P

P = np.zeros_like(mu_range)
Pi = np.zeros_like(mu_range)
P0 = np.zeros_like(mu_range)
Pp = np.zeros_like(mu_range)
normalization = rho * A * v_tip**3
for i, mu in enumerate(mu_range):
    Pi[i] = Cpi[i] * normalization
    P0[i] = Cp0[i] * normalization
    Pp[i] = Cpp[i] * normalization
    P[i] = Pi[i] + P0[i] + Pp[i]
    
#%% plotting

plot_Cp_curves = True
if plot_Cp_curves:
    fig, ax = plt.subplots()
    ax.plot(mu_range, Cpi, label='Cpi', color='goldenrod')
    ax.plot(mu_range, Cp0, label='Cp0', color='blue')
    ax.plot(mu_range, Cpp, label='Cpp', color='red')
    ax.plot(mu_range, Cp, label='Cp total', color='forestgreen')
    def annot_P_min(x,y, ax=None):
        xmin = x[np.argmin(y)]
        ymin = min(y)
        text= "$\mu_{{BE}}$: $\mu$={:.3f}, $C_P$={:.2e}".format(xmin, ymin)
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="arc3")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmin, ymin), xytext=(0.5,0.96), **kw)
    
    def plot_best_range(x,y):
        x_br = (k * Ct**2 / (f/A))**(1/4)  # assuming that d(Cp0)/dmu is negligibly small
        lambda_i = Ct / (2 * np.sqrt(x_br**2 + Lambda**2))
        Cpi = k * lambda_i * Ct
        Cp0 = sigma*Cd0/8 * (1 + 4.6 * x_br**2)
        Cpp = 1/2 * f/A * x_br**3
        y_br = Cpi + Cp0 + Cpp
        ax.plot(x_br, y_br, 'o', color='orchid')
        mu_maxtrix = np.linspace(0, x_br)
        y_matrix = y_br/x_br * mu_maxtrix
        ax.plot(mu_maxtrix, y_matrix, linestyle='dotted', color='firebrick', label='Best Range')
        
        text= "$\mu_{{BR}}$: $\mu$={:.3f}, $C_P$={:.2e}".format(x_br, y_br)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="arc")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(x_br, y_br), xytext=(0.78,0.87), **kw)
        
    ax.plot(mu_range[np.argmin(Cp)], min(Cp), 'o', color='orchid')
    annot_P_min(mu_range, Cp)
    plot_best_range(mu_range, Cp)
    ax.legend(loc='lower right')
    ax.set_title(f'Coefficients of power ($C_P$) vs advance ratio ($\mu$): $C_T$ = {round(Ct, 4)}')
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$C_P$')
    # ax.set_ylim(0, 100)
    # ax.yaxis.set_major_locator(MultipleLocator(10))  # to label every multiple of whatever is in the ()
    
plot_P_curves = False
if plot_P_curves:
    fig, ax = plt.subplots()
    ax.plot(mu_range, Pi, label='Pi', color='goldenrod')
    ax.plot(mu_range, P0, label='P0', color='blue')
    ax.plot(mu_range, Pp, label='Pp', color='red')
    ax.plot(mu_range, P, label='P total', color='forestgreen')
    def annot_P_min(x,y, ax=None):
        xmin = x[np.argmin(y)]
        ymin = min(y)
        text= "$\mu_{{BE}}$: $\mu$={:.3f}, $P$={:.2e}".format(xmin, ymin)
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="arc3")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmin, ymin), xytext=(0.5,0.96), **kw)
    
    def plot_best_range(x,y):
        x_br = (k * Ct**2 / (f/A))**(1/4)  # assuming that d(Cp0)/dmu is negligibly small
        lambda_i = Ct / (2 * np.sqrt(x_br**2 + Lambda**2))
        Pi = k * lambda_i * Ct * normalization
        P0 = sigma*Cd0/8 * (1 + 4.6 * x_br**2) * normalization
        Pp = 1/2 * f/A * x_br**3 * normalization
        y_br = Pi + P0 + Pp
        ax.plot(x_br, y_br, 'o', color='orchid')
        mu_maxtrix = np.linspace(0, x_br)
        y_matrix = y_br/x_br * mu_maxtrix
        ax.plot(mu_maxtrix, y_matrix, linestyle='dotted', color='firebrick', label='Best Range')
        
        text= "$\mu_{{BR}}$: $\mu$={:.3f}, $P$={:.2e}".format(x_br, y_br)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="arc")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(x_br, y_br), xytext=(0.78,0.87), **kw)
        
    ax.plot(mu_range[np.argmin(P)], min(P), 'o', color='orchid')
    annot_P_min(mu_range, P)
    plot_best_range(mu_range, P)
    ax.legend(loc='lower right')
    ax.set_title(f'Power required vs advance ratio ($\mu$): $C_T$ = {round(Ct, 4)}')
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$P$')
    # ax.set_ylim(0, 100)
    # ax.yaxis.set_major_locator(MultipleLocator(10))  # to label every multiple of whatever is in the ()
    
#%% extra data

tf = timeit.default_timer() # final time
print("total walltime:", tf - ti)

T = Ct * rho * v_tip**2 * A
P_ideal = T * v_h  # in hover
FM = P_ideal / P[0]  # figure of merit in hover
print('FM in hover:', FM)

print('Ct:', Ct)

# plt.plot(mu_range, Ct_matrix)
# plt.xlabel('mu')
# plt.ylabel('Ct')
# plt.title('Ct vs mu')

print("test update: successful")








