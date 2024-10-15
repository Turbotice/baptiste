# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:07:03 2024

@author: Banquise
"""
#%% L LAMBDA
disp.figurejolie()
disp.joliplot(r'$\lambda$ (m)', r'l (m)', lambda_s, l_s, color = 17)

if save : 
    plt.savefig(save_path + 'l_lambda_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'l_lambda_png_' + tools.datetimenow() + '.png', dpi = 500)
    

    
def fct_x(x, a) :
    return a * x

lambda_plot= np.linspace(0, np.max(lambda_s)*1.05, 100)

popt, pcov = curve_fit(fct_x, lambda_s, l_s)

l_fit = lambda_plot * popt[0]

disp.joliplot(r'$\lambda$ (m)', r'l (m)', lambda_plot, l_fit, color = 8, exp = False)

plt.ylim(0,0.08)
plt.xlim(0, np.max(lambda_s)*1.05)

#%% KAPPA LAMBDA

disp.figurejolie()
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, k_s, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ $L_d$ ', lambda_s, k_s * L_d, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ h ', lambda_s, k_s * h, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'k $L_d$', lambda_s, L_d * 2 * np.pi / lambda_s, zeros = True, color = 8)

# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, 1 / (k_s**2*h**2 * lambda_s), zeros = True, color = 8)


disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', lambda_s, k_s, zeros = True, color = 8, log = True)
plt.axis('equal')

if save : 
    plt.savefig(save_path + 'kappac_lambda_' + tools.datetimenow() + '.pdf')
    plt.savefig(save_path + 'kappac_lambda_png_' + tools.datetimenow() + '.png', dpi = 500)





# kld = np.pi *2 / lambda_s * L_d
fits.fit_powerlaw(lambda_s, k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)')

plt.xlim(0.04, 0.6)
plt.ylim(4, 60)