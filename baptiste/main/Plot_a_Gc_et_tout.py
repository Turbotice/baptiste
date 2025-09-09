# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:38:58 2025

@author: Banquise
"""


long_onde_top, pente_a = tools.sort_listes(long_onde, a)
lambda_s_top, kappa_c = tools.sort_listes(lambda_s, k_s)
lambda_s_top, Lkappa = tools.sort_listes(lambda_s, l_s)
lambda_s_top, D_top = tools.sort_listes(lambda_s, D)
lambda_s_top, h_top = tools.sort_listes(lambda_s, h)
lambda_s_top, Ld_top = tools.sort_listes(lambda_s, L_d)

Gcc = kappa_c**2  * D_top / 2 / h_top * Lkappa

E = 65e6
hhh = (Ld_top**(4/3) * (10 * 1000 * 9.81)**0.33) / E**0.33


disp.figurejolie(width = 8.6 * 4 / 5.2)
disp.joliplot( r'$\lambda$ (m)','pente $a$', long_onde_top, pente_a, cm = 4, log = True )


sigma_kappa = kappa_c * h_top * E
disp.figurejolie(width = 8.6 * 4.2 / 5)
disp.joliplot( r'$\lambda$ (m)','$\sigma_{kappa}$ (Pa)', long_onde_top, sigma_kappa, cm = 4, log = True )

x_lam = np.linspace(np.min(long_onde_top), np.max(long_onde_top), 100)

def fct_05(x,a) :
    return a * x ** -0.5


def fct_075(x,a) :
    return a * x ** -0.75

popt05, pcov05 = fits.fit(fct_05, long_onde_top, sigma_kappa,  legend_data = r'Experimental Data', legend_fit = '', log = False, cm = 2)

popt075, pcov075 = fits.fit(fct_075, long_onde_top, sigma_kappa, display = True, err = False, nb_param = 1, p0 = [0], bounds = False, 
        zero = False, th_params = False, xlabel = r'k (m$^{-1}$)', ylabel = r'$\omega$', legend_data = r'Experimental Data', 
        legend_fit = 'h = ', log = False, cm = 7)
disp.joliplot( r'$\lambda$ (m)','$\sigma_{kappa}$ (Pa)', long_onde_top, sigma_kappa, cm = 4, log = True )

plt.plot(0.33, 9e4, 'ro', ms = 8)

disp.figurejolie(width = 8.6 * 4 / 5.2)
disp.joliplot( r'$D$ (J.m$^{-2}$)','pente $a$', D_top, pente_a, cm = 4, log = True)


fits.fit_powerlaw(lambda_s_top[:-2], Gcc[:-2] * lambda_s_top[:-2] * pente_a[:-2], display = True, xlabel = r'$\lambda$ (m)', ylabel = r'1/Pente $a$', legend = '', new_fig = True, fit = 'poly', color = 18)


plt.ylim(3e-3, 1e-1)
plt.xlim(5e-7, 5e-5)





