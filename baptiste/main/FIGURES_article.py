# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:07:03 2024

@author: Banquise
"""
#%% L LAMBDA
disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$\lambda$ (m)', r'l (m)', lambda_s * 100, l_s * 100, color = 18, width = 8.6 * 4 / 5)

    


    
def fct_x(x, a) :
    return a * x

lambda_plot= np.linspace(0, np.max(lambda_s)*1.05, 1000)

popt, pcov = curve_fit(fct_x, lambda_s, l_s)

l_fit = lambda_plot * popt[0]

disp.joliplot(r'$\lambda$ (cm)', r'$L_{\kappa}$ (cm)', lambda_plot * 100, l_fit * 100, color = 8, exp = False, legend = r'$L_{kappa} = \alpha \lambda$')
plt.errorbar(lambda_s * 100, l_s * 100, yerr = err_lkappa*100, fmt = 'None', ecolor = disp.vcolors(4), linewidth = 1.1)

plt.ylim(0,0.057 * 100)
plt.xlim(0, np.max(lambda_s)*1.05 * 100)



#%% KAPPA LAMBDA

"""Tentative avec kappa h"""
E = 65e6
hh = (L_d**(4/3) * (10 * 1000 * 9.81)**0.33) / E**0.33
# def fct_kh(x,a):
#     return a * x**(-1/2)
# disp.figurejolie()   
# popt, pcov = fits.fit(fct_kh, lambda_s, k_s * hh, display = True, err = False, nb_param = 1, p0 = [0], bounds = False, 
#         zero = False, th_params = False, xlabel = r'k (m$^{-1}$)', ylabel = r'$\omega$', legend_data = r'Experimental Data', 
#         legend_fit = 'h = ', log = False)


# disp.figurejolie(width = 8.6 * 5 / 6)
# x_lam  = np.linspace(np.min(lambda_s), np.max(lambda_s), 1000)
# disp.joliplot('$\lambda$ (m)', r'$\kappa_c h$', lambda_s, k_s * hh, zeros = False, color = 19, log = True, width = 8.6 * 5 / 6 )
# disp.joliplot('$\lambda$ (m)', r'$\kappa_c h$', x_lam, fct_kh(x_lam, popt - 0.0001), zeros = False, color = 8, log = True, width = 8.6 * 5 / 6, exp = False )

# disp.figurejolie(width = 8.6 * 5 / 6)
# disp.joliplot('$\lambda$ (m)', r'$\kappa_c h$', lambda_s, k_s ** 2 / hh * l_s, zeros = False, color = 19, log = True, width = 8.6 * 5 / 6 )

err_kappa_combinee = np.zeros(10)
for i in  range (10):
    err_kappa_combinee[i] = np.max( (err_kappa_proche[i], erreur_kappa[i]))

err_kappa_combinee[-1] = np.sqrt(np.diag(err_kappa[-1]))[0] / popt_kappa[-1][0] * k_s[-1]
cst = 0.0002778
alpha = 0.079

disp.figurejolie(width = 8.6 * 1.4, height = 8.6 * 1.0)

disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', lambda_s, k_s, zeros = False, color = 18, log = True, width =8.6 * 1.1 )
plt.xlim(0.03,0.8)
plt.ylim(3, 60)
plt.axis('equal')

x_lam = np.linspace (np.min(lambda_s), np.max (lambda_s), 100)
y_lam = np.sqrt(cst / np.mean(h) / alpha) / np.sqrt(x_lam)

disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', x_lam, y_lam, zeros = False, color = 8, log = True, width = 8.6 *1.1, exp = False, legend = r'$\kappa_c = \beta \lambda^{-\frac{1}{2}}$' )

plt.xlim(0.04, 0.7)
plt.ylim(4, 70)
plt.errorbar(lambda_s, k_s, yerr = err_kappa_combinee, fmt = 'None', ecolor = disp.vcolors(4), linewidth = 1.1)

disp.figurejolie(width = 8.6 , height = 8.6*5/6)
E = 65e6
hh = (L_d**(4/3) * (10 * 1000 * 9.81)**0.33) / E**0.33
disp.joliplot('$\lambda$ (m)', r'$\kappa_c h$', lambda_s, k_s * h, zeros = False, cm = 4, log = True, width = 8.6 * 5 /5 )
plt.xlim(0.04,0.7)
plt.ylim(3.5e-4,4e-3)


x_lam = np.linspace (np.min(lambda_s), np.max (lambda_s), 100)
y_lam = np.sqrt(cst / np.mean(h) / alpha) / np.sqrt(x_lam)

disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', x_lam, y_lam, zeros = False, color = 8, log = True, width = 8.6 * 5 / 6, exp = False, legend = r'$\kappa_c = \beta \lambda^{-\frac{1}{2}}$' )

# plt.xlim(0.04, 0.6)
# plt.ylim(4, 60)
plt.errorbar(lambda_s, k_s * hh, yerr = erreur_kappa / k_s * k_s * hh, fmt = 'None', ecolor = '#990000')


"""AVEC ERREUR mesure et h"""
err_hetkappa = np.sqrt((erreur_kappa / k_s)**2 + 0.14**2)
fits.fit_powerlaw( l_s, k_s * h, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c h$', width = 8.6 * 5 / 6 , color = 19)
# plt.ylim(3.5e-4,4e-3)
# plt.xlim(0.04,0.6)
plt.errorbar( l_s, k_s * hh, yerr = err_hetkappa * k_s * hh, fmt = 'None', ecolor = '#990000')

"""AVEC ERREUR"""

erreur_tot = np.zeros(len(a_s))

for i in range (len(a_s)):
    err_a = np.sqrt(np.diag(err_kappa[i]))[0] * a_s[i]**2
    err_b = np.sqrt(np.diag(err_kappa[i]))[1] * a_s[i]
    erreur_tot[i] = err_a + err_b
    
fits.fit_powerlaw(lambda_s, k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)', width = 8.6 * 5 / 6 )
# plt.figure()
# plt.plot(l_s, k_s, 'rx')
# plt.errorbar(lambda_s, k_s, yerr = erreur_tot, fmt = 'r.')
plt.errorbar(lambda_s, k_s, yerr = erreur_kappa, fmt = 'None', linewidth = 1)

plt.ylim(0,30)



err_Ac = 0.00025
err_k_1 = k_s * err_Ac / a_s
err_k_2 = np.zeros(len(k_s))
for i in range (len(a_s)) :
    aaa = popt_kappa[i][0]
    bbb = popt_kappa[i][1]
    err_k_2[i] = (aaa * ( (a_s[i] + err_Ac)**2 - (a_s[i] - err_Ac)**2 ) + bbb * ( (a_s[i] + err_Ac) - (a_s[i] - err_Ac) ))



fits.fit_powerlaw(L_d, k_s * hh, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)', width = 8.6 * 5 / 6 )
# plt.figure()
# plt.plot(l_s, k_s, 'rx')
plt.errorbar(lambda_s, k_s, yerr = erreur_kappa, fmt = 'none', ecolor = '#990000')


disp.figurejolie(width = 8.6 * 5 / 6)
E = 65e6
hh = (L_d**(4/3) * (10 * 1000 * 9.81)**0.33) / E**0.33
disp.joliplot('$kL_d$ (m)', r'$\kappa_c$ (m$^{-1}$)',  2 * np.pi * L_d / lambda_s,  k_s * hh , zeros = True, color = 2, log = True, width = 8.6 * 5 / 6 )


po, pc = fits.fit_powerlaw(lambda_s,  k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)', width = 8.6 * 5 / 6 )


def fct(x, a) : 
    return a * x**(-0.5)

popt, pcov = curve_fit(fct, lambda_s, k_s)


#%% kappa avec Ac et Lk

cst = 0.0002778
alpha = 0.079

kappa_cc = a_s / l_s**2

err_Ac = 0.00025
err_k_1 = kappa_cc * err_Ac / a_s
err_k_2 = np.zeros(len(k_s))
for i in range (len(a_s)) :
    aaa = popt_kappa[i][0]
    bbb = popt_kappa[i][1]
    err_k_2[i] = (aaa * ( (a_s[i] + err_Ac)**2 - (a_s[i] - err_Ac)**2 ) + bbb * ( (a_s[i] + err_Ac) - (a_s[i] - err_Ac) ))

disp.figurejolie(width = 8.6 * 5 / 6)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, k_s, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ $L_d$ ', lambda_s, k_s * L_d, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ h ', lambda_s, k_s * h, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'k $L_d$', lambda_s, L_d * 2 * np.pi / lambda_s, zeros = True, color = 8)

# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, 1 / (k_s**2*h**2 * lambda_s), zeros = True, color = 8)

disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', lambda_s, k_s, zeros = False, color = 5, log = True, width = 8.6 * 5 / 6 )
disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', lambda_s, kappa_cc, zeros = False, color = 2, log = True, width = 8.6 * 5 / 6 )
plt.axis('equal')


# kld = np.pi *2 / lambda_s * L_d
fits.fit_powerlaw(lambda_s, kappa_cc, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)', width = 8.6 * 5 / 6 )


#%% Ac (lambda)

uuu = fits.fit_powerlaw(lambda_s, a_s, display = True, legend = 'Threshold', xlabel =r"$\lambda$ (m)", ylabel = r"A$_c$ (m)" )

plt.xlim(0.04, 1)
plt.ylim(0.001, 0.025)

disp.figurejolie(width = 8.6 * 1.4/2)
disp.joliplot(r'$\lambda$ (m)', '$A$ (m)', lambda_s, a_s, width = 8.6, color = 2, exp = True, log = True)
lllamb = np.linspace(np.min(long_onde), np.max(long_onde), 2000)
disp.joliplot(r'$\lambda$ (m)', '$A$ (m)', lllamb, lllamb ** uuu[0][0] * np.exp(uuu[0][1]), color = 8, exp = False, log = True, width = 10)

plt.xlim(0.04, 1)
plt.ylim(0.001, 0.025)

#%% Lcrack (A)
u = -3
if display :
    disp.figurejolie(width = 8.6/2 * 1.2)
    disp.joliplot(r"Amp (cm)", r"$L_{crack}$ (cm)", ampp_lc0[u] *100, lcrack_lc0[u] *100 , color = 8, zeros = True, width = 8.6*0.8)

popt = np.polyfit(ampp[u]*100, lcracktot[u]*100, 1, full = True)
a[u] = popt[0][0]

b[u] = popt[0][1]
#(long_onde[u] - b[u]) / a[u] 
if len (ampp[u]) > 2 :
    erreur[u] = popt[1:][0][0]
elif len (ampp[u]) == 2 :
    erreur[u] = 0#(ampp[1] - amp_s[u])/ampp[1] / 2

x_amp = np.linspace(-200, 200, 1000)
if display :
    disp.joliplot(r"Amplitude (m)", r"$L_{crack}$ (m)", x_amp, x_amp * a[u] + b[u] , color = 5, exp = False, width = 8.6 * 0.7)
    
zeroo = np.linspace(0, 0, 1000)
plt.plot(x_amp, zeroo)
disp.joliplot(r"Amplitude (cm)", r"$L_{crack}$ (cm)", x_amp, zeroo, color = 5, exp = False, width = 8.6)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

plt.xlim([0, np.max(ampp_lc0[u]) * 1.05 * 100 ])
plt.ylim([-np.max(lcrack_lc0[u]) / 10 * 100, np.max(lcrack_lc0[u]) * 1.1 * 100 ]) 

#%% G (lambda)
def fct_x(x, a) :
    return a * x

popt, pcov = curve_fit(fct_x, lambda_s, l_s)

l_fit = lambda_s * popt[0]

disp.figurejolie(width = 8.6 * 4 /5)
disp.joliplot(r'$\lambda$ (cm)', r'$\kappa_c^{2}L_{\kappa} D / h$ (J.m$^{-2}$)', lambda_s * 100, l_fit * k_s**2 * np.mean(D) /np.mean(h),
              log = False, color = 18, zeros = True, width = 8.6 * 4 / 6)

# plt.ylim([0,0.39])
# plt.xlim([0,0.58 * 100])

stdd = np.std(l_fit * k_s**2 * np.mean(D) /np.mean(h))
yy = np.linspace(0,0,100) + np.mean(l_fit * k_s**2 * np.mean(D) /np.mean(h))

xx = np.linspace(0,0.6,100) * 100

yy_sp = np.linspace(0,0,100) + np.mean(l_fit * k_s**2 * np.mean(D) /np.mean(h)) + stdd

yy_sm = np.linspace(0,0,100) + np.mean(l_fit * k_s**2 * np.mean(D) /np.mean(h)) - stdd
plt.plot(xx,yy, 'k-', linewidth = 1)
plt.plot(xx,yy_sp, 'k--', linewidth = 1)
plt.plot(xx,yy_sm, 'k--', linewidth = 1)

#%% Encart 4 a kappa x


disp.figurejolie(width =  8.6 / 2.5)
disp.joliplot(r'x (m)', '$\eta$ (mm)', x_plotexp , forme * 1000, exp = False, color = 14, width =  8.6 / 2)

disp.joliplot('x (m)', r'$\kappa^{2}$ (m$^{-1}$)', x_kappa, popt_x_mean, exp = False, color = 8, width =  8.6 / 2)
# disp.joliplot('x (m)', r'$\kappa^{2}$ (m$^{-1}$)', x_kappa, popt_x ** 2, exp = False, color = 2)
disp.joliplot('x (m)', r'$\kappa^{2}$ (m$^{-1}$)', x_kappa, popt_x, exp = False, color = 2, width =  8.6 / 2)
fig, ax1 = plt.subplots(figsize = disp.set_size(width = 8.6, fraction = 1, subplots = (1,1)))

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  

ax1.plot(x_plotexp * 100, forme * 1000, color = '#990000')
ax2.plot( (x_kappa - 0.003) * 100, popt_x_mean, color = disp.vcolors(2))
   
ax2.set_ylabel(r'$\kappa$ (m$^{-1}$)', color = disp.vcolors(2))
ax1.set_ylabel(r'$\eta$ (mm)', color = '#990000') 
ax1.set_xlabel(r'$x$ (cm)') 
   
# ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax1.ticklabel_format(axis='x', style="sci", scilimits=(0,0))
# ax2.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
plt.show()

#%% 4 c
# k_s = kappa_cc

hh = (L_d**(4/3) * (10 * 1000 * 9.81)**0.33) / E**0.33

# fits.fit_powerlaw( (1/l_s),k_s**2 * np.mean(h)**2, display = True, xlabel = r'$1/L_{kappa}$ (m$^{-1}$)', ylabel = r'$(\kappa_c h)^2$', legend = '', new_fig = True, fit = 'poly', color = False)

fits.fit_powerlaw( 2 * np.pi* L_d/ lambda_s,  k_s**2 * h**0 * l_s**1 / D**-1  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)




disp.figurejolie()
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_d / lambda_s,k_s**2 * np.mean(h) * l_s, log = False, color = 18, zeros = True, width = 8.6, title = r'kappa2 h Lk (kLd)')


alpha = np.linspace(-3,3,100)
uu = []
errr = []
for aa in alpha :
    popt = fits.fit_powerlaw( 2 * np.pi * L_d / lambda_s, k_s**1 * h**0.5 * l_s**0.5 / D**0.25, display = False, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 h L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)
    uu += [popt[0][0]]
    errr += [np.sqrt(np.diag(popt[1]))[0]]
plt.figure()
plt.plot(alpha, uu)
plt.hlines(0, -3, 3, color = 'red')
plt.figure()
plt.plot(alpha, errr)



'''4 c top'''
disp.figurejolie(width = 8.6 * 3 / 4)
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_d / lambda_s, k_s**2 * np.mean(h) * l_s , log = False, color = 18, zeros = False, width = 8.6*2/3)

plt.xscale('log')
plt.xlim (10**-2,1)
plt.ylim(0,8.2e-4)

err_tot = np.sqrt( (0.14)**2 + (2 *err_kappa_combinee/k_s)**2 + (err_lkappa / l_s)**2 )
plt.errorbar(2 * np.pi * L_d / lambda_s, k_s**2 * np.mean(h)  * l_s, yerr = err_tot * k_s**2 *np.mean(h)  * l_s, fmt = 'none', ecolor =  disp.vcolors(4))

stdd = np.std(k_s**2 * np.mean(h) * l_s)
yy = np.linspace(0,0,100) + np.mean(k_s**2 * np.mean(h) * l_s)
xx = np.linspace(0,1,100) 
yy_sp = np.linspace(0,0,100) + np.mean(k_s**2 * np.mean(h) * l_s) + stdd
yy_sm = np.linspace(0,0,100) + np.mean(k_s**2 * np.mean(h) * l_s) - stdd

plt.plot(xx,yy, 'k-', linewidth = 1)
plt.plot(xx,yy_sp, 'k--', linewidth = 1)
plt.plot(xx,yy_sm, 'k--', linewidth = 1)




fits.fit_powerlaw( 2 * np.pi* L_d/ lambda_s,  k_s**2 * h**0 * l_s**1 / D**-1  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)

plt.errorbar(2 * np.pi * L_d / lambda_s, k_s**2 * np.mean(h)**0  * l_s * D, yerr = err_tot *k_s**2 * np.mean(h)**0  * l_s * D, fmt = 'none', ecolor =  '#990000')


''' Gc (lambda) et Gc (kLd)'''

fits.fit_powerlaw( 2 * np.pi* L_d/ lambda_s,  k_s**2 * hh**-1 * l_s**1 * D  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa} / h$', legend = '', new_fig = True, fit = 'poly', color = False)

plt.errorbar(2 * np.pi * L_d / lambda_s, k_s**2 * hh**-1 * l_s**1 * D, yerr = err_tot * k_s**2 * hh**-1 * l_s**1 * D, fmt = 'none', ecolor =  '#990000')



fits.fit_powerlaw(  L_d,  k_s**2 * hh**-1 * l_s**1 * D  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa} / h$', legend = '', new_fig = True, fit = 'poly', color = False)

plt.errorbar(lambda_s, k_s**2 * hh**-1 * l_s**1 * D, yerr = err_tot * k_s**2 * hh**-1 * l_s**1 * D, fmt = 'none', ecolor =  '#990000')

disp.figurejolie(width = 8.6 * 3 / 4)
disp.joliplot(r'$\lambda$',r'$L_d$', lambda_s ,  L_d,  log = False, color = 19, zeros = False, width = 8.6*2/3)
# plt.xscale('log')
plt.errorbar(  L_d, k_s**2 * hh**-1 * l_s**1 * D, yerr = err_tot * k_s**2 * hh**-1 * l_s**1 * D, fmt = 'none', ecolor =  '#990000')



#%% 4c avec kappa de A et Lk plus h de Ld et E

E = 65e6
hh = (L_d**(4/3) * (10 * 1000 * 9.81)**0.33) / E**0.33

L_dd = (E * hh**3 / (10 * 10000))**0.25

#Vrai L_d, h avec L_d
disp.figurejolie(width = 8.6 )
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_d / lambda_s, k_s**2 * l_s / h * D , log = False, color = 17, zeros = False, width = 8.6*2/3)
# plt.xscale('log')
# plt.xlim (10**-2,1)
# plt.ylim(0,8.2e-4)

disp.figurejolie(width = 8.6 )
disp.joliplot(r'$\lambda$', r'$\sigma_{\kappa}$', lambda_s, k_s**2 * l_s * h , log = False, cm = 6, zeros = False, width = 8.6)
plt.axhline(y=1.4e-3 * E, color = disp.mcolors(3))
# plt.axhline(y=1.4e-3)
fits.fit_powerlaw( lambda_s,   k_s * h * E  , display = True, xlabel = r'$\lambda$', ylabel = r'$\sigma_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)
plt.axhline(y=1.4e-3 * E)

#L_d avec h, vrai h
disp.figurejolie(width = 8.6 )
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_dd / lambda_s, k_s**2 * hh * l_s , log = False, color = 17, zeros = False, width = 8.6*2/3)
plt.xscale('log')

#Vrai L_d et h
disp.figurejolie(width = 8.6 )
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_d / lambda_s, k_s**2 * hh * l_s , log = False, color = 17, zeros = False, width = 8.6*2/3)
plt.xscale('log')

#Gc (kLd), h avec L_d
disp.figurejolie(width = 8.6 )
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_d / lambda_s, k_s**2 / h * l_s * D , log = False, color = 17, zeros = False, width = 8.6*2/3)
plt.xscale('log')

fits.fit_powerlaw( k_s[:-1],  k_s[:-1]**2 * l_s[:-1] * np.mean(hh)  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)

fits.fit_powerlaw( l_s,  k_s, display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)


# err_h = h * dico['variables_globales']['CM']['err_CM'] / dico['variables_globales']['CM']['CM']
err_h = np.std(h) 
err_h= 0.14

err_tot = np.sqrt( (0.14)**2 + (err_k_2/k_s)**2 )

plt.errorbar(2 * np.pi * L_d / lambda_s, k_s**2 * h  * l_s, yerr = err_tot * k_s**2 * h  * l_s, fmt = 'none', ecolor =  disp.vcolors(4))

fits.fit_powerlaw( 2 * np.pi* L_d/ lambda_s,  k_s**2 * h  * l_s  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)


disp.figurejolie(width = 8.6 * 3 / 4)
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_d / lambda_s, k_s**2 * D * l_s / h , log = True, color = 17, zeros = False, width = 8.6*2/3)


fits.fit_powerlaw( 2 * np.pi / lambda_s,  k_s , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)

fits.fit_powerlaw(lambda_s, k_s**2 * h * l_s  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)



G_c = k_s**2 * D * l_s / hh

disp.figurejolie(width = 8.6)  
for i in range(10):
    colors = disp.mcolors( int(0 / np.max(G_c) * 9))

    plt.scatter( 2 * np.pi / lambda_s[i] * L_d[i], G_c[i] , color = colors, marker = 'x', s = 3, linewidths= 10)
plt.plot(0,0)
# plt.xscale('log')














