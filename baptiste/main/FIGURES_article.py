# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:07:03 2024

@author: Banquise
"""
#%% L LAMBDA
disp.figurejolie(width = 8.6 * 4 / 5)
disp.joliplot(r'$\lambda$ (m)', r'l (m)', lambda_s * 100, l_s * 100, color = 17, width = 8.6 * 4 / 6)

    


    
def fct_x(x, a) :
    return a * x

lambda_plot= np.linspace(0, np.max(lambda_s)*1.05, 100)

popt, pcov = curve_fit(fct_x, lambda_s, l_s)

l_fit = lambda_plot * popt[0]

disp.joliplot(r'$\lambda$ (cm)', r'$L_{\kappa}$ (cm)', lambda_plot * 100, l_fit * 100, color = 8, exp = False)

plt.ylim(0,0.079 * 100)
plt.xlim(0, np.max(lambda_s)*1.05 * 100)

#%% KAPPA LAMBDA

cst = 0.0002778
alpha = 0.079

disp.figurejolie(width = 8.6 * 5 / 6)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, k_s, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ $L_d$ ', lambda_s, k_s * L_d, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'$\kappa$ h ', lambda_s, k_s * h, zeros = True, color = 8)
# disp.joliplot('$\lambda$ (m)', r'k $L_d$', lambda_s, L_d * 2 * np.pi / lambda_s, zeros = True, color = 8)

# disp.joliplot('$\lambda$ (m)', r'$\kappa$ (m$^{-1}$)', lambda_s, 1 / (k_s**2*h**2 * lambda_s), zeros = True, color = 8)


disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', lambda_s, k_s, zeros = False, color = 2, log = True, width = 8.6 * 5 / 6 )
plt.axis('equal')


# kld = np.pi *2 / lambda_s * L_d
fits.fit_powerlaw(lambda_s, k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)', width = 8.6 * 5 / 6 )

x_lam = np.linspace (np.min(lambda_s), np.max (lambda_s), 100)
y_lam = np.sqrt(cst / np.mean(h) / alpha) / np.sqrt(x_lam)

# disp.joliplot('$\lambda$ (m)', r'$\kappa_c$ (m$^{-1}$)', x_lam, y_lam, zeros = False, color = 8, log = True, width = 8.6 * 5 / 6, exp = False, legend = r'$\kappa_c = \beta \lambda^{-\frac{1}{2}}$' )

# plt.xlim(0.04, 0.6)
# plt.ylim(4, 60)

fits.fit_powerlaw(2 * np.pi * L_d / lambda_s, k_s * h, display = True, xlabel = '$k l_d$', ylabel = r'$\kappa_c h$', width = 8.6 * 5 / 6 )

"""AVEC ERREUR"""

erreur_tot = np.zeros(len(a_s))

for i in range (len(a_s)):
    err_a = np.sqrt(np.diag(err_kappa[i]))[0] * a_s[i]**2
    err_b = np.sqrt(np.diag(err_kappa[i]))[1] * a_s[i]
    erreur_tot[i] = err_a + err_b
    
fits.fit_powerlaw(lambda_s, k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)', width = 8.6 * 5 / 6 )
# plt.figure()
# plt.plot(l_s, k_s, 'rx')
plt.errorbar(lambda_s, k_s, yerr = erreur_tot, fmt = 'r.')

# plt.ylim(0,30)



err_Ac = 0.00025
err_k_1 = k_s * err_Ac / a_s
err_k_2 = np.zeros(len(k_s))
for i in range (len(a_s)) :
    aaa = popt_kappa[i][0]
    bbb = popt_kappa[i][1]
    err_k_2[i] = (aaa * ( (a_s[i] + err_Ac)**2 - (a_s[i] - err_Ac)**2 ) + bbb * ( (a_s[i] + err_Ac) - (a_s[i] - err_Ac) ))



fits.fit_powerlaw(lambda_s, k_s, display = True, xlabel = '$\lambda$ (m)', ylabel = r'$\kappa_c$ (m$^{-1}$)', width = 8.6 * 5 / 6 )
# plt.figure()
# plt.plot(l_s, k_s, 'rx')
plt.errorbar(lambda_s, k_s, yerr = err_k_2, fmt = 'none', ecolor = '#990000')


#%% Ac (lambda)

uuu = fits.fit_powerlaw(long_onde, amp_s, display = True, legend = 'Threshold', xlabel =r"$\lambda$ (m)", ylabel = r"A$_c$ (m)" )

plt.xlim(0.04, 1)
plt.ylim(0.001, 0.025)

disp.figurejolie(width = 8.6 * 1.4/2)
disp.joliplot(r'$\lambda$ (m)', '$A$ (m)', long_onde, amp_s, width = 8.6, color = 2, exp = True, log = True)
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
fig, ax1 = plt.subplots(figsize = set_size(width = 8.6, fraction = 1, subplots = (1,1)))

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

disp.figurejolie(width = 8.6 * 3 / 4)
disp.joliplot(r'$kL_d$', r'$\kappa_c^2 h L_{\kappa}$', 2 * np.pi * L_d / lambda_s, k_s**2 * np.mean(h) * l_s , log = False, color = 17, zeros = False, width = 8.6*2/3)
plt.xscale('log')
plt.xlim (10**-2,1)
plt.ylim(0,8.2e-4)

err_h = h * dico['variables_globales']['CM']['err_CM'] / dico['variables_globales']['CM']['CM']
err_h = np.std(h) 


err_tot = np.sqrt( (err_h/h)**2 + (err_k_2/k_s)**2 )

plt.errorbar(2 * np.pi * L_d / lambda_s, k_s**2 * np.mean(h)  * l_s, yerr = err_tot * k_s**2 *np.mean(h)  * l_s, fmt = 'none', ecolor =  disp.vcolors(4))
# plt.errorbar(2 * np.pi * L_d / lambda_s, k_s**2 * h * l_s, yerr = err_h * k_s**2 * l_s, fmt = 'b.' )

# stdd = np.std(k_s**2 * np.mean(h) * l_s)
# yy = np.linspace(0,0,100) + np.mean(k_s**2 * np.mean(h) * l_s)

# xx = np.linspace(0,1,100) 

# yy_sp = np.linspace(0,0,100) + np.mean(k_s**2 * np.mean(h) * l_s) + stdd

# yy_sm = np.linspace(0,0,100) + np.mean(k_s**2 * np.mean(h) * l_s) - stdd
# plt.plot(xx,yy, 'k-', linewidth = 1)
# plt.plot(xx,yy_sp, 'k--', linewidth = 1)
# plt.plot(xx,yy_sm, 'k--', linewidth = 1)
fits.fit_powerlaw( 2 * np.pi* L_d/ lambda_s,  k_s**2 * h**0 * l_s**1 / D**-1  , display = True, xlabel = r'$kL_d$', ylabel = r'$\kappa_c^2 D L_{\kappa}$', legend = '', new_fig = True, fit = 'poly', color = False)

plt.errorbar(2 * np.pi * L_d / lambda_s, k_s**2 * np.mean(h)**0  * l_s * D, yerr = err_tot *k_s**2 * np.mean(h)**0  * l_s * D, fmt = 'none', ecolor =  '#990000')

