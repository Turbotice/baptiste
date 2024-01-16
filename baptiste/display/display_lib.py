import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np

import baptiste.files.save as sv
import baptiste.tools.tools as tools

## TAILLE DES FIGURES

def set_size(width = 15, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # if width == 'thesis':
    #     width_cm = 12
    # elif width == 'beamer':
    #     width_cm = 10
    # else:
    width_cm = width #en cm

    # Width of figure (in pts)
    
    cm_to_pt = 1/0.0351459804
    golden_ratio = .9
    fig_width_pt = int(width_cm * fraction * cm_to_pt)
    
    # Convert from pt to inches
    
    inches_per_pt = 1 / 72.27

    # # Golden ratio to set aesthetic figure height
    # # https://disq.us/p/2940ij3
    # #golden_ratio = (5**.5 - 1) / 2
    # golden_ratio = .9
    # # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    # #fig_height_in = fig_width_in * .9 * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


## FONTS

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize":10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}

plt.rcParams.update(tex_fonts)
plt.rcParams['text.usetex'] = True
mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = [
#     r'\usepackage{lmodern}', #lmodern: lateX font; tgheros: helvetica font; helvet pour helvetica
#     r'\usepackage{sansmath}', # math-font matching helvetica
#     r'\sansmath' # actually tell tex to use it!
#     r'\usepackage[scientific-notation=false]{siunitx}', # micro symbols
#     r'\sisetup{detect-all}', # force siunitx to use the fonts
#     ]

mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'  + r'\usepackage{sansmath}' + r'\sansmath' + r'\usepackage[scientific-notation=false]{siunitx}' + r'\sisetup{detect-all}'

# plt.rcParams["text.latex.preamble"].join([
#         r"\usepackage{dashbox}",              
#         r"\setmainfont{xcolor}",
# ])
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}' 
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}'
# mpl.rcParams['text.latex.preamble'] = r'\sansmath'
# mpl.rcParams['text.latex.preamble'] = r'\usepackage[scientific-notation=false]{siunitx}'
# mpl.rcParams['text.latex.preamble'] = r'\sisetup{detect-all}'


## CHOIX DES COULEURS
def vcolors(n) :
    vcolor = plt.cm.viridis(np.linspace(0,1,10))
    return vcolor[n]


    
def figurejolie(params = False, num_fig = False, subplot = False, nom_fig = False, width = 20):
    if subplot == False :
        
        #si params renseigné
        if type(params) != bool :
            #crée un nombre random qui sera le num de la figure, ce numero sera ajouté dans params[num_fig]
            if 'num_fig' not in params.keys() :
                params['num_fig'] = []
            if type(num_fig) != bool :
                params['num_fig'].append(num_fig)
            else :
                randnumfig = tools.datetimenow(date = False, time = False, micro_sec = True)
                params['num_fig'].append(randnumfig)
                num_fig = randnumfig
                
            plt.figure(num = num_fig, figsize = set_size(width = width, fraction = 1, subplots = (1,1)))
            
            #ajoute nom_fig dans les params
            if type(nom_fig) != bool :
                params[str(num_fig)] = {'nom_fig' : nom_fig}
            else :
                params[str(num_fig)] = {'nom_fig' : str(num_fig)}
                
            return params
        else :
            if type(num_fig) != bool :
                plt.figure(num = num_fig, figsize = set_size(fraction = 1, subplots = (1,1)))
            else :
                plt.figure(figsize = set_size(fraction = 1, subplots = (1,1)))
    else :
        fig = plt.figure(num = num_fig, figsize = set_size(fraction = 1, subplots = subplot))
        axes = []
                           
        return fig, axes


def joliplot(xlabel, ylabel, xdata, ydata, color = False, fig = False, axes = [], title = False, subplot = False, legend = False, 
             log = False, exp = True, image = False, zeros = False, params = False, table = False, tcbar = ''):
    """
    Plot un graph ou un subplot ou une image

    Parameters
    ----------
    xlabel : 
    ylabel : 
    xdata : 
    ydata : 
    color : optional, 1 - 16

    fig : optional, default is False.
    axes : optional, default is [].
    title : optional, default is False.
    subplot : optional, default is False.
    legend : optional, default is False.
    log : optional, default is False.
    exp : optional, default is True.
    image : optional, default is False.
    zeros : optional, default is False. Si on veut que le l'origine soit dans le graph

    Returns
    -------
    axes : TYPE
        DESCRIPTION.
    Si tableau ou image, mettre le contenue dans table ou image, et mettre les axes sur xdata et ydata.

    """
    
    """Jolis graphs predefinis"""
    
    
    n = 17
    markers = ['0','x','o','v','p','X','d','s','s','h','.','.','o','o','o','v','v','o']
    markeredgewidth = [1.5,1.8,1.5, 1.5, 2.5, 1.3, 1.3, 1.6,1.6,2,1.6,1.6,2,2,2,2,2,2.2]
    ms = [7,6.5,7, 7, 9.2, 8, 8, 8, 7,7, 7, 7,9,7,7,7,7,7]
    mfc = ['None','#91A052','#990000',vcolors(5),'None','None','None','None','k','None','None','None','None','None','None','None','None', vcolors(4)]
    colors = ['g','#91A052','#990000', vcolors(5), '#008B8B', vcolors(2), '#FF8000', vcolors(6), 'k',vcolors(1),'#01FA22',vcolors(3), vcolors(1),'#990000', vcolors(2),'#990000', vcolors(2), vcolors(4) ]
    
    """Pour un simple plot"""
    if subplot == False:
        
        #si c'est une image la mettre dans image
        if type(image) != bool :
            plt.imshow(image, cmap = plt.cm.gray)
            plt.xlabel(xlabel) 
            plt.ylabel(ylabel)
            if title != False:
                plt.title(title)
            plt.grid('off')
            plt.axis('off')
            plt.tight_layout()
            
            if params :
                return sv.data_to_dict([xlabel, ylabel], [xdata, ydata], image)
            
        #pour un tableau 2 ou 3D (3D affiche juste le t0) (contenu dans table)
        if type(table) != bool :
            dim = len(table.shape)
                
            if dim == 1 :
                
                print("tableau 1D à faire en plot pas table")
                
            if dim == 2 :
                plt.pcolormesh(xdata, ydata, np.flip(np.rot90(table),0), shading = 'auto')
                cbar = plt.colorbar()
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                cbar.set_label(tcbar)
                plt.grid('off')
                
            if dim == 3 :
                print('Carefull : 3D tables are ploted 2D')
                plt.pcolormesh(xdata, ydata, np.flip(np.rot90(table[:,:,0]),0), shading = 'auto')
                cbar = plt.colorbar()
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                cbar.set_label(tcbar)
                plt.grid('off')
                plt.axis('equal')
                
            if params :
                return sv.data_to_dict([xlabel, ylabel], [xdata, ydata], table)
        
        #pour un graph
        else :
            #si color = False fait une couleur au hasard
            if color == False:
                color = np.random.randint(1,n+1)
            
                
            if exp :
                marker = ' ' + markers[color]
            
            else :
                marker = '-'
                if color == 8 :
                    marker = '--'
                if color == 2 :
                    marker = '-.'
    
            
            
            if title != False:
                plt.title(title)
               
            if legend != False :
                plt.plot(xdata, ydata, marker, color = colors[color], mfc = mfc[color], markeredgewidth = markeredgewidth[color], ms = ms[color], label = legend)
                plt.legend()
            else :
                plt.plot(xdata, ydata, marker, color = colors[color], mfc = mfc[color], markeredgewidth = markeredgewidth[color], ms = ms[color])
                
            plt.xlabel(xlabel) 
            plt.ylabel(ylabel)
            if log :
                plt.yscale('log')
                plt.xscale('log')
                plt.tight_layout()
                
            plt.grid()
            
            if zeros :
               plt.xlim(left=0 )
               plt.ylim(bottom=0)
               
            if params :
                x_range = np.linspace(np.min(xdata), np.max(xdata), len(xdata))
                y_range = np.linspace(np.min(ydata), np.max(ydata), len(ydata))
                return sv.data_to_dict([xlabel, ylabel], [x_range, y_range], [xdata, ydata])
            


    #pour un subplot
    else:
        if image != False :
            axes.append( fig.add_subplot(subplot[0], subplot[1], len(axes) + 1) )
            axes[-1].imshow(image, cmap = plt.cm.gray)
            axes[-1].set_xlabel(xlabel)
            axes[-1].set_ylabel(ylabel)
            if title != False:
                axes[-1].set_title(title)
            axes[-1].grid('off')
            plt.tight_layout()
                
        else :
            if color == False:
                color = np.random.randint(1,n+1)
                    
            if exp :
                marker = ' ' + markers[color]
            if exp == False :
                marker = '-'
    
                
            axes.append( fig.add_subplot(subplot[0], subplot[1], len(axes) + 1) )
            axes[-1].plot(xdata, ydata, marker, color = colors[color], mfc = mfc[color], markeredgewidth = markeredgewidth[color], ms = ms[color], label = legend)
            axes[-1].set_xlabel(xlabel)
            axes[-1].set_ylabel(ylabel) 
                
            if title != False :
                axes[-1].set_title(title)
            if legend != False :
                axes[-1].legend()
            if log == True :
                axes[-1].set_yscale('log')
                axes[-1].set_xscale('log')
                plt.tight_layout()
            
        return axes
            

def set_axe_pi (nb_ticks, x, axxe = 'x') :
    if axxe == 'x'or axxe == 'xy' :
        axes = plt.gca()
        
        axes.xaxis.set_ticks([u * np.max(x) / (nb_ticks - 1) for u in range (0, nb_ticks)])
        
        axes.xaxis.set_ticklabels([r'$0$', r'$\pi$'] + [ str(u) + r'$\pi$' for u in range(2, nb_ticks)])
    if axxe == 'y' or axxe == 'xy' :
        axes = plt.gca()

        axes.yaxis.set_ticks([u * np.max(x) / (nb_ticks - 1) for u in range (0, nb_ticks)])

        axes.yaxis.set_ticklabels([r'$0$', r'$-\pi$'] + [ str(u) + r'$\pi$' for u in range(2, nb_ticks)] )

