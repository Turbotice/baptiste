## 0 padding (dans une direction)

# n = 2^nextpow2(L);
def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return np.ceil(np.log2(number))

def Zeropadding1D(Z, A = 4, B = 2):
    [nt, ny]= Z.shape
    Nzero = int(A*pow(B,NextPowerOfTwo(ny)))
    Zeropadded_etay = np.zeros((nt,ny+Nzero))
    Zeropadded_etay[:,:ny] = Z
    return Zeropadded_etay


def Zeropadding2d(Z,A=2,B=2):
    [nx, ny]= Z.shape
    Nzx = int(A*pow(2,NextPowerOfTwo(nx)))
    Nzy = int(A*pow(2,NextPowerOfTwo(ny)))

    #rajouter une condition sur np.mod
    nx0 = int((Nzx-nx)/2)
    ny0 = int((Nzy-ny)/2)
    
    Zeropadded_eta = np.zeros((Nzx,Nzy))+0*1j
    Zeropadded_eta[nx0:nx0+nx,ny0:ny0+ny] = Z
    return Zeropadded_eta


## Filtre subpixellaire Methode dite de Eddi

def filtre_subpixellaire(indice, k, TF):
    Delta = (k[1]-k[0]) * ( np.abs(TF[indice+1])-np.abs(TF[indice-1])) / ( 2* (2*np.abs(TF[indice])-np.abs(TF[indice+1]) -np.abs(TF[indice-1]) ))
    k_detect = k[indice]+Delta
    return k_detect

## FAST FOURIER TRANSFORM

def fft_gld_v2022(s,fe, fx):
    [dy, dt] = s.shape
    t = np.arange(0,dt)/fe
    Y = np.arange(0,dy)*fx
    debut = 1
    fin = len(Y)
    x = Y;
    Nmin = 1
    Nmax = dt
    print('Number of frames: ' + str(Nmax))
    H = np.zeros((fin-debut+1,Nmax-Nmin+1)); #initialisation du champ spatiotemporel total
    for i in range(Nmin,Nmax):
        H[:,i] = s[:,i]*fx
        compteur = 0; 
        Max_TF   = 10**(12);
        indicesx = np.zeros(len(x))

        ## TF temporelle 
        TFt = np.zeros((len(x), Nmax)); ## On initialise le tableau de la Transformee de fourier temporelle
        for j in range(1,len(x)):
            TFp = np.abs(np.fft.fftshift(np.fft.fft(H[j,:])-np.mean(H[:,i]))); ## Transformee de fourier spatiale 
            compteur = compteur + 1;
            indicesx[compteur]=j;
            TFt[compteur,:]=TFp;
            TFt[compteur,:] = np.abs(np.fft.fftshift(np.fft.fft(H[j,:])))
    ## Definition de l'axe des temps
    n = len(TFt[1,:])+1;
    f = np.arange(-fe/2,fe/2, fe/(n-1))

    midf=int(len(f)/2);
    # filtrage du mode n=1
    #TFt[midf-1:midf+1,:]=0;
    # 
    plt.figure()
    plt.plot(np.abs(f),TFt[1,:]**2);
    plt.xlabel('f')
    plt.ylabel('$TFt^2$')
    ## TF spatiale
    compteur = 0; 
    indicest = np.zeros(Nmax)
    TFx = np.zeros((Nmax, len(x))); # On initialise la transformee de fourier spatiale
    for i in range(1, Nmax):
        TFp=np.abs(np.fft.fftshift(np.fft.fft(H[:,1]-np.mean(H[:,i]))));
        compteur=compteur+1;
        indicest[compteur]=i;
        TFx[compteur,:]=TFp;

    #  Definition de l'axe des k
    ke = 1./fx; # echantillonage en k
    nx = len(TFx[:,1])+1;
    k  = np.arange(-ke/2,ke/2,ke/(nx-1))*2*np.pi; # axe des k 
    midk=int(len(k)/2);
    # filtrage du mode n=1
    #TFx[:,midk-1:midk+1]=0;
    #  Moyenne temporelle de la transformee de fourier spatiale
    TF_x = np.mean(TFx, axis = 1);
    plt.figure()
    plt.loglog(np.abs(k),TF_x**2);
    plt.xlabel(r'k')
    plt.ylabel(r'$TFx^2$')
    
    TF=np.fft.fftshift(np.fft.fftshift(np.fft.fft2(H-np.mean(H)),0),1); # fft dans l'axe des temps puis spatial

    nx=len(TF[:,1])+1;
    k=np.arange(-ke/2,ke/2,ke/(nx-1))*2*np.pi; # axe des k
    nt=len(TF[1,:])+1;
    f =np.arange(-fe/2,fe/2,fe/(nt-1)); # axe des temps
    midk=int(len(k)/2);
    midf=int(len(f)/2);
    # filtrage du mode n=1
    #☺TF[:,midk-1:midk+1]=0;
    #TF[midf-1:midf+1,:]=0; 
    TFT = np.abs(TF)
    
    # plt.figure()
    # plt.pcolormesh(k,f,TFT, vmin =0, vmax = 1000)
    # plt.xlabel(r'$k (\rm cm^{-1})$')
    # plt.ylabel(r'$f (\rm s^{-1})$')
    # plt.xlim([-5, 5])
    # plt.ylim([-25, 25])
    # cbar = plt.colorbar()
    # cbar.set_label(r'$S\left(\eta(y,t)\right)$')
    return k,f,TFT, TFt, TF_x
