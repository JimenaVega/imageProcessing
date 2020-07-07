# coding: utf-8

#ejercicio 6 PROCOM 2019 
# In[14]:

import numpy as np
import matplotlib.pyplot as plt
from tool._fixedInt import *



## Parametros generales
T     = 1.0/1.0e9 # Periodo de baudio
Nsymb = 1000          # Numero de simbolos
os    = 8
## Parametros de la respuesta en frecuencia
Nfreqs = 256          # Cantidad de frecuencias

## Parametros del filtro de caida cosenoidal
beta   = [0.0,0.5,1.0] # Roll-Off
Nbauds = 16.0          # Cantidad de baudios del filtro
## Parametros funcionales
Ts = T/os              # Frecuencia de muestreo


 # In[15]:
 
def rcosine(beta, Tbaud, oversampling, Nbauds, Norm):
     """ Respuesta al impulso del pulso de caida cosenoidal """
     t_vect = np.arange(-0.5*Nbauds*Tbaud, 0.5*Nbauds*Tbaud, 
                        float(Tbaud)/oversampling) 
     
 
     y_vect = []
     for t in t_vect:
         y_vect.append(np.sinc(t/Tbaud)*(np.cos(np.pi*beta*t/Tbaud)/
                                         (1-(4.0*beta*beta*t*t/
                                             (Tbaud*Tbaud)))))
 
     y_vect = np.array(y_vect)
 
     if(Norm):
         return (t_vect, y_vect/np.sqrt(np.sum(y_vect**2)))
         #return (t_vect, y_vect/y_vect.sum())
     else:
         return (t_vect,y_vect)
# =============================================================================

def rta_impulso(t0,t1,t2,r0,r1,r2,rc0,rc1,rc2,beta,time):
    
    plt.figure()
    plt.suptitle('beta = ' + beta, fontsize=14, fontweight='bold')
    
    #S(8,7)
    plt.subplot(3,1,1)
    plt.plot(time,t0, 'ro-', linewidth = 2.0) #truncado rojo
    plt.plot(time,r0, 'bs-', linewidth = 2.0) #redondeo azul
    plt.plot(time,rc0,'k^-', linewidth = 2.0) #original
    plt.title('S(8,7)')    
    plt.xlabel('Muestras')
    plt.ylabel('Magnitud')
    
    #S(3,2)
    plt.subplot(3,1,2)
    plt.plot(time,t1, 'ro-', linewidth = 2.0)
    plt.plot(time,r1, 'bs-', linewidth = 2.0)
    plt.plot(time,rc1,'k^-', linewidth = 2.0)
    plt.title('S(3,2)')    
    plt.xlabel('Muestras')
    plt.ylabel('Magnitud')
    
    #S(6,4)
    plt.subplot(3,1,3)
    plt.plot(time,t2, 'ro-', linewidth = 2.0)
    plt.plot(time,r2, 'bs-', linewidth = 2.0)
    plt.plot(time,rc2,'k^-', linewidth = 2.0)
    plt.title('S(6,4)')    
    plt.xlabel('Muestras')
    plt.ylabel('Magnitud')
    
    
    
# In[16]:

### Calculo de tres pusos con diferente roll-off (beta)
(t,rc0) = rcosine(beta[0], T,os,Nbauds,Norm=False)
(t,rc1) = rcosine(beta[1], T,os,Nbauds,Norm=False)
(t,rc2) = rcosine(beta[2], T,os,Nbauds,Norm=False)

### Filtros cuantizados

rc087t = arrayFixedInt(8, 7, rc0, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc087r = arrayFixedInt(8, 7, rc0, signedMode='S', roundMode='round', saturateMode='saturate')
rc032t = arrayFixedInt(3, 2, rc0, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc032r = arrayFixedInt(3, 2, rc0, signedMode='S', roundMode='round', saturateMode='saturate')
rc064t = arrayFixedInt(6, 4, rc0, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc064r = arrayFixedInt(6, 4, rc0, signedMode='S', roundMode='round', saturateMode='saturate')

rc187t = arrayFixedInt(8, 7, rc1, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc187r = arrayFixedInt(8, 7, rc1, signedMode='S', roundMode='round', saturateMode='saturate')
rc132t = arrayFixedInt(3, 2, rc1, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc132r = arrayFixedInt(3, 2, rc1, signedMode='S', roundMode='round', saturateMode='saturate')
rc164t = arrayFixedInt(6, 4, rc1, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc164r = arrayFixedInt(6, 4, rc1, signedMode='S', roundMode='round', saturateMode='saturate')

rc287t = arrayFixedInt(8, 7, rc2, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc287r = arrayFixedInt(8, 7, rc2, signedMode='S', roundMode='round', saturateMode='saturate')
rc232t = arrayFixedInt(3, 2, rc2, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc232r = arrayFixedInt(3, 2, rc2, signedMode='S', roundMode='round', saturateMode='saturate')
rc264t = arrayFixedInt(6, 4, rc2, signedMode='S', roundMode='trunc', saturateMode='saturate')
rc264r = arrayFixedInt(6, 4, rc2, signedMode='S', roundMode='round', saturateMode='saturate')



plt.figure()
plt.plot(t,rc0,'ro-',linewidth=2.0,label=r'$\beta=0.0$')
plt.plot(t,rc1,'gs-',linewidth=2.0,label=r'$\beta=0.5$')
plt.plot(t,rc2,'k^-',linewidth=2.0,label=r'$\beta=1.0$')
plt.legend()
plt.grid(True)
plt.xlabel('Muestras')
plt.ylabel('Magnitud')

#lista con cada array de objetos
all_f = [rc087t,rc087r,rc032t,rc032r,rc064t,rc064r, #beta = 0.0
         rc187t,rc187r,rc132t,rc132r,rc164t,rc164r, #beta = 0.5
         rc287t,rc287r,rc232t,rc232r,rc264t,rc264r] #beta = 1.0

quant=[] #guarda all_f pero convertidos de objetos -> float

for i in range(len(all_f)):
    vect=[]
    for j in range(len(all_f[i])):
       
        vect.append(all_f[i][j].fValue)
        
        
    quant.append(vect)

rta_impulso(quant[0] , quant[2], quant[4] , quant[1] , quant[3] , quant[5] ,rc0,rc1,rc2,'0.0', t)
rta_impulso(quant[6] , quant[8], quant[10], quant[7] , quant[9] , quant[11],rc0,rc1,rc2,'0.5', t)
rta_impulso(quant[12],quant[14], quant[16], quant[13], quant[15], quant[17],rc0,rc1,rc2,'1.0', t)


## In[17]:
#
#def resp_freq(filt, Ts, Nfreqs):
#    
#    H = [] # Lista de salida de la magnitud
#    A = [] # Lista de salida de la fase
#    filt_len = len(filt)
#
#    #### Genero el vector de frecuencias
#    freqs = np.matrix(np.linspace(0,1.0/(2.0*Ts),Nfreqs))
#    #### Calculo cuantas muestras necesito para 20 ciclo de
#    #### la mas baja frec diferente de cero
#    Lseq = 20.0/(freqs[0,1]*Ts)
#
#    #### Genero el vector tiempo
#    t = np.matrix(np.arange(0,Lseq))*Ts
#
#    #### Genero la matriz de 2pifTn
#    Omega = 2.0j*np.pi*(t.transpose()*freqs)
#
#    #### Valuacion de la exponencial compleja en todo el
#    #### rango de frecuencias
#    fin = np.exp(Omega)
#
#    #### Suma de convolucion con cada una de las exponenciales complejas
#    #se plotea en un semilog
#    for i in range(0,np.size(fin,1)):
#        fout = np.convolve(np.squeeze(np.array(fin[:,i].transpose())),filt)
#        mfout = abs(fout[filt_len:len(fout)-filt_len])
#        afout = np.angle(fout[filt_len:len(fout)-filt_len])
#        H.append(mfout.sum()/len(mfout))
#        A.append(afout.sum()/len(afout))
#
#    return [H,A,list(np.squeeze(np.array(freqs)))]
#def graf_freq(F0,F1,F2,H0,H1,H2):
#    plt.figure()
#    plt.semilogx(F0, 20*np.log10(H0),'r', linewidth=2.0, label=r'$\beta=0.0$')
#    plt.semilogx(F1, 20*np.log10(H1),'g', linewidth=2.0, label=r'$\beta=0.5$')
#    plt.semilogx(F2, 20*np.log10(H2),'k', linewidth=2.0, label=r'$\beta=1.0$')
#    
#    plt.axvline(x=(1./Ts)/2.      , color='k' , linewidth=2.0)
#    plt.axvline(x=(1./T)/2.       , color='k' , linewidth=2.0)
#    plt.axhline(y=20*np.log10(0.5), color='k' , linewidth=2.0)
#    plt.legend(loc=3)
#    plt.grid(True)
#    plt.xlim(F2[1],F2[len(F2)-1])
#    plt.xlabel('Frequencia [Hz]')
#    plt.ylabel('Magnitud [dB]')
#
## In[18]:
#
#### Calculo respuesta en frec para los tres pulsos
#[H0,A0,F0] = resp_freq(rc0, Ts, Nfreqs)
#[H1,A1,F1] = resp_freq(rc1, Ts, Nfreqs)
#[H2,A2,F2] = resp_freq(rc2, Ts, Nfreqs)
#
#plt.figure()
#plt.suptitle('Respuesta en freq sin cuantizacion ', fontsize=14, fontweight='bold')
#plt.semilogx(F0, 20*np.log10(H0),'r', linewidth=2.0, label=r'$\beta=0.0$')
#plt.semilogx(F1, 20*np.log10(H1),'g', linewidth=2.0, label=r'$\beta=0.5$')
#plt.semilogx(F2, 20*np.log10(H2),'k', linewidth=2.0, label=r'$\beta=1.0$')
#
#plt.axvline(x=(1./Ts)/2.      , color='k' , linewidth=2.0)
#plt.axvline(x=(1./T)/2.       , color='k' , linewidth=2.0)
#plt.axhline(y=20*np.log10(0.5), color='k' , linewidth=2.0)
#plt.legend(loc=3)
#plt.grid(True)
#plt.xlim(F2[1],F2[len(F2)-1])
#plt.xlabel('Frequencia [Hz]')
#plt.ylabel('Magnitud [dB]')
#
####Rta en frecuencia de filtros cuantizados
#for i in range(len(quant)/3):
#    [H0,A0,F0] = resp_freq(quant[i],Ts,Nfreqs)
#    [H1,A1,F1] = resp_freq(quant[i+1],Ts,Nfreqs)
#    [H2,A2,F2] = resp_freq(quant[i+2],Ts,Nfreqs)
#    i=i*3
#    graf_freq(F0,F1,F2,H0,H1,H2)
#
##plt.show()

# In[19]:

symbolsI = 2*(np.random.uniform(-1,1,Nsymb)>0.0)-1;
symbolsQ = 2*(np.random.uniform(-1,1,Nsymb)>0.0)-1;
#bits
zsymbI = np.zeros(os*Nsymb);    zsymbI[1:len(zsymbI):int(os)]=symbolsI  
zsymbQ = np.zeros(os*Nsymb);    zsymbQ[1:len(zsymbQ):int(os)]=symbolsQ 


plt.figure()
plt.subplot(2,1,1)
plt.plot(zsymbI,'o', label = 'ZsymbI')
plt.xlim(0,20)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(zsymbQ,'o', label = 'ZsymbQ')
plt.xlim(0,20)
plt.grid(True)

# In[20]:

def graf_conv(conv_quanti0, conv_quanti1, conv_quanti2, conv_quanti3, conv_quanti4, conv_quanti5, title):
    plt.figure()
    plt.suptitle(title , fontsize=14, fontweight='bold')
    plt.subplot(2,1,1)
    plt.plot(conv_quanti0,'r-',linewidth=2.0,label=r'$\beta=%2.2f$'%beta[0])
    plt.plot(conv_quanti1,'g-',linewidth=2.0,label=r'$\beta=%2.2f$'%beta[1])
    plt.plot(conv_quanti2,'k-',linewidth=2.0,label=r'$\beta=%2.2f$'%beta[2])
    plt.xlim(1000,1250)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Muestras')
    plt.ylabel('Magnitud')
    
    plt.subplot(2,1,2)
    plt.plot(conv_quanti0,'r-',linewidth=2.0,label=r'$\beta=%2.2f$'%beta[0])
    plt.plot(conv_quanti1,'g-',linewidth=2.0,label=r'$\beta=%2.2f$'%beta[1])
    plt.plot(conv_quanti2,'k-',linewidth=2.0,label=r'$\beta=%2.2f$'%beta[2])
    plt.xlim(1000,1250)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Muestras')
    plt.ylabel('Magnitud')

def switch_title(i):
    if(i == 0): return 'S(8,7)t'
    if(i == 1): return 'S(8,7)r'
    if(i == 2): return 'S(3,2)t'
    if(i == 3): return 'S(3,2)r'
    if(i == 4): return 'S(6,4)t'
    if(i == 5): return 'S(6,4)r'

#CONVOLUSION DE SIMBOLOS CON LOS FILTROS cosenos realzados con distintos roll offs
symb_out0I = np.convolve(rc0,zsymbI,'same'); symb_out0Q = np.convolve(rc0,zsymbQ,'same')
symb_out1I = np.convolve(rc1,zsymbI,'same'); symb_out1Q = np.convolve(rc1,zsymbQ,'same')
symb_out2I = np.convolve(rc2,zsymbI,'same'); symb_out2Q = np.convolve(rc2,zsymbQ,'same')

title = 'Convolusiones sin cuantizacion '

graf_conv(symb_out0I, symb_out1I, symb_out2I, symb_out0Q, symb_out1Q, symb_out2Q, title )

#guarda las convolusiones de los primeros simbolos con cada tipo de filtro

conv_quanti0 = [] #vector con filtros de beta = 0.0
conv_quanti1 = [] #vector con filtros de beta = 0.5
conv_quanti2 = [] #vector con filtros de beta = 1.0

conv_quanti3 = []
conv_quanti4 = []
conv_quanti5 = []


for e in range(6):
    conv_quanti0.append(np.convolve(quant[e]   , zsymbI, 'same'))
    conv_quanti1.append(np.convolve(quant[e+6] , zsymbI, 'same'))
    conv_quanti2.append(np.convolve(quant[e+12], zsymbI, 'same'))
    
    conv_quanti3.append(np.convolve(quant[e]   , zsymbQ, 'same'))
    conv_quanti4.append(np.convolve(quant[e+6] , zsymbQ, 'same'))
    conv_quanti5.append(np.convolve(quant[e+12], zsymbQ, 'same'))
    
# Se imprimen lo plots
for i in range(6):
    title = switch_title(i)
    graf_conv(conv_quanti0[i],conv_quanti1[i],conv_quanti2[i],
              conv_quanti3[i],conv_quanti4[i],conv_quanti5[i],title) 

 # In[25]:
 
def constellation(sI,sQ,offset,color,tit,i,b):
   
    
    plt.subplot(2,3,i , title = tit)
    plt.suptitle(b , fontsize=14, fontweight='bold')
    plt.plot(sI[100+offset:len(sI)-(100-offset):int(os)],
             sQ[100+offset:len(sQ)-(100-offset):int(os)],
                 color,linewidth=2.0)
    plt.axis('equal')
    plt.axis([-2, 2, -2, 2])
    plt.grid(True)
    plt.xlabel('Real')
    plt.ylabel('Imag')
    
 
# In[26]:
    
offset = 6
#-----------------------------beta = 0.0--------------------------------------
plt.figure()
plt.plot(symb_out0I[100+offset:len(symb_out0I)-(100-offset):int(os)],
         symb_out0Q[100+offset:len(symb_out0Q)-(100-offset):int(os)],
             'k.',linewidth=2.0)
plt.axis('equal')
plt.axis([-2, 2, -2, 2])
plt.grid(True)
plt.xlabel('Real')
plt.ylabel('Imag')
#---------------------------------beta = 0.5----------------------------------
plt.figure()
plt.plot(symb_out1I[100+offset:len(symb_out1I)-(100-offset):int(os)],
         symb_out1Q[100+offset:len(symb_out1Q)-(100-offset):int(os)],
             'r.',linewidth=2.0)
plt.axis('equal')
plt.axis([-2, 2, -2, 2])
plt.grid(True)
plt.xlabel('Real')
plt.ylabel('Imag')
#---------------------------------beta = 1.0 ----------------------------------
plt.figure()
plt.plot(symb_out2I[100+offset:len(symb_out2I)-(100-offset):int(os)],
         symb_out2Q[100+offset:len(symb_out2Q)-(100-offset):int(os)],
             'b.',linewidth=2.0)
plt.axis('equal')
plt.axis([-2, 2, -2, 2])
plt.grid(True)
plt.xlabel('Real')
#----------------------------filtros cuntizados--------------------------------

plt.ylabel('Imag')

plt.figure()
for i in range(6):
    b = '0.0'
    title = switch_title(i) 
    constellation(conv_quanti0[i],conv_quanti3[i],offset,'k.',title,(i+1),b)
plt.figure()
for i in range(6):
    b = '0.5'
    title = switch_title(i)
    constellation(conv_quanti1[i],conv_quanti4[i],offset,'r.',title,(i+1),b)
plt.figure()
for i in range(6):
    b= '1.0'
    title = switch_title(i)
    constellation(conv_quanti2[i],conv_quanti5[i],offset,'b.',title,(i+1),b)
    
    
## In[]:
#def eyediagram(symb, n, offset, period, title):
#    data = symb[100:len(symb)-100]
#    span     = 2*n
#    segments = int(len(data)/span)
#    xmax     = (n-1)*period
#    xmin     = -(n-1)*period
#    x        = list(np.arange(-n,n,)*period)
#    xoff     = offset
#
#    plt.figure()
#    plt.suptitle(title , fontsize=14, fontweight='bold')
#    for i in range(0,segments-1):
#        plt.plot(x, data[(i*span+xoff):((i+1)*span+xoff)],'b')
#        plt.hold(True)
#        plt.grid(True)
#
#    plt.xlim(xmin, xmax)
#title = 'sin cuantizacion' 
#eyediagram(symb_out0I,os,6,Nbauds,title)
#eyediagram(symb_out0Q,os,6,Nbauds,title)
#eyediagram(symb_out1I,os,6,Nbauds,title)
#eyediagram(symb_out1Q,os,6,Nbauds,title)
#eyediagram(symb_out2I,os,6,Nbauds,title)
#eyediagram(symb_out2Q,os,6,Nbauds,title)
#
#for i in range(6):
#    title = switch_title(i)
#    eyediagram(conv_quanti0[i],os,6,Nbauds,title)
#    eyediagram(conv_quanti1[i],os,6,Nbauds,title)
#    eyediagram(conv_quanti2[i],os,6,Nbauds,title)
#    eyediagram(conv_quanti3[i],os,6,Nbauds,title)
#    eyediagram(conv_quanti4[i],os,6,Nbauds,title)
#    eyediagram(conv_quanti5[i],os,6,Nbauds,title)


plt.show(block=False)
raw_input('Press Enter to Continue')
plt.close()