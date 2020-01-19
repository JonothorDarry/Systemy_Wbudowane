import math
import numpy as np
cimport numpy as np
from cpython cimport array
import matplotlib.pyplot as plt
import array

Hyperparameters={'Vertikal':(6,3,4,1), 'Upgrade':70, 'Blackness':10, 'Depth':3, 'Lineslayer':0.25, 'Showpaths':0, 'Change':4}

def petrify(path, axe):
    for i in range(len(path)):
        axe.scatter(path[i][1], path[i][0], color='red')

def blackening(np.ndarray[unsigned char, ndim=2] bwimg, np.ndarray[long, ndim=2] path):
    cdef int s=0, t=0, gr=0, ij, i, jj
    cdef np.ndarray[int, ndim=1] dep=np.zeros((len(path)), dtype='int32')
    cdef np.ndarray[int, ndim=1] end1=np.zeros((len(path)), dtype='int32')
    cdef np.ndarray[int, ndim=1] end2=np.zeros((len(path)), dtype='int32')
    
    for i in range(0, len(path)):
        for ij in range(0, -10, -1):
            a, b=path[i,0]+ij, path[i,1]
            if (bwimg[a,b]==255):
                break
        end1[i]=ij
        
        for ij in range(1, 10, 1):
            a, b=path[i,0]+ij, path[i,1]
            
            if (bwimg[a,b]==255):
                break
        end2[i]=ij
        
        dep[i]=end2[i]-end1[i]
    
    #dep - jak długie jest pasmo w pionie, grub - grubość
    ld=len(dep)
    dep2=sorted(dep)
    small=dep2[math.floor(0.15*ld)]
    large=dep2[math.floor(0.85*ld)]
    
    dep2=dep2[math.floor(0.1*ld):math.floor(0.9*ld)]
    grub=sum(dep2)/len(dep2)-2
    #print('DEPTH {} {}'.format(grub, dep))
    
    #Midian - środkowy punkt pasma w punkcie
    midian=[0]*len(path)
    cur=0
    last=0
    midian[0]=path[0,0]
    for i in range(1, len(path)):
        if (dep[i]>grub+3):
            cur+=1
            continue
        else:
            midian[i]=path[i,0]+(end2[i]+end1[i])//2
            while (cur>0):
                p=path[i,0]-cur
                midian[i-cur]=midian[i] if (cur<p-last) else midian[last]
                tv=midian[i-cur]
                jj=1
                while (bwimg[tv, path[i-cur,1]]==255):
                    tv=tv+jj
                    jj=-jj-1*jj//abs(jj) 
                midian[i-cur]=tv
                cur-=1
            last=i
            
    #print(list(zip(path, midian)))
    dt0=math.ceil(grub)
    for i in range(len(path)):
        sc, t1, t2=0, 0, 0
        for jj in range(1, 10):
            if (bwimg[midian[i]+jj, path[i,1]]==255):
                break
        t1=jj
        for jj in range(0, -10, -1):
            if (bwimg[midian[i]+jj, path[i,1]]==255):
                break
        t2=jj
        sc=t1-t2
        
        if (sc<=dt0+2):
            for jj in range(t2, t1+1, 1):
                bwimg[midian[i]+jj, path[i,1]]=255
    c1=math.floor(0.1*len(midian))
    c2=math.floor(0.9*len(midian))
    
    return (sum(midian[c1:c2])/len(midian[c1:c2]), grub)


def cleanblack(bwimg, pathway):
    for x in pathway:
        bwimg[x[0]-5:x[0]+5,x[1]]=255
    return (1,2)

#limit - o ile y może się odchylić od y2
def pathfinder(np.ndarray[unsigned char, ndim=2] bwimg, F, np.ndarray[unsigned char, ndim=2] check, np.ndarray[int, ndim=1] par, np.ndarray[int, ndim=1] w, int y, int x, int p, int vv, int lowx, int highx, int y2, np.ndarray[int, ndim=1] miss, np.ndarray[int, ndim=1] added, np.ndarray[int, ndim=1] Vertikal, int myconst, int Vertikalpenalty, np.ndarray[int, ndim=1] bestl, np.ndarray[int, ndim=1] bestr, np.ndarray[int, ndim=1] slaycount):
    cdef int limit=10000, l2=Hyperparameters['Blackness']
    
    if (y>=bwimg.shape[0] or x>=bwimg.shape[1] or x<0 or y<0 or abs(y-y2)>limit or check[y,x]==1):
        return (p, lowx, highx)
    if ((bwimg[y,x]==255 or (x>=lowx and x<=highx)) and added[vv]+1>myconst):
        return (p, lowx, highx)
    if (y>F[vv][0] or y<F[vv][0]):
        if ((Vertikal[vv]+Hyperparameters['Vertikal'][1]>Vertikalpenalty) or (bwimg[y,x]==255 and Vertikal[vv]+Hyperparameters['Vertikal'][2]>Vertikalpenalty)):
            return (p, lowx, highx)
    if ((bwimg[y,x]==255 or (x>=bestl[vv] and x<=bestr[vv])) and slaycount[vv]+1>Hyperparameters['Change']):
        return (p, lowx, highx)
    
    if (bwimg[y,x]==0 or miss[vv]<l2):
        if (bwimg[y,x]==255):
            miss[p]=miss[vv]+1
        else:
            miss[p]=0
            #miss[p]=max(miss[vv]-2, 0)
        if (y>F[vv][0] or y<F[vv][0]):
            if (bwimg[y,x]==255):
                Vertikal[p]=Vertikal[vv]+Hyperparameters['Vertikal'][2]
            else:
                Vertikal[p]=Vertikal[vv]+Hyperparameters['Vertikal'][1]
        else:
            Vertikal[p]=max(Vertikal[vv]-Hyperparameters['Vertikal'][3], 0)
        F.append((y, x))
        check[y,x]=1
        par[p]=vv
        
        #if (bwimg[y,x]==0 and x<bestl[vv]):
        if (x<bestl[vv]):
            slaycount[p]=0
            bestl[p]=x
            bestr[p]=bestr[vv]
        #elif (x>bestr[vv]):
        elif (x>bestr[vv]):
            slaycount[p]=0
            bestr[p]=x
            bestl[p]=bestl[vv]
        else:
            slaycount[p]+=1
            bestl[p]=bestl[vv]
            bestr[p]=bestr[vv]
        
        if (bwimg[y,x]==0 and x<lowx):
            lowx=x
            added[p]=0
            w[p]=-1
        elif (bwimg[y,x]==0 and x>highx):
            highx=x
            w[p]=1
            added[p]=0
        else:
            w[p]=0
            added[p]=added[vv]+1
        return (p+1, lowx, highx)
    return (p, lowx, highx)


def findlinez(np.ndarray[unsigned char, ndim=2] bwimg, shp):
    cdef int skv=2, y=1, kk=1, iF=0, jF=1, deadl=0, deadr=1, highx, lowx, x1, x2, C=140000, myconst=Hyperparameters['Upgrade']
    cdef int Vertikalpenalty=Hyperparameters['Vertikal'][0]
    solution=[]
    cdef np.ndarray[unsigned char, ndim=2] check=np.zeros((bwimg.shape[0], bwimg.shape[1]), dtype='uint8')
    cdef np.ndarray[int, ndim=1] par=np.zeros((C), dtype='int32')
    cdef np.ndarray[int, ndim=1] miss=np.zeros((C), dtype='int32')
    cdef np.ndarray[int, ndim=1] w=np.zeros((C), dtype='int32')
    cdef np.ndarray[int, ndim=1] added=np.zeros((C), dtype='int32')
    cdef np.ndarray[int, ndim=1] Vertikal=np.zeros((C), dtype='int32')
    
    cdef np.ndarray[int, ndim=1] bestl=np.zeros((C), dtype='int32')
    cdef np.ndarray[int, ndim=1] bestr=np.zeros((C), dtype='int32')
    cdef np.ndarray[int, ndim=1] slaycount=np.zeros((C), dtype='int32')
    
    for kk in range(1, skv):
        y=1
        x1=(kk*bwimg.shape[1])//skv-2
        x2=(kk*bwimg.shape[1])//skv+2
        while (y<bwimg.shape[0]-1):
            vs=bwimg[y,x1:x2+1]
            if (0 in vs):
                xl=kk*(bwimg.shape[1]//skv)
                yl=y
                
                pathway=[]
                F=[(yl, xl)]
                par[0]=-1
                w[0]=0
                miss[0]=0
                Vertikal[0]=0
                bestl[0]=xl
                bestr[0]=xl
                slaycount[0]=0
                
                highx, lowx=xl, xl
                iF, jF=0, 1
                while(iF<jF):
                    s=F[iF]
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0]-1, s[1], jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, Vertikalpenalty, bestl, bestr, slaycount)
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0]+1, s[1], jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, Vertikalpenalty, bestl, bestr, slaycount)
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0], s[1]-1, jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, Vertikalpenalty, bestl, bestr, slaycount)
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0], s[1]+1, jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, Vertikalpenalty, bestl, bestr, slaycount)
                    iF+=1
                #print(y, jF, lowx, highx)
                #print('PedE', pointdead, jF, highx, lowx)
                for j1 in range(jF-1, -1, -1):
                    if (w[j1]==-1 and bwimg[F[j1][0], F[j1][1]]==0):
                        break
                while (j1>=0):
                    pathway.append(F[j1])
                    j1=par[j1]
                
                for j2 in range(jF-1, -1, -1):
                    if (w[j2]==1 and bwimg[F[j2][0], F[j2][1]]==0):
                        break
                
                p2=[]
                while (j2>0):
                    p2.append(F[j2])
                    j2=par[j2]
                
                pathway.extend(p2[::-1])
                for x in F:
                    check[x[0], x[1]]=0
                if (highx-lowx>shp[1]*Hyperparameters['Lineslayer']):
                    sv, gr=blackening(bwimg, np.asarray(pathway))
                    #sv, gr=cleanblack(bwimg, pathway)
                    if (Hyperparameters['Showpaths']==1):
                        fig, ax=plt.subplots(1,1,figsize=(18,18))
                        petrify(pathway, ax)
                        ax.imshow(bwimg, cmap='Greys',  interpolation='nearest')
                        plt.show()
                    
                    solution.append((sv, gr, lowx, highx, pathway))
            y+=1
    return solution
