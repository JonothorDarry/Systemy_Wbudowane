import math
import numpy as np
cimport numpy as np
from cpython cimport array
cimport cython

Hyperparameters={'Vertikal':(6,3,4,1), 'Upgrade':70, 'Blackness':10, 'Depth':3, 'Lineslayer':0.25, 'Showpaths':0, 'Change':4}
cdef int HPARVert0=Hyperparameters['Vertikal'][0]
cdef int HPARVert1=Hyperparameters['Vertikal'][1]
cdef int HPARVert2=Hyperparameters['Vertikal'][2]
cdef int HPARVert3=Hyperparameters['Vertikal'][3]
cdef int HPARUpgr=Hyperparameters['Upgrade']
cdef int HPARBlac=Hyperparameters['Blackness']
cdef int HPARDept=Hyperparameters['Depth']
cdef double HPARLine=Hyperparameters['Lineslayer']
cdef int HPARChan=Hyperparameters['Change']

cdef blackening(unsigned char[:,:] bwimg, int[:,:] path, int lenpath):
    cdef int s=0, t=0, gr=0, ij, i, jj
    cdef np.ndarray[int, ndim=1] dep=np.zeros((lenpath), dtype='int32')
    cdef np.ndarray[int, ndim=1] end1=np.zeros((lenpath), dtype='int32')
    cdef np.ndarray[int, ndim=1] end2=np.zeros((lenpath), dtype='int32')
    cdef np.ndarray[int, ndim=1] midian=np.zeros((lenpath), dtype='int32')
    
    for i in range(0, lenpath):
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
    
    cur=0
    last=0
    midian[0]=path[0,0]
    for i in range(1, lenpath):
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
    for i in range(lenpath):
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (int,int,int) pathfinder(unsigned char[:,:] bwimg, int[:,:] F,
                unsigned char[:,:] check, int[:] par,
                int[:] w, int y, int x, int p, int vv, int lowx, int highx, int y2,
                int[:] miss, int[:] added, int[:] Vertikal,
                int myconst, int[:] bestl, int[:] bestr,
                int[:] slaycount, int vv1, int vv2):
    cdef int l2=HPARBlac
    
    if(x<0 or y<0 or y>=vv1 or x>=vv2 or check[y,x]==1):
        return (p, lowx, highx)
    
    if ((y>F[vv,0] or y<F[vv,0]) and
            ((Vertikal[vv]+HPARVert1>HPARVert0) or (bwimg[y,x]==255 and Vertikal[vv]+HPARVert2>HPARVert0))):
            return (p, lowx, highx)
    
    if (added[vv]+1>myconst and (bwimg[y,x]==255 or (x>=lowx and x<=highx))):
        return (p, lowx, highx)
    
    if (slaycount[vv]+1>HPARChan and (bwimg[y,x]==255 or (x>=bestl[vv] and x<=bestr[vv]))):
        return (p, lowx, highx)
    
    if (bwimg[y,x]==0 or miss[vv]<l2):
        if (bwimg[y,x]==255):
            miss[p]=miss[vv]+1
        else:
            miss[p]=0
        if (y>F[vv,0] or y<F[vv,0]):
            if (bwimg[y,x]==255):
                Vertikal[p]=Vertikal[vv]+HPARVert2
            else:
                Vertikal[p]=Vertikal[vv]+HPARVert1
        else:
            Vertikal[p]=max(Vertikal[vv]-HPARVert3, 0)
        F[p,0]=y
        F[p,1]=x
        
        check[y,x]=1
        par[p]=vv
        
        if (x<bestl[vv]):
            slaycount[p]=0
            bestl[p]=x
            bestr[p]=bestr[vv]
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


def findlinez(unsigned char[:,:] bwimg, shp):
    cdef int skv=2, y=1, kk=1, iF=0, jF=1, deadl=0, deadr=1, highx, lowx, x1, x2, C=140000, myconst=Hyperparameters['Upgrade']
    cdef int lenpath=0, lp2=0, D=3000
    solution=[]
    cdef unsigned char[:,:] check=np.zeros((bwimg.shape[0], bwimg.shape[1]), dtype='uint8')
    cdef int[:] par=np.zeros((C), dtype='int32')
    cdef int[:] miss=np.zeros((C), dtype='int32')
    cdef int[:] w=np.zeros((C), dtype='int32')
    cdef int[:] added=np.zeros((C), dtype='int32')
    cdef int[:] Vertikal=np.zeros((C), dtype='int32')
    
    cdef int[:] bestl=np.zeros((C), dtype='int32')
    cdef int[:] bestr=np.zeros((C), dtype='int32')
    cdef int[:] slaycount=np.zeros((C), dtype='int32')
    cdef int[:,:] F=np.zeros((C, 2), dtype='int32')
    cdef int[:,:] pathway=np.zeros((D, 2), dtype='int32')
    cdef int[:,:] p2=np.zeros((D, 2), dtype='int32')
    
    cdef int vv1=bwimg.shape[0], vv2=bwimg.shape[1]
    
    
    for kk in range(1, skv):
        y=1
        x1=(kk*bwimg.shape[1])//skv-2
        x2=(kk*bwimg.shape[1])//skv+2
        while (y<bwimg.shape[0]-1):
            vs=bwimg[y,x1:x2+1]
            if (0 in vs):
                xl=kk*(bwimg.shape[1]//skv)
                yl=y
                
                lenpath=0
                F[0,0]=yl
                F[0,1]=xl
                
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
                    s=[F[iF,0],F[iF,1]]
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0]-1, s[1], jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, bestl, bestr, slaycount, vv1, vv2)
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0]+1, s[1], jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, bestl, bestr, slaycount, vv1, vv2)
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0], s[1]-1, jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, bestl, bestr, slaycount, vv1, vv2)
                    jF, lowx, highx=pathfinder(bwimg, F, check, par, w, s[0], s[1]+1, jF, iF, lowx, highx, y, miss, added, Vertikal, myconst, bestl, bestr, slaycount, vv1, vv2)
                    iF+=1
                
                for j1 in range(jF-1, -1, -1):
                    if (w[j1]==-1 and bwimg[F[j1,0], F[j1,1]]==0):
                        break
                        
                while (j1>=0):
                    pathway[lenpath,0]=F[j1,0]
                    pathway[lenpath,1]=F[j1,1]
                    lenpath+=1
                    j1=par[j1]
                
                for j2 in range(jF-1, -1, -1):
                    if (w[j2]==1 and bwimg[F[j2,0], F[j2,1]]==0):
                        break
                        
                lp2=0
                while (j2>0):
                    p2[lp2,0]=F[j2,0]
                    p2[lp2,1]=F[j2,1]
                    lp2+=1
                    j2=par[j2]
                
                for x in range(lp2-1,-1,-1):
                    pathway[lenpath,0]=p2[x,0]
                    pathway[lenpath,1]=p2[x,1]
                    lenpath+=1
                
                
                for x in range(jF):
                    check[F[x,0], F[x,1]]=0
                
                if (highx-lowx>shp[1]*Hyperparameters['Lineslayer']):
                    sv, gr=blackening(bwimg, pathway, lenpath)
                    pway=[]
                    for x in range(lenpath):
                        pway.append((pathway[x,0],pathway[x,1]))
                    
                    solution.append((sv, gr, lowx, highx, pway))
            y+=1
    
    return solution
