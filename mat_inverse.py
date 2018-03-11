# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:38:34 2018

@author: ainurrofiq
"""
import numpy as np
import itertools
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
import scipy as sc
import matplotlib.pyplot as plt


def voronoi_finite_polygons_2d(vor, point=None, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """
    

    new_regions = []
    new_vertices = vor.vertices.tolist()
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), l in zip(vor.ridge_points, vor.ridge_vertices):
        
        all_ridges.setdefault(p1, []).append(tuple([p2]+l))
        all_ridges.setdefault(p2, []).append(tuple([p1]+l))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        if p1 in point :
            vertices = vor.regions[region]
    
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue
            
            # reconstruct a non-finite region
            try :
                ridges = all_ridges[p1]
            except :
                continue
            new_region = [v for v in vertices if v >= 0]
            
            
            for ll in ridges:
                ll = list(ll)
                p2 = ll[0]
                vv = ll[1:len(ll)]
                if np.all(np.array(vv)>=0):
                    # finite ridge: already in the region
                    continue
                if np.any(np.array(vv)<0):
                    for nominus in range(len(vv)):
                        if vv[nominus]>=0:
                            vpoint = vor.vertices[vv[nominus]]
                            a = vor.points[p1]
                            c = vor.points[p2]
                            b = vpoint
                            ab = b-a
                            ac = c-a
                            p = np.dot(ac,ab)/np.sqrt(np.dot(ac,ac))
                            acn = ac/np.sqrt(np.dot(ac,ac))
                            ap = a+p*acn
                            bp = ap-b   
                            far_point = b+bp
                            new_region.append(len(new_vertices))
                            new_vertices.append(far_point.tolist())
                
            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
    
            # finish
            new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def misfit(data, calculated) :
    data, calculated = data.reshape(-1), calculated.reshape(-1)
    nonzero=np.argwhere(data>0)
    if len(nonzero)/len(data)<0.5:
        return 99
    else:
        data, calculated = data, calculated
        return (np.sum((data-calculated)**2/((data**2)*len(data))))**0.5

def ErrorPercentage(data, calculated) :
    data, calculated = data.reshape(-1), calculated.reshape(-1)
    nonzero=np.argwhere(data>0)
    if len(nonzero)/len(data)<0.5:
        return 99
    else:
        data, calculated = data[nonzero], calculated[nonzero]
        return np.mean((abs(data-calculated)/calculated)*100)

def forPlot(cvs, thk):
    depth, Vs = [], []
    dp = 0
    for idx in range(len(cvs)):
        Vs.append(cvs[idx])
        Vs.append(cvs[idx]) 
        if idx < len(cvs)-1:
            dp += thk[idx]
            depth.append(dp)
            depth.append(dp)
        else :
            dp += thk[idx-1]
            depth.append(dp)
    return Vs, [0]+depth

def PointToParam(points, lapisan):
    vsIt, vpIt, thkIt = [], [], []
    for j in range(len(points)):
        if j <= lapisan-1:
            vsIt.append(points[j])
        elif j >= lapisan and j<2*lapisan:
            vpIt.append(points[j])
        else:
            thkIt.append(points[j])
    return vsIt, vpIt, thkIt

    
def InOrOut(polygon, point):
    point=point.reshape((1,len(polygon[0])))
    idx = np.arange(0,len(point[0]))
    dim = 2 #see in dimension
    comb = list(itertools.combinations(idx,dim))
    statusAll = []
    for acomb in comb :
        hull = ConvexHull(polygon[:,acomb],)
        new_points = np.append(polygon[:,acomb],point[:,acomb], axis=0)
        new_hull = ConvexHull(new_points)
        if list(hull.vertices) == list(new_hull.vertices):
            status = True
        else:
            status = False
        statusAll.append(status)
    if np.any(np.array(statusAll)==False) :
        return False
    else :
        return True
    
def findMin(array, number,doubleasone=True):
    arrayCopy = np.copy(np.float64(array))
    idx = []
    for ni in range(number):
        if doubleasone is not True :
            if len(idx) == 0:
                idx.append(np.nanargmin(arrayCopy))
                arrayCopy[np.array(idx)]=np.NaN
            else:
                idx.append(np.nanargmin(arrayCopy))
                arrayCopy[np.array(idx)]=np.NaN
        else :
            if len(idx) == 0:
                idx.append(np.argmin(arrayCopy))
                # cek if double
                doubleID = np.where(arrayCopy[np.nanargmin(arrayCopy)]==array)
                arrayCopy[doubleID]=np.NaN
            else:
                # cek if double
                doubleID = np.where(arrayCopy[np.nanargmin(arrayCopy)]==array)
                idx.append(np.nanargmin(arrayCopy))
                arrayCopy[doubleID]=np.NaN
    return np.array(idx)

def gibbs(parameter_, npoints, border=None, center=None):
    if border is None :
        parameter = []
        for i in parameter_:
            parameter.append(np.array(i))
    else :
        parameter = []
        for j in range(len(border[0])):
            if len(np.argwhere(np.logical_and(border[:,j]>=parameter_[j][0],
                                              border[:,j]<=parameter_[j][1])))!=0:
                idx = np.argwhere(np.logical_and(border[:,j]>=parameter_[j][0],
                                                 border[:,j]<=parameter_[j][1]))
                
                parameter.append(np.array([border[idx,j].reshape(-1).min(),
                                           border[idx,j].reshape(-1).max()]))
            else:
                #Have To Check
                print 'WatchOut'
                minim = center[j]-center[j]*0.2
                maxim = center[j]+center[j]*0.2
                parameter.append([minim, maxim])
    if border is None :            
        param = []
        for perparam in parameter:
            param.append(np.mean(perparam))
    else :
        param = center  
        
    distribution = [param]
    while len(distribution)< npoints+1 :
        for j in range(len(parameter)) :
            coord = []
            for k in range(len(parameter)) :
                if j == k:
                    sp = parameter[k].min()+(parameter[k].max()-parameter[k].min())*np.random.random()
                    coord.append(sp)
                else :
                    coord.append(distribution[len(distribution)-1][k])
            distribution.append(coord)
    return np.array(distribution)


"""
thk:    vector of N layer thicknesses
dns:    vector of N+1 layer mass densities
cvp:    vector of N+1 complex-valued layer compression wave velocities
cvs:    vector of N+1 complex-valued layer shear wave velocities
freq:    vector of M frequencies in Hz
om = 2 * pi * freq
kmax = om / crmin
kmin = om / crmax
k1 = kmax
k2 = kmin
"""
# GLOBAL
# Tolerance for declaring a zero
global MAXROOT, NUMINC


# Assumed maximum number of modes at any frequency
MAXROOT = 1;

# Maximum number of increments to search between vrmin and vrmax
NUMINC = 100;

def psv(thk, dns, cvp, cvs, om, k):
    
    """
    This function calculates the E and Lambda matrices (up-going and
    down-going matrices) for the P-SV case. Note that a separate function,
    updown, is provided for calculating the Lambda matrices for use in
    determining the displacement-stress vectors.
    """
    
    cvs2 = cvs ** 2
    cvp2 = cvp ** 2
    mu = dns * cvs2
    
    e11 = np.zeros((2, 2, len(cvs)),dtype=complex)
    e12 = np.zeros((2, 2, len(cvs)),dtype=complex)
    e21 = np.zeros((2, 2, len(cvs)),dtype=complex)
    e22 = np.zeros((2, 2, len(cvs)),dtype=complex)
    du = np.zeros((2, 2, len(thk)),dtype=complex)

    if om == 0 :
    
        kappa = (1.0 + cvs2 / cvp2) / (1.0 - cvs2 / cvp2)
        kmu = k * mu
        
        e11[0, 0,:] = np.ones((1, len(cvs)),dtype=complex)
        e11[0, 1,:] = e11[0, 0,:]
        e12[0, 0,:] = e11[0, 0,:]
        e12[0, 1,:] = e11[0, 0,:]
        e11[1, 0,:] = - (kappa - 1.0)
        e11[1, 1,:] = e11[0, 0,:]
        e12[1, 0,:] = - e11[1, 0,:]
        e12[1, 1,:] = - e11[0, 0,:]
        e21[0, 0,:] = (kappa - 3.0) * kmu
        e21[0, 1,:] = - 2 * kmu
        e22[0, 0,:] = - e21[0, 0,:]
        e22[0, 1,:] = - e21[0, 1,:]
        e21[1, 0,:] = (kappa - 1.0) * kmu
        e21[1, 1,:] = - 2 * kmu
        e22[1, 0,:] = e21[1, 0,:]
        e22[1, 1,:] = e21[1, 1,:]
        
        du[0, 0,:] = np.exp(-k*thk, dtype=complex)
        du[1, 1,:] = np.exp(-k*thk, dtype=complex)
        du[1, 0,:] = - k * thk * np.exp(-k*thk, dtype=complex)
    
    else :
    
        k2 = k ** 2
        om2 = om ** 2
        
        ks2 = om2 / cvs2
        nus = np.sqrt(k2-ks2, dtype=complex)
        #print nus
        index = np.where((-1j*nus).imag > 0.0)

        nus[index] = - nus[index]
        gammas = nus / k
        
        kp2 = om2 / cvp2
        nup = np.sqrt(k2-kp2, dtype=complex)
        index = np.where((-1j*nup).imag > 0)
        nup[index] = - nup[index]
        
        gammap = nup / k
        
        chi = 2.0 * k - ks2 / k
        
        e11[0, 0,:] = - np.ones((1, len(cvs)),dtype=complex)
        e11[0, 1,:] = gammas
        e12[0, 0,:] = e11[0, 0,:]
        e12[0, 1,:] = gammas
        e11[1, 0,:] = - gammap
        e11[1, 1,:] = - e11[0, 0,:]
        e12[1, 0,:] = gammap
        e12[1, 1,:] = e11[0, 0,:]
        e21[0, 0,:] = 2 * mu * nup
        e21[0, 1,:] = - mu * chi
        e22[0, 0,:] = - e21[0, 0,:]
        e22[0, 1,:] = - e21[0, 1,:]
        e21[1, 0,:] = - e21[0, 1,:]
        e21[1, 1,:] = - 2 * mu * nus
        e22[1, 0,:] = - e21[0, 1,:]
        e22[1, 1,:] = e21[1, 1,:]
        
        du[0, 0,:] = np.exp(-nup[0:len(thk)]*thk, dtype=complex)
        du[1, 1,:] = np.exp(-nus[0:len(thk)]*thk, dtype=complex)
        
        
    return np.complex64(e11), np.complex64(e12), np.complex64(e21), np.complex64(e22), np.complex64(du), np.complex64(mu), np.complex64(nus), np.complex64(nup)

def modrt(e11, e12, e21, e22, du):
    
    """
    This function calculates the modified R/T coefficients
    """
    
    # Determine the number of layers, N, not including the half space
    N = du.shape[2]
    
    # Initialize a 4x4xN matrix
    X = np.zeros((4, 4, N),dtype=complex)
    
    # Loop through the first N-1 layers
    for j in range(N-1):
        A = np.concatenate((np.concatenate((e11[:,:, j+1],-e12[:,:, j]), axis=1),
                            np.concatenate((e21[:,:, j+1],-e22[:,:, j]), axis=1)))
        B = np.concatenate((np.concatenate((e11[:,:, j],-e12[:,:, j+1]), axis=1),
                            np.concatenate((e21[:,:, j],-e22[:,:, j+1]), axis=1)))
        L = np.concatenate((np.concatenate((du[:,:, j],np.zeros((2,2))), axis=1),
                            np.concatenate((np.zeros((2,2)),du[:,:, j+1]), axis=1)))
        X[:,:, j] = np.linalg.lstsq(A,np.matmul(B,L))[0]
    
    # Calculate the Nth layer
    A = np.concatenate((np.concatenate((e11[:,:, N],-e12[:,:, N-1]), axis=1),
                        np.concatenate((e21[:,:, N],-e22[:,:, N-1]), axis=1)))
    B = np.concatenate((np.matmul(e11[:,:, N-1], du[:,:, N-1]),
                        np.matmul(e21[:,:, N-1], du[:,:, N-1])))
    X[:, 0:2, N-1] = np.linalg.lstsq(A,B)[0]
    
    # Extract R/T submatrices
    td = X[0:2, 0:2,:]
    ru = X[0:2, 2:4,:]
    rd = X[2:4, 0:2,:]
    tu = X[2:4, 2:4,:]
    
    return np.complex64(td), np.complex64(tu), np.complex64(rd), np.complex64(ru)

def genrt(td, tu, rd, ru):
    N = td.shape[2]
    
    # Initialize 2x2xN matrices
    Td = np.zeros((2, 2, N),dtype=complex)
    Rd = np.zeros((2, 2, N),dtype=complex)

    # Calculate the Td and Rd matrices for the Nth layer
    Td[:,:, N-1] = td[:,:, N-1]
    Rd[:,:, N-1] = rd[:,:, N-1]
    
    for j in range(N-2,-1,-1):
        Td[:,:, j] = np.linalg.solve(np.eye(2) - np.matmul(ru[:,:, j], Rd[:,:, j+1]),td[:,:, j])
        Rd[:,:, j] = rd[:,:, j] + np.matmul(np.matmul(tu[:,:, j], Rd[:,:, j+1]),Td[:,:, j])
        
    return np.complex64(Td), np.complex64(Rd)

def secular(k, om, thk, dns, cvp, cvs):
    
    """
    This function calculates the absolute value of the secular function for
    a particular frequency and wavenumber.
    """
    
    epsilon = 0.0001
    while any(abs(om/k-cvs) < epsilon) or any(abs(om/k-cvp) < epsilon) :
        k = k * (1+epsilon)
    [e11, e12, e21, e22, du, mu, nus, nup] = psv(thk, dns, cvp, cvs, om, k)
    [td, tu, rd, ru] = modrt(e11, e12, e21, e22, du)
    [Td, Rd] = genrt(td, tu, rd, ru)
    
    #Note that the absolute value of the secular function is calculated
    d = abs(np.linalg.det(e21[:,:, 0]+np.matmul(np.matmul(e22[:,:, 0],du[:,:, 0]),Rd[:,:, 0]))/
            (nus[0] * nup[0] * mu[0] ** 2))

    return d

def homogeneous(cvp, cvs):
    # Define Poisson's Ratio
    nu = 0.5 * ((1.0*cvp * cvp - 2.0 * cvs * cvs) / (1.0*cvp * cvp - 1.0*cvs * cvs))
    
    # Define Coefficients of Rayleigh's Equation
    a = 1.0
    b = -8.0
    c = 8.0 * (3.0 - 2.0 * (cvs * cvs) / (1.0*cvp * cvp))
    d = 16.0 * ((1.0*cvs * cvs) / (1.0*cvp * cvp) - 1)
    
    # Solution of Rayleigh Equation
    p = np.array([a, b, c, d])
    x = np.roots(p)
    cr = cvs * np.sqrt(x,dtype=complex)
    
    # Determine which of the roots is correct using the estimated velocity (Achenbach, 1973)
    crest = cvs * ((0.862 + 1.14 * nu) / (1 + nu))
    index = np.where(abs(cr-crest) == min(abs(cr-crest)))
    cvr = abs(cr[index][0])
    if len(index) == 0:
        print 'No root found for homogeneous half space'
    return cvr

def modal(freq, thk, dns, cvp, cvs, crmin, crmax):
    """
    This function calculates the modal phase velocities in an elastic,
    vertically heterogeneous medium using search techniques.
    """
    
    # Initialize a matrix to store modal phase velocities
    cr = np.zeros((len(freq),MAXROOT))
    
    # Loop through the frequencies
    for j in range(len(freq)):
        if len(cvs) == 1:
            cr[j, 0] = homogeneous(cvp, cvs)
            
        else :
            numroot = -1;
            om = 2.0 * np.pi * freq[j]
            
            # Establish the search parameters
            kmax = om / crmin
            kmin = om / crmax
            dk = (kmax - kmin) / NUMINC
            
            # Establish the first and second points
            k1 = kmax
            f1 = secular(k1, om, thk, dns, cvp, cvs)
            k2 = kmax - dk
            f2 = secular(k2, om, thk, dns, cvp, cvs)
            
            # Establish an arbitrary high value for kold
            kold = 1.1 * kmax
            TOL = 1e+00
            # Loop through the remaining points
            for m in np.arange(1,NUMINC - 1):
                
                k3 = kmax - (m+1) * dk
                f3 = secular(k3, om, thk, dns, cvp, cvs)
                # Determine if a minimum is bracketed
                if (f2 < f1) and (f2 < f3) :
                    
                    # Use golden search/parabolic interpolation to refine minimun
                    if k3<k1 :
                        res = sc.optimize.minimize_scalar(secular, bounds=(k3, k1),
                                                          args=(om, thk, dns, cvp, cvs),
                                                          method='Bounded')
                    else :
                        res = sc.optimize.minimize_scalar(secular, bounds=(k1, k3),
                                                          args=(om, thk, dns, cvp, cvs),
                                                          method='Bounded')
                    ktrial, ftrial  = res.x, res.fun
                    
                    # Check to see if ktrial is a zero and different from the previous zero
                    if (ftrial < TOL and abs((ktrial - kold)/kold) > 0.001) :
                        numroot = numroot + 1;
                        cr[j, numroot] = om / ktrial
                        kold = ktrial
                        
                if numroot == MAXROOT -1:
                    break
                k1 = k2
                f1 = f2
                k2 = k3
                f2 = f3
                
    return cr
    
def mat_disperse(thk, dns, cvp, cvs, freq) :
    
    thk = np.array(thk)
    dns = np.array(dns)
    cvs = np.array(cvs)
    cvp = np.array(cvp)
    
    # Determine the minimum and maximum body wave velocities
    cvpmin = min(cvp)
    cvpmax = max(cvp)
    cvsmin = min(cvs)
    cvsmax = max(cvs)
    
    # Determine the minimum and maximum Rayleigh phase velocities in a
    # homogeneous half space corresponding to the minimum and maximum
    # compression and shear velocities
    vrmin = homogeneous(cvpmin, cvsmin)
    vrmax = homogeneous(cvpmax, cvsmax)
    
    # Note: the following empirical rules need further study
    vrmin = 0.98 * vrmin
    vrmax = 1.00 * cvsmax
    
    # Determine the modal phase velocities
    vr = modal(freq, thk, dns, cvp, cvs, vrmin, vrmax);
    return vr

def InversionDispersion(vr, freq, params, dns, iters, ns0, nr0, ns, nr, maxerr) :
    vrAll = []
    OutPoint, OutPolygon ,OutError , OutModel, OutVertices = [], [], [], [], []
    TotalModel, TotalVr, TotalError = [] , [] , []
    OutPoint_ = [None]*nr0
    OutPolygon_ = [None]*nr0
    OutError_ = [None]*nr0
    OutVertices_ = [None]*nr0
    for i in range(iters):
        if i == 0:
            points = gibbs(params,ns0)
            points = points[0:ns0]
            vor = Voronoi(points,qhull_options='QJ')
            vrAll_, OutModel_, OutError__ = [], [], []
            for j in range(len(vor.regions)):
                vsIt, vpIt, thkIt = PointToParam(points[j], lapisan=(len(params)+1)/3)
                vrNew = mat_disperse(thkIt, dns, vpIt, vsIt, freq)
                error = ErrorPercentage(vrNew, vr)
                vrAll_.append(vrNew)
                OutModel_.append([vsIt, vpIt, thkIt])
                OutError__.append(error)
                TotalModel.append([vsIt, vpIt, thkIt])
                TotalVr.append(vrNew)
                TotalError.append(error)
                print 'it = {:} | Model {:} | Error = {:.2f} % | ErrMin = {:.2f} %'.format(i+1,len(TotalModel),error,np.array(OutError__).min())
                
            idxMin1  = findMin(np.array(OutError__), nr0, doubleasone=False)
            regions, vertices = voronoi_finite_polygons_2d(vor, point=idxMin1)
            OutVertices.append(vertices)
            for ii,ni in enumerate(idxMin1):
                OutPolygon.append(regions[ii])
                OutPoint.append(vor.points[ni])
                OutError.append(OutError__[ni])
                OutModel.append(OutModel_[ni])
                vrAll.append(vrAll_[ni])
                
                
                OutPoint_[ii] = ([vor.points[ni]])
                OutError_[ii] = ([OutError__[ni]])
                OutVertices_[ii] = ([vertices])
                OutPolygon_[ii] = ([regions[ii]])
                
        elif i == 1 :
            for ij in range(len(idxMin1)) :
                bestRegions = OutPolygon[ij]
                bestPoints = OutPoint[ij]
                bestVertices = np.array(OutVertices[i-1])[bestRegions]
                points = gibbs(params,ns,
                               border = bestVertices,
                               center = bestPoints)
            
                points = points[1::1]
                points = points[0:ns]
                vor = Voronoi(points,qhull_options='QJ')
                vrAll_, OutModel_, OutError__ = [], [], []
                for j in range(len(vor.regions)):
                    vsIt, vpIt, thkIt = PointToParam(points[j], lapisan=(len(params)+1)/3)
                    vrNew = mat_disperse(thkIt, dns, vpIt, vsIt, freq)
                    error = ErrorPercentage(vrNew, vr)
                    vrAll_.append(vrNew)
                    OutModel_.append([vsIt, vpIt, thkIt])
                    OutError__.append(error)
                    TotalModel.append([vsIt, vpIt, thkIt])
                    TotalVr.append(vrNew)
                    TotalError.append(error)
                    print 'it = {:} | Model {:} | Error = {:.2f} % | ErrMin = {:.2f} %'.format(i+1,len(TotalModel),error,np.array(OutError).min())
                    
                idxMin2  = findMin(np.array(OutError__),nr, doubleasone=False)
                regions, vertices = voronoi_finite_polygons_2d(vor, point=idxMin2)
                            
                for ni,nn in enumerate(idxMin2):
                    OutPolygon.append(regions[ni])
                    OutPoint.append(vor.points[nn])
                    OutError.append(OutError__[nn])
                    OutModel.append(OutModel_[nn])
                    vrAll.append(vrAll_[nn])
                    
                    OutPoint_[ij].append(vor.points[nn].reshape(-1))
                    OutVertices_[ij].append(vertices)
                    OutError_[ij].append(OutError__[nn])
                    OutPolygon_[ij].append(regions[ni])
                    
        else :
            for ij in range(nr0) :
                minErrs = findMin(np.array(OutError_[ij]), nr, doubleasone=False)
                for minErr in minErrs :
                    bestRegions = OutPolygon_[ij][minErr]
                    bestPoints = OutPoint_[ij][minErr]
                    bestVertices = np.array(OutVertices_[ij][minErr])[bestRegions]
                    points = gibbs(params,ns,
                                   border = bestVertices,
                                   center = bestPoints)
                    points = points[1::1]
                    points = points[0:ns]
                    vor = Voronoi(points,qhull_options='QJ')
                    vrAll_, OutModel_, OutError__ = [], [], []
                    for j in range(len(vor.regions)):
                        vsIt, vpIt, thkIt = PointToParam(points[j], lapisan=(len(params)+1)/3)
                        vrNew = mat_disperse(thkIt, dns, vpIt, vsIt, freq)
                        error = ErrorPercentage(vrNew, vr)
                        vrAll_.append(vrNew)
                        OutModel_.append([vsIt, vpIt, thkIt])
                        OutError__.append(error)                
                        TotalModel.append([vsIt, vpIt, thkIt])
                        TotalVr.append(vrNew)
                        TotalError.append(error)
                        print 'it = {:} | Model {:} | Error = {:.2f} % | ErrMin = {:.2f} %'.format(i+1,len(TotalModel),error,np.array(OutError).min())
                        
                    idxMin3  = findMin(np.array(OutError__), 1, doubleasone=False)
                    regions, vertices = voronoi_finite_polygons_2d(vor, point=idxMin3)
                    OutPoint_[ij].append(vor.points[idxMin3].reshape(-1))
                    OutVertices_[ij].append(vertices)
                    OutError_[ij].append(OutError__[idxMin3[0]])
                    OutPolygon_[ij].append(regions[0])
                    for ni,nn in enumerate(idxMin3):
                        OutPolygon.append(regions[ni])
                        OutPoint.append(vor.points[nn])
                        OutError.append(OutError__[nn])
                        OutModel.append(OutModel_[nn])
                        vrAll.append(vrAll_[nn])
        if  np.array(TotalError).min()<maxerr:
            break
    return TotalVr, TotalModel, TotalError
        
        
##############################################################################
##############################################################################
##############################################################################
import time
start = time.time()     
np.random.seed(12345)

# 1-D Model
dns = [1.7,1.8,1.8,1.8]
cvs = [200,400,600,950]
cvp = [300,650,950,1200]
thk = [30,50,50]

# Forward Model
freqmin, freqmax = 0.2, 20
freq = 10**(np.linspace(np.log10(freqmin),np.log10(freqmax),20))
vr= mat_disperse(thk, dns, cvp, cvs, freq)

# Try Inversion
# Fix Density
parameter = [cvs, cvp, thk]
params = []
for i in range(len(parameter)):
    for j in range(len(parameter[i])):
        params.append([parameter[i][j]-0.3*parameter[i][j],
                       parameter[i][j]+0.5*parameter[i][j]])



# Setting
iters = 12
ns0 = 45
nr0 = 1
ns = 15
nr= 3
maxerr = 2 #%


TotalVr, TotalModel, TotalError = InversionDispersion(vr, freq, params, dns, iters, ns0, nr0, ns, nr, maxerr)

# -----------------------------------------------------------------------------
# PLOT RESULT
# -----------------------------------------------------------------------------
idErr = np.argwhere(np.logical_and(np.array(TotalError)<50,np.array(TotalError)>0))
Err, Mod, Vr = [], [], []
for ii in idErr :
    Err.append(TotalError[ii[0]])
    Mod.append(TotalModel[ii[0]])
    Vr.append(TotalVr[ii[0]])
    
# Actual Model
Vs,depth = forPlot(cvs, thk)

# Best Model
BestModel = Mod[np.argmin(Err)]
BestVr = Vr[np.argmin(Err)]
bVs, bdepth = forPlot(BestModel[0],BestModel[2])

N = len(Err)

fig=plt.figure(figsize=(6,9))
ax = fig.add_subplot(111)
for ii in range(N):
    Model = Mod[ii]
    Vsi, depthi = forPlot(Model[0],Model[2])
    maxDepth = np.max([np.array(depth).max(),np.array(depthi).max()])
    depth[len(depth)-1] = maxDepth
    depthi[len(depthi)-1] = maxDepth
    ax.plot(Vsi,depthi, '-', color='gray')
    
ax.plot(bVs,bdepth, 'b--',lw=5)  
ax.plot(Vs,depth,'r--', lw=5)   
ax.invert_yaxis()  
ax.set_xlim(xmin=0)
ax.set_ylim(np.min([np.array(depth).max(),np.array(bdepth).max()]),0)
ax.set_xlabel('Vs [m/s]')
ax.set_ylabel('Depth [m]')
plt.title('Error = {:.4f}'.format(Err[np.argmin(Err)]))
plt.grid()    


plt.figure()
for ii in range(N):
    #plt.plot(freq, Vr[ii],color=plt.cm.RdYlBu(TotalError[ii]/np.max(Err)))
    plt.plot(freq, Vr[ii], color='gray')
    #print Err[ii], np.sum(Vr[ii]-vrNew)
plt.semilogx(freq, BestVr, 'b--',lw=5)
plt.semilogx(freq, vr,'^', c='r')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase Velocity [m/s]')
plt.title('Error = {:.4f}'.format(Err[np.argmin(Err)]))
plt.ylim(ymin=0)
plt.grid()
print 'Finished = {:.2f} Menit'.format((time.time()-start)/60.0)
##############################################################################
