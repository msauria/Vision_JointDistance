# distutils: language = c++
#cython: language_level=3

import cython
cimport numpy as np
import numpy

ctypedef np.float32_t DTYPE_t
ctypedef np.float64_t DTYPE_64_t
ctypedef np.int32_t DTYPE_int_t
ctypedef np.int64_t DTYPE_int64_t
ctypedef np.uint32_t DTYPE_uint_t
ctypedef np.int8_t DTYPE_int8_t
cdef double Inf = numpy.inf

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    double log10(double x) nogil
    double sqrt(double x) nogil
    double pow(double x, double x) nogil
    double abs(double x) nogil
    double round(double x) nogil
    double floor(double x) nogil
    double ceil(double x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def predicted(
        np.ndarray[DTYPE_int_t, ndim=2] tads not None,
        np.ndarray[DTYPE_int_t, ndim=2] EP not None,
        np.ndarray[DTYPE_64_t, ndim=2] rna not None,
        np.ndarray[DTYPE_int_t, ndim=1] strand,
        np.ndarray[DTYPE_64_t, ndim=3] pfeatures,
        np.ndarray[DTYPE_64_t, ndim=3] cfeatures not None,
        np.ndarray[DTYPE_int_t, ndim=2] cre_ranges not None,
        np.ndarray[DTYPE_64_t, ndim=2] contactP not None,
        np.ndarray[DTYPE_64_t, ndim=3] betas not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss_coords not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_coords not None,
        np.ndarray[DTYPE_64_t, ndim=2] cBetas not None,
        np.ndarray[DTYPE_64_t, ndim=2] predicted not None,
        int prom_dist):
    cdef long long int featureN = betas.shape[1]
    cdef long long int stateN = betas.shape[2]
    cdef long long int tssN = rna.shape[0]
    cdef long long int cellN = rna.shape[1]
    cdef long long int creN = cfeatures.shape[0]
    cdef long long int tadN = tads.shape[0]
    cdef long long int cre_feat = featureN - 1
    cdef int tss_listN, cre_listN, i, j, k, l, tss_index, ce, cs, tss_coord, cre_coord
    cdef double cp
    # I=tssN, N=cellN, M=stateN, K=creN
    # C = sum_I( sum_N( O_in - P_in)^2 ) )
    # P_in = sum_M( Pr_inm * Bp_m ) + sum_Ki( Cp_ik / (H + Cp_ik) * sum_M( Cr_knm * Bc_m ) )
    with nogil:
        for i in range(creN):
            for j in range(cellN):
                cBetas[i, j] = 0
                for k in range(stateN):
                    cBetas[i, j] += cfeatures[i, j, k] * betas[0, cre_feat, k]
        if pfeatures is not None:
            for i in range(tssN):
                for j in range(cellN):
                    predicted[i, j] = 0
                    for k in range(stateN):
                        predicted[i, j] += pfeatures[i, j, k] * betas[0, 0, k]
        for i in range(tadN):
            tss_listN = 0
            cre_listN = 0
            for j in range(tads[i, 0], tads[i, 1]):
                if EP[j, 1] == 1:
                    tss_list[tss_listN] = EP[j, 0]
                    tss_coords[tss_listN] = EP[j, 2]
                    tss_listN += 1
                else:
                    cre_list[cre_listN] = EP[j, 0]
                    cre_coords[cre_listN] = EP[j, 2]
                    cre_listN += 1
            cre_listN -= 1
            for j in range(tss_listN):
                tss_index = tss_list[j]
                tss_coord = tss_coords[j]
                cs = cre_ranges[tss_index, 0]
                ce = cre_ranges[tss_index, 1]
                if cre_listN >= 0:
                    for k in range(cre_list[0], cre_list[cre_listN] + 1):
                        cre_coord = cre_coords[k - cre_list[0]]
                        if strand is None:
                            if abs(tss_coord - cre_coord) < prom_dist:
                                continue
                        elif strand[tss_index] * (tss_coord - cre_coord) < prom_dist:
                            continue
                        cp = contactP[tss_index, k - cs]
                        for l in range(cellN):
                            predicted[tss_index, l] += cp * cBetas[k, l]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gradient(
        np.ndarray[DTYPE_int_t, ndim=2] tads not None,
        np.ndarray[DTYPE_int_t, ndim=2] EP not None,
        np.ndarray[DTYPE_64_t, ndim=2] rna not None,
        np.ndarray[DTYPE_int_t, ndim=1] strand,
        np.ndarray[DTYPE_64_t, ndim=3] pfeatures,
        np.ndarray[DTYPE_64_t, ndim=3] cfeatures not None,
        np.ndarray[DTYPE_int_t, ndim=2] cre_ranges not None,
        np.ndarray[DTYPE_64_t, ndim=2] contactP not None,
        np.ndarray[DTYPE_64_t, ndim=3] EP_distance not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss not None,
        np.ndarray[DTYPE_64_t, ndim=3] betas not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss_coords not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_coords not None,
        np.ndarray[DTYPE_64_t, ndim=1] agradient not None,
        np.ndarray[DTYPE_64_t, ndim=2] bgradient not None,
        np.ndarray[DTYPE_64_t, ndim=2] predicted not None,
        np.ndarray[DTYPE_64_t, ndim=1] dC_dP not None,
        int prom_dist):
    cdef long long int featureN = betas.shape[1]
    cdef long long int stateN = betas.shape[2]
    cdef long long int tssN = tss.shape[0]
    cdef long long int cellN = rna.shape[1]
    cdef long long int maxCRE = contactP.shape[1]
    cdef long long int maxTSS = tss_list.shape[0]
    cdef long long int tadN = tads.shape[0]
    cdef long long int cre_feat = betas.shape[1] - 1
    cdef int tss_listN, cre_listN, i, j, k, l, m, tss_index, cs, tss_coord, cre_coord
    cdef double cp, cf, b, lnD
    # I=tssN, N=cellN, M=stateN, K=creN
    # C = sum_I( sum_N( O_in - P_in)^2 ) )
    # P_in = sum_M( Pr_inm * Bp_m ) + sum_Ki( Cd_ik^alpha * sum_M( Cr_knm * Bc_m ) )
    # dC/dP_in = -2(O_in - P_in)
    # dP/dBp_m = sum_I( sum_N( Pr_inm ) )
    # dP/dBc_m = sum_I( sum_N( sum_Ki( Cd^alpha * Cr_knm ) )
    # dP/dAlpha = sum_I( sum_N( sum_M( sum_Ki( Cr_knm * Bc_m * Cd^alpha * ln(Cd) ) ) ) )
    with nogil:
        for i in range(tadN):
            tss_listN = 0
            cre_listN = 0
            for j in range(tads[i, 0], tads[i, 1]):
                if EP[j, 1] == 1:
                    tss_list[tss_listN] = EP[j, 0]
                    tss_coords[tss_listN] = EP[j, 2]
                    tss_listN += 1
                else:
                    cre_list[cre_listN] = EP[j, 0]
                    cre_coords[cre_listN] = EP[j, 2]
                    cre_listN += 1
            cre_listN -= 1
            for j in range(tss_listN):
                tss_index = tss_list[j]
                tss_coord = tss_coords[j]
                cs = cre_ranges[tss_index, 0]
                for  m in range(cellN):
                    dC_dP[m] = 2 * (predicted[tss_index, m] - rna[tss_index, m])
                if cre_listN >= 0:
                    for k in range(cre_list[0] - cs, cre_list[cre_listN] + 1 - cs):
                        cre_coord = cre_coords[k - cre_list[0]]
                        if strand is None:
                            if abs(tss_coord - cre_coord) < prom_dist:
                                continue
                        elif strand[tss_index] * (tss_coord - cre_coord) < prom_dist:
                            continue
                        lnD = EP_distance[tss_index, k, 1]
                        cp = contactP[tss_index, k]
                        for l in range(stateN):
                            b = betas[0, cre_feat, l] * lnD
                            for m in range(cellN):
                                cf = cfeatures[k + cs, m, l] * cp * dC_dP[m]
                                bgradient[cre_feat, l] -= cf
                                agradient[0] -= cf * b
                if pfeatures is not None:
                    for m in range(cellN):
                        for l in range(stateN):
                            bgradient[0, l] -= (
                                pfeatures[tss_index, m, l] * dC_dP[m])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_tads(
        np.ndarray[DTYPE_int_t, ndim=2] EP not None,
        np.ndarray[DTYPE_64_t, ndim=2] rna not None,
        np.ndarray[DTYPE_int_t, ndim=1] strand,
        np.ndarray[DTYPE_64_t, ndim=3] cfeatures not None,
        np.ndarray[DTYPE_int_t, ndim=2] cre_ranges not None,
        np.ndarray[DTYPE_64_t, ndim=2] contactP not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss not None,
        np.ndarray[DTYPE_64_t, ndim=3] betas not None,
        np.ndarray[DTYPE_64_t, ndim=2] predicted not None,
        np.ndarray[DTYPE_64_t, ndim=2] tad_pred not None,
        np.ndarray[DTYPE_64_t, ndim=2] cBetas not None,
        np.ndarray[DTYPE_64_t, ndim=3] inTad_cres not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss_coords not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_coords not None,
        np.ndarray[DTYPE_int_t, ndim=1] paths not None,
        np.ndarray[DTYPE_64_t, ndim=1] scores not None,
        np.ndarray[DTYPE_int_t, ndim=2] tads not None,
        int prom_dist):
    cdef long long int featureN = betas.shape[1]
    cdef long long int stateN = betas.shape[2]
    cdef long long int tssN = tss.shape[0]
    cdef long long int epN = EP.shape[0]
    cdef long long int cellN = rna.shape[1]
    cdef long long int creN = cfeatures.shape[0]
    cdef long long int maxCRE = contactP.shape[1]
    cdef long long int maxTSS = tss_list.shape[0]
    cdef long long int cre_feat = featureN - 1
    cdef long long int tss_listN, cre_listN, i, j, k, l, m, n, p, tss_i, prev_best_path
    cdef long long int tss_index, cre_index, inv_j, start, end, tadN, cs, ce
    cdef int cre_coord, tss_coord
    cdef double mse, cp, prev_best_score
    # I=tssN, N=cellN, M=stateN, K=creN
    # C = sum_I( sum_N( (O_in - P_in)^2 ) ) )
    # P_in = sum_M( Pr_inm * Bp_m ) + sum_Ki( Cp_ik / (H + Cp_ik) * sum_M( Cr_knm * Bc_m ) )
    with nogil:
        paths[0] = 0
        scores[0] = 0
        for i in range(epN + 1):
            paths[i] = -1
            scores[i] = Inf
        for i in range(creN):
            for j in range(cellN):
                cBetas[i, j] = 0
                for k in range(stateN):
                    cBetas[i, j] += cfeatures[i, j, k] * betas[0, cre_feat, k]
        for i in range(tssN):
            cs = cre_ranges[i, 0]
            ce = cre_ranges[i, 1]
            for j in range(ce - cs):
                cp = contactP[i, j]
                for k in range(cellN):
                    inTad_cres[i, k, j] = cp * cBetas[j + cs, k]
        prev_best_score = 0
        prev_best_path = 0
        for tss_i in range(tssN):
            i = tss[tss_i] # TSS index in EP
            # Try each possible TAD starting point, up to but not including
            # the next upstream TSS 
            for j in range(EP[i, 3], i + 1):
                if scores[j] == Inf:
                    scores[j] = prev_best_score
                    paths[j] = prev_best_path
            for inv_j in range(i - EP[i, 3] + 1):
                j = i - inv_j # start index for TAD
                cre_listN = 0
                tss_listN = 0
                # Add all included upstream CREs to list
                for k in range(j, i):
                    cre_list[cre_listN] = EP[k, 0]
                    cre_listN += 1
                # Try each possible end point
                for k in range(i, EP[j, 4]): # end index for TAD
                    mse = scores[j]
                    if EP[k, 1] == 1:
                        prev_best_score = Inf
                        # If the next feature is a TSS, add to the tss list
                        tss_index = EP[k, 0]
                        tss_coord = EP[k, 2]
                        tss_list[tss_listN] = tss_index
                        tss_coords[tss_listN] = tss_coord
                        # Initialize the predicted expression
                        for m in range(cellN):
                            tad_pred[tss_listN, m] = predicted[tss_index, m]
                        # Find the CRE contribution to the predicted value
                        cs = cre_ranges[tss_index, 0]
                        ce = cre_ranges[tss_index, 1]
                        if cre_listN > 0:
                            for l in range(cre_list[0],
                                           cre_list[cre_listN - 1] + 1):
                                cre_coord = cre_coords[l - cre_list[0]]
                                if strand is None:
                                     if abs(tss_coord - cre_coord) < prom_dist:
                                        continue
                                elif strand[tss_index] * (tss_coord - cre_coord) < prom_dist:
                                    continue
                                for m in range(cellN):
                                    tad_pred[tss_listN, m] += (
                                        inTad_cres[tss_index, m, l - cs])
                        tss_listN += 1
                    else:
                        cre_index = EP[k, 0]
                        cre_coord = EP[k, 2]
                        cre_list[cre_listN] = cre_index
                        cre_coords[cre_listN] = cre_coord
                        for l in range(tss_listN):
                            tss_index = tss_list[l]
                            tss_coord = tss_coords[l]
                            if strand is None:
                                if abs(tss_coord - cre_coord) < prom_dist:
                                    continue
                            elif strand[tss_index] * (tss_coord - cre_coord) < prom_dist:
                                continue
                            cs = cre_index - cre_ranges[tss_index, 0]
                            for m in range(cellN):
                                tad_pred[l, m] += (
                                    inTad_cres[tss_index, m, cs])
                        cre_listN += 1
                    for l in range(tss_listN):
                        tss_index = tss_list[l]
                        for m in range(cellN):
                            mse += pow(tad_pred[l, m] - rna[tss_index, m], 2)
                    # If this is the best path to the endpoint,
                    # update scores and paths
                    if mse < scores[k + 1]:
                        scores[k + 1] = mse
                        paths[k + 1] = j
                    if mse < prev_best_score:
                        prev_best_score = mse
                        prev_best_path = k + 1
        # Find the best endpoint score that includes the last TSS
        j = epN
        mse = scores[j]
        start = j
        while j > 1 and EP[j - 1, 1] != 1:
            j -= 1
            if scores[j] < mse:
                mse = scores[j]
                start = j
        # Backtrack to find the TADs that give the best score
        tadN = 0
        while start > 0:
            end = start
            start = paths[end]
            k = 0
            for i in range(start, end):
                if EP[i, 1] == 1:
                    k += 1
            if k > 0:
                tads[tadN, 0] = start
                tads[tadN, 1] = end
                tadN += 1
    return tadN


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_erps(
        np.ndarray[DTYPE_int_t, ndim=2] EP not None,
        np.ndarray[DTYPE_int_t, ndim=1] strand,
        np.ndarray[DTYPE_64_t, ndim=3] cfeatures not None,
        np.ndarray[DTYPE_64_t, ndim=3] pfeatures,
        np.ndarray[DTYPE_int_t, ndim=2] cre_ranges not None,
        np.ndarray[DTYPE_64_t, ndim=2] contactP not None,
        np.ndarray[DTYPE_int_t, ndim=2] tads not None,
        np.ndarray[DTYPE_64_t, ndim=3] betas not None,
        np.ndarray[DTYPE_64_t, ndim=2] cBetas not None,
        np.ndarray[DTYPE_64_t, ndim=2] pBetas,
        np.ndarray[DTYPE_int_t, ndim=1] tss_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_list not None,
        np.ndarray[DTYPE_int_t, ndim=1] tss_coords not None,
        np.ndarray[DTYPE_int_t, ndim=1] cre_coords not None,
        np.ndarray[DTYPE_64_t, ndim=2] erps not None,
        np.ndarray[DTYPE_int_t, ndim=2] indices not None,
        int prom_dist):
    cdef long long int featureN = betas.shape[1]
    cdef long long int stateN = betas.shape[2]
    cdef long long int cellN = erps.shape[1]
    cdef long long int creN = cfeatures.shape[0]
    cdef long long int cre_feat = featureN - 1
    cdef long long int tadN = tads.shape[0]
    cdef long long int tss_listN, cre_listN, i, j, k, l, tssN
    cdef long long int tss_index, cre_index, cs, ts, te, pos
    cdef int cre_coord, tss_coord
    if pfeatures is not None:
        tssN = pBetas.shape[0]
    else:
        tssN = 0
    cdef double mse, cp, prev_best_score
    with nogil:
        for i in range(creN):
            for j in range(cellN):
                cBetas[i, j] = 0
                for k in range(stateN):
                    cBetas[i, j] += cfeatures[i, j, k] * betas[0, cre_feat, k]
        if pfeatures is not None:
            for i in range(tssN):
                for j in range(cellN):
                    pBetas[i, j] = 0
                    for k in range(stateN):
                        pBetas[i, j] += pfeatures[i, j, k] * betas[0, 0, k]
        pos = 0
        for i in range(tadN):
            ts = tads[i, 0]
            te = tads[i, 1]
            tss_listN = 0
            cre_listN = 0
            for j in range(ts, te):
                if EP[j, 1] == 1:
                    tss_list[tss_listN] = EP[j, 0]
                    tss_coords[tss_listN] = EP[j, 2]
                    tss_listN += 1
                else:
                    cre_list[cre_listN] = EP[j, 0]
                    cre_coords[cre_listN] = EP[j, 2]
                    cre_listN += 1
            for j in range(tss_listN):
                tss_index = tss_list[j]
                tss_coord = tss_coords[j]
                cs = cre_ranges[tss_index, 0]
                if pfeatures is not None:
                    for l in range(cellN):
                        erps[pos, l] = pBetas[tss_index, l]
                    indices[pos, 0] = tss_index
                    indices[pos, 1] = -1
                    pos += 1
                for k in range(cre_listN):
                    cre_coord = cre_coords[k]
                    if strand is None:
                        if abs(tss_coord - cre_coord) < prom_dist:
                            continue
                    elif strand[tss_index] * (tss_coord - cre_coord) < prom_dist:
                        continue
                    cre_index = cre_list[k]
                    cp = contactP[tss_index, cre_index - cs]
                    for l in range(cellN):
                        erps[pos, l] = cBetas[cre_index, l] * cp
                    indices[pos, 0] = tss_index
                    indices[pos, 1] = cre_index
                    pos += 1




















