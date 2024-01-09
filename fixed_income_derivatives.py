import numpy as np
from scipy.special import ndtr
from scipy.stats import norm, ncx2
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermfit, hermval, hermder
import copy

def spot_rates_to_zcb(T,spot_rate):
    M = len(T)
    p = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-8:
            p[i] = 1
        else:
            p[i] = np.exp(-spot_rate[i]*T[i])
    return p

def zcb_to_spot_rates(T,p):
    M = len(T)
    R = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-8:
            R[i] = 0
        else:
            R[i] = -np.log(p[i])/T[i]
    return R

def zcb_to_forward_rates(T,p,horizon = 1):
    # horizon = 0 corresponds to instantaneous forward rates
    M = len(T)
    f = np.zeros([M])
    if horizon == 0:
        f[0] = (np.log(p[0])-np.log(p[1]))/(T[1]-T[0])
        f[-1] = (np.log(p[-2])-np.log(p[-1]))/(T[-1]-T[-2])
        m = 1
        while m < M - 1.5:
            f[m] = (np.log(p[m-1])-np.log(p[m+1]))/(T[m+1]-T[m-1])
            m += 1
    elif 0 < horizon:
        m = horizon
        while m < M - 0.5:
            f[m] = (np.log(p[m-horizon])-np.log(p[m]))/(T[m]-T[m-horizon])
            m += 1
    return f

def zcb_to_forward_LIBOR_rates(T,p,horizon = 1):
    M = len(T)
    f = np.zeros([M])
    i = horizon
    while i < M - 0.5:
        f[i] = (p[i-horizon]-p[i])/(p[i]*(T[i]-T[i-horizon]))
        i += 1
    return f

def zcb_to_accrual_factor(T_n,T_N,fixed_freq,T,p):
    if fixed_freq == "quarterly":
        p_fix = np.zeros([int(T_N-T_n)*4])
        T_fix = np.array([T_n + i*0.25 for i in range(1,int(T_N-T_n)*4 + 1)])
    elif fixed_freq == "semiannual":
        p_fix = np.zeros([int(T_N-T_n)*2])
        T_fix = np.array([T_n + i*0.5 for i in range(1,int(T_N-T_n)*2 + 1)])
    elif fixed_freq == "annual":
        p_fix = np.zeros([int(T_N-T_n)])
        T_fix = np.array([T_n + i*0.25 for i in range(1,int(T_N-T_n) + 1)])
    S = 0
    for i in range(0,len(T_fix)):
        I_fix, idx_fix = value_in_list_returns_I_idx(T_fix[i],T)
        if I_fix is True:
            T_fix[i] = T[idx_fix]
            p_fix[i] = p[idx_fix]
            if i == 0:
                S += T_fix[i]*p_fix[i]
            else:
                S += (T_fix[i]-T_fix[i-1])*p_fix[i]
    return S

def zcb_to_par_swap_rate(T_n,T_N,fixed_freq,T,p):
    I_n, idx_n = value_in_list_returns_I_idx(T_n,T)
    I_N, idx_N = value_in_list_returns_I_idx(T_N,T)
    D = p[idx_n]-p[idx_N]
    S = zcb_to_accrual_factor(T_n,T_N,fixed_freq,T,p)
    R = D/S
    return R

def spot_rate_bump(T_bump,size_bump,T,R_input,p_input):
    R, p = R_input.copy(), p_input.copy()
    if type(T_bump) == int or type(T_bump) == float or type(T_bump) == np.float64:
        I_bump, idx_bump = value_in_list_returns_I_idx(T_bump,T)
        R[idx_bump] = R[idx_bump] + size_bump
        p[idx_bump] = np.exp(-R[idx_bump]*T_bump)
    elif type(T_bump) == tuple or type(T_bump) == list or type(T_bump) == np.ndarray:
        if type(size_bump) == int or type(size_bump) == float or type(size_bump) == np.float64:
            for i in range(0,len(T_bump)):
                I_bump, idx_bump = value_in_list_returns_I_idx(T_bump[i],T)
                R[idx_bump] = R[idx_bump] + size_bump
                p[idx_bump] = np.exp(-R[idx_bump]*T_bump[i])
        elif type(size_bump) == tuple or type(size_bump) == list or type(size_bump) == np.ndarray:
            for i in range(0,len(T_bump)):
                I_bump, idx_bump = value_in_list_returns_I_idx(T_bump[i],T)
                R[idx_bump] = R[idx_bump] + size_bump[i]
                p[idx_bump] = np.exp(-R[idx_bump]*T_bump[i])
    return R, p

def market_rate_bump(idx_bump,size_bump,data,interpolation_options = {"method": "linear"},resolution = 1):
    data_bump = copy.deepcopy(data)
    if type(idx_bump) == int or type(idx_bump) == float or type(idx_bump) == np.float64:
        data_bump[idx_bump]["rate"] += size_bump
        T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
        p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_fit_bump,R_fit_bump,interpolation_options = interpolation_options,resolution = resolution)
    elif type(idx_bump) == tuple or type(idx_bump) == list or type(idx_bump) == np.ndarray:
        if type(size_bump) == int or type(size_bump) == float or type(size_bump) == np.float64:
            for i in range(0,len(idx_bump)):
                data_bump[idx_bump[i]]["rate"] += size_bump
            T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
            p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_fit_bump,R_fit_bump,interpolation_options = interpolation_options,resolution = resolution)
        elif type(size_bump) == tuple or type(size_bump) == list or type(size_bump) == np.ndarray:
            for i in range(0,len(idx_bump)):
                data_bump[idx_bump[i]]["rate"] += size_bump[i]
            T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
            p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_fit_bump,R_fit_bump,interpolation_options = interpolation_options,resolution = resolution)
    return p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump, data_bump

##########################################
### Ho-Lee with Nelson-Siegel function ###
##########################################
def f_ns(param,tau):
    if type(tau) == int or type(tau) == float:
        f_inf, a, b = param
        K = len(a)
        f = f_inf + a[0]*np.exp(-b[0]*tau)
        for k in range(1,K):
            f += a[k]*tau**k*np.exp(-b[k]*tau)
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        f_inf, a, b = param
        K = len(a)
        M = len(tau)
        f = np.zeros([M])
        for m in range(0,M):
            f[m] = f_inf + a[0]*np.exp(-b[0]*tau[m])
            for k in range(1,K):
                f[m] += a[k]*tau[m]**k*np.exp(-b[k]*tau[m])
    return f

def fit_forward_rate_ns_obj(param,f_star,tau,scaling):
    """ Fit Forward rates to data for Ho-Lee."""
    K = int((len(param)-1)/2 - 1)
    f_inf, a, b = param[0], param[1:1+K+1], param[K+2:K+2+K+1]
    param_fit = f_inf, a, b
    M = len(tau)
    f_fit = f_ns(param_fit,tau) # reference to other function
    y = 0
    for m in range(0,M):
        y += scaling*(f_fit[m] - f_star[m])**2
    return y

def fit_forward_rate_ns_jac(param,f_star,tau,scaling):
    N = int((len(param)-1)/2 - 1)
    param_fit = param[0], param[1:1+N+1], param[N+2:N+2+N+1]
    M = len(tau)
    f_fit = f_ns(param_fit,tau)
    jac = np.zeros([len(param)])
    for m in range(0,M):
        jac += scaling*2*(f_fit[m] - f_star[m])*f_ns_jac(param_fit,tau[m])
    return jac

def fit_forward_rate_ns_hess(param,f_star,tau,scaling):
    N = int((len(param)-1)/2 - 1)
    param_fit = param[0], param[1:1+N+1], param[N+2:N+2+N+1]
    M = len(tau)
    f_fit = f_ns(param_fit,tau)
    hess = np.zeros([len(param),len(param)])
    for m in range(0,M):
        # f_deriv = fid.f_ns_jac(param_fit,T[m])
        f_deriv = f_ns_jac(param_fit,tau[m])
        hess += scaling*2*(f_fit[m] - f_star[m])*f_ns_hess(param_fit,tau[m]) + 2*np.outer(f_deriv,f_deriv)
    return hess

def f_ns_jac(param,tau):
    f_inf, a, b = param
    N = len(a) - 1
    jac = np.zeros([2*(N+1)+1])
    jac[0] = 1
    for n in range(0,N+1):
        jac[1+n] = tau**n*np.exp(-b[n]*tau)
        jac[1+N+1+n] = -a[n]*tau**(n+1)*np.exp(-b[n]*tau)
    return jac

def f_ns_hess(param,tau):
    f_inf, a, b = param
    N = len(a) - 1
    hess = np.zeros([2*(N+1)+1,2*(N+1)+1])
    for n in range(0,N+1):
        hess[1+n,1+N+1+n] = - tau**(n+1)*np.exp(-b[n]*tau)
        hess[1+N+1+n,1+n] = - tau**(n+1)*np.exp(-b[n]*tau)
        hess[1+N+1+n,1+N+1+n] = a[n]*tau**(n+2)*np.exp(-b[n]*tau)
    return hess

def f_ns_T(param,tau):
    """ Derivative. """
    if type(tau) == int or type(tau) == float:
        f_inf, a, b = param
        N = len(a)
        f_T = -a[0]*b[0]*np.exp(-b[0]*tau)
        for n in range(1,N):
            f_T += a[n]*n*tau**(n-1)*np.exp(-b[n]*tau) - a[n]*b[n]*tau**n*np.exp(-b[n]*tau)
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        f_inf, a, b = param
        N = len(a)
        M = len(tau)
        f_T = np.zeros([M])
        for m in range(0,M):
            f_T[m] = -a[0]*b[0]*np.exp(-b[0]*tau[m])
            for n in range(1,N):
                f_T[m] += a[n]*n*tau[m]**(n-1)*np.exp(-b[n]*tau[m]) - a[n]*b[n]*tau[m]**n*np.exp(-b[n]*tau[m])
    return f_T

def theta_ns(param,tau):
    if type(tau) == int or type(tau) == float:
        f_inf, a, b, sigma = param
        K = len(a)
        theta = -a[0]*b[0]*np.exp(-b[0]*tau) + sigma**2*tau
        for k in range(1,K):
            theta += a[k]*k*tau**(k-1)*np.exp(-b[k]*tau) - a[k]*b[k]*tau**k*np.exp(-b[k]*tau)
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        f_inf, a, b, sigma = param
        K = len(a)
        M = len(tau)
        theta = np.zeros([M])
        for m in range(0,M):
            theta[m] = -a[0]*b[0]*np.exp(-b[0]*tau[m]) + sigma**2*tau[m]
            for k in range(1,K):
                theta[m] += a[k]*k*tau[m]**(k-1)*np.exp(-b[k]*tau[m]) - a[k]*b[k]*tau[m]**k*np.exp(-b[k]*tau[m])
    return theta

##################
#### Vasicek #####
##################
def zcb_price_vasicek(r0,a,b,sigma,tau):
    # See Vasicek in OneNote
    if type(tau) == int or type(tau) == float:
        B = (1/a)*(1-np.exp(-a*tau))
        A = (B-tau)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
        p = np.exp(A-r0*B)
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        M = len(tau)
        p = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*tau[i]))
            A = (B-tau[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
            p[i] = np.exp(A-r0*B)
    else:
        print(f"tau not a recognized type")
        p = False
    return p

def spot_rate_vasicek(r0,a,b,sigma,tau):
    """ Compute term structure of spot rates."""
    # See Vasicek in OneNote
    if type(tau) == int or type(tau) == float:
        B = (1/a)*(1-np.exp(-a*tau))
        A = (B-tau)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
        if tau < 1e-6:
            r = 0
        elif tau >= 1e-6:
            r = (-A+r0*B)/tau
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        M = len(tau)
        r = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*tau[i]))
            A = (B-tau[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
            if tau[i] < 1e-6:
                r[i] = 0
            else:
                # basically just using ZCB_to_spot, R[i] = -np.log(p[i])/T[i]
                r[i] = (-A+r0*B)/tau[i]
    else:
        print(f"tau not a recognized type")
        r = False
    return r

def forward_rate_vasicek(r0,a,b,sigma,tau):
    """ Compute the term structure of instantenous forward rates. """
    # See Vasicek in OneNote
    if type(tau) == int or type(tau) == float:
        B = (1/a)*(1-np.exp(-a*tau))
        B_T = np.exp(-a*tau)
        if tau < 1e-6:
            f = 0
        elif tau >= 1e-6:
            f = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        M = len(tau)
        f = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*tau[i]))
            B_T = np.exp(-a*tau[i])
            if tau[i] < 1e-6:
                f[i] = 0
            else:
                f[i] = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    else:
        print(f"tau not a recognized type")
        f = False
    return f

def ci_vasicek(r0,a,b,sigma,T,size_ci,method = "two_sided"):
    if type(T) == int or type(T) == float:
        if method == "lower":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, np.inf
        elif method == "upper":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = -np.inf, mean + z*std
        elif method == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            if T < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, mean + z*std
        print(f"method: {method}, z: {z}")
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        lb, ub = np.zeros([N]), np.zeros([N])
        if method == "lower":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, np.inf
        elif method == "upper":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = -np.inf, mean + z*std
        elif method == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, mean + z*std
    else:
        print(f"tau not a recognized type")
        lb,ub = False, False
    return lb, ub

def fit_vasicek_obj(param,R_star,T,scaling = 1):
    """ Compute the sum of squared errors between the fitted SPOT RATES (NOT PRICES) using the Vasicek model and the actual spot rates NOT knowing sigma.
    """
    r0, a, b, sigma = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def fit_vasicek_no_sigma_obj(param,sigma,R_star,T,scaling = 1):
    """ Compute the sum of squared errors between the fitted SPOT RATES (NOT PRICES) using the Vasicek model and the actual spot rates knowing sigma.
    """
    r0, a, b = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def euro_option_price_vasicek(K,T,U,p_T,p_U,a,sigma,type = "call"):
    sigma_p = (sigma/a)*(1-np.exp(-a*(U-T)))*np.sqrt((1-np.exp(-2*a*T))/(2*a))
    d1 = (np.log(p_U/(p_T*K)))/sigma_p + 0.5*sigma_p
    d2 = d1 - sigma_p
    if type == "call":
        price = p_U*ndtr(d1) - p_T*K*ndtr(d2)
    elif type == "put":
        price = p_T*K*ndtr(-d2) - p_U*ndtr(-d1)
    return price

#############################
#### Cox-Ingersoll-Ross #####
#############################
def zcb_price_cir(r0,a,b,sigma,tau):
    if type(tau) == int or type(tau) == float or type(tau) == np.float64:
        gamma = np.sqrt(a**2+2*sigma**2)
        D = (gamma+a)*(np.exp(gamma*tau)-1)+2*gamma
        A = ((2*gamma*np.exp(0.5*tau*(a+gamma)))/D)**((2*a*b)/(sigma**2))
        B = 2*(np.exp(gamma*tau)-1)/D
        p = A*np.exp(-r0*B)
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        M = len(tau)
        p = np.zeros([M])
        for i in range(0,M):
            gamma = np.sqrt(a**2+2*sigma**2)
            D = (gamma+a)*(np.exp(gamma*tau[i])-1)+2*gamma
            A = ((2*gamma*np.exp(0.5*tau[i]*(a+gamma)))/D)**((2*a*b)/(sigma**2))
            B = 2*(np.exp(gamma*tau[i])-1)/D
            p[i] = A*np.exp(-r0*B)
    else:
        print(f"tau not a recognized type")
        p = False
    return p

def spot_rate_cir(r0,a,b,sigma,tau):
    """ Compute term structure of spot rates."""
    if type(tau) == int or type(tau) == float:
        if tau < 1e-6:
            r = 0
        else:
            gamma = np.sqrt(a**2+2*sigma**2)
            D = (gamma+a)*(np.exp(gamma*tau)-1)+2*gamma
            A = ((2*gamma*np.exp(0.5*tau*(a+gamma)))/D)**((2*a*b)/(sigma**2))
            B = 2*(np.exp(gamma*tau)-1)/D
            r = (-np.log(A)+r0*B)/(tau)
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        M = len(tau)
        r = np.zeros([M])
        for i in range(0,M):
            if tau[i] < 1e-6:
                r[i] = 0
            else:
                gamma = np.sqrt(a**2+2*sigma**2)
                D = (gamma+a)*(np.exp(gamma*tau[i])-1)+2*gamma
                A = ((2*gamma*np.exp(0.5*tau[i]*(a+gamma)))/D)**((2*a*b)/(sigma**2))
                B = 2*(np.exp(gamma*tau[i])-1)/D
                r[i] = (-np.log(A)+r0*B)/(tau[i])
    else:
        print(f"tau not a recognized type")
        r = False
    return r

def forward_rate_cir(r0,a,b,sigma,tau):
    """ Compute the term structure of instantenous forward rates. """
    if type(tau) == int or type(tau) == float:
        if tau < 1e-6:
            f = 0
        else:
            c = (2*a*b)/(sigma**2)
            gamma = np.sqrt(a**2+2*sigma**2)
            N = 2*gamma*np.exp(0.5*tau*(a+gamma))
            N_T = gamma*(gamma+a)*np.exp(0.5*tau*(a+gamma))
            D = (gamma+a)*(np.exp(gamma*tau)-1)+2*gamma
            D_T = gamma*(a+gamma)*np.exp(gamma*tau)
            M = 2*(np.exp(gamma*tau)-1)
            M_T = 2*gamma*np.exp(gamma*tau)
            f = c*(-N_T/N+D_T/D)+r0*(M_T*D-M*D_T)/D**2
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        N = len(tau)
        f = np.zeros([N])
        for i in range(0,N):
            if tau[i] < 1e-6:
                f[i] = 0
            else:
                c = (2*a*b)/(sigma**2)
                gamma = np.sqrt(a**2+2*sigma**2)
                N = 2*gamma*np.exp(0.5*tau[i]*(a+gamma))
                N_T = gamma*(gamma+a)*np.exp(0.5*tau[i]*(a+gamma))
                D = (gamma+a)*(np.exp(gamma*tau[i])-1)+2*gamma
                D_T = gamma*(a+gamma)*np.exp(gamma*tau[i])
                M = 2*(np.exp(gamma*tau[i])-1)
                M_T = 2*gamma*np.exp(gamma*tau[i])
                f[i] = c*(-N_T/N+D_T/D)+r0*(M_T*D-M*D_T)/D**2
    else:
        print(f"tau not a recognized type")
        f = False
    return f

def ci_cir(r0,a,b,sigma,tau,size_ci,method = "two_sided"):
    if type(tau) == int or type(tau) == float:
        if method == "lower":
            if tau < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*tau)*r0)/(sigma**2*(1-np.exp(-a*tau)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*tau)))
                lb, ub = ncx2.ppf(1-size_ci,df,nc)/scaling, np.inf
        elif method == "upper":
            if tau < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*tau)*r0)/(sigma**2*(1-np.exp(-a*tau)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*tau)))
                lb, ub = 0, ncx2.ppf(size_ci,df,nc)/scaling
        elif method == "two_sided":
            if tau < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*tau)*r0)/(sigma**2*(1-np.exp(-a*tau)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*tau)))
                lb, ub = ncx2.ppf((1-size_ci)/2,df,nc)/scaling, ncx2.ppf(size_ci+(1-size_ci)/2,df,nc)/scaling
    elif type(tau) == tuple or type(tau) == list or type(tau) == np.ndarray:
        N = len(tau)
        lb, ub = np.zeros([N]), np.zeros([N])
        if method == "lower":
            for i in range(0,N):
                if tau[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*tau[i])*r0)/(sigma**2*(1-np.exp(-a*tau[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*tau[i])))
                    lb[i], ub[i] = ncx2.ppf(1-size_ci,df,nc)/scaling, np.inf
        elif method == "upper":
            for i in range(0,N):
                if tau[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*tau[i])*r0)/(sigma**2*(1-np.exp(-a*tau[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*tau[i])))
                    lb[i], ub[i] = 0, ncx2.ppf(size_ci,df,nc)/scaling
        elif method == "two_sided":
            for i in range(0,N):
                if tau[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*tau[i])*r0)/(sigma**2*(1-np.exp(-a*tau[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*tau[i])))
                    lb[i], ub[i] = ncx2.ppf((1-size_ci)/2,df,nc)/scaling, ncx2.ppf(size_ci+(1-size_ci)/2,df,nc)/scaling
                    # (4*b)/(sigma**2*(1-np.exp(-b*tau[i])))*
    else:
        print(f"tau not a recognized type")
        lb,ub = False, False
    return lb, ub

def fit_cir_obj(param,R_star,T,scaling = 1):
    """ Compute the sum of squared errors between the fitted spot rates using the Vasicek model and the actual spot rates NOT knowing sigma.
    """
    r0, a, b, sigma = param
    M = len(T)
    R_fit = spot_rate_cir(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def fit_cir_no_sigma_obj(param,sigma,R_star,T,scaling = 1):
    """ Compute the sum of squared errors between the fitted spot rates using the Vasicek model and the actual spot rates knowing sigma.
    """
    r0, a, b = param
    M = len(T)
    R_fit = spot_rate_cir(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

####################################
### Simulation of the short rate ###
####################################
def short_rate_simul(r0,param,M,T,method = "vasicek"):
    """One simulation of a particular models using a Euler-scheme. """
    
    r = np.zeros([M+1])
    r[0] = r0
    delta = T/M
    if method == "vasicek":
        a, b, sigma = param
        delta_sqrt = np.sqrt(delta)
        Z = np.random.standard_normal(M)
        for m in range(1,M+1):
            r[m] = r[m-1] + (b-a*r[m-1])*delta + sigma*delta_sqrt*Z[m-1]
    elif method == "cir":
        a, b, sigma = param
        Z = np.random.standard_normal(M)
        for m in range(1,M+1):
            r[m] = r[m-1] + a*(b-r[m-1])*delta + sigma*np.sqrt(delta*r[m-1])*Z[m-1]
    elif method == "ho_lee_ns":
        f_inf, a, b, sigma = param
        Z = np.random.standard_normal(M)
        for m in range(1,M+1):
            param_theta = (f_inf, a, b, sigma)
            theta = theta_ns(param_theta,m*delta)
            r[m] = r[m-1] + theta*delta + sigma*np.sqrt(delta*r[m-1])*Z[m-1]        
    else:
        raise ValueError(f"Method '{method}' is not supported. Choose 'vasicek', 'cir', or 'ho_lee_ns'.")

    return r

####################################
### Black Scholes Option Pricing ###
####################################
def bs_price(S, K, T, r, sigma, type = "call"):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    if type == 'put':
        price = K*np.exp(-r*T)*ndtr(-d2) - S*ndtr(-d1)
    else:
        price = S*ndtr(d1)-K*np.exp(-r*T)*ndtr(d2)
    return price

def bs_vega(S, K, T, r, sigma,type = "call"):
    if type == "call":
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S*norm.pdf(d1) * np.sqrt(T)
    return vega

def bs_iv(C, S, K, T, r, iv0 = 0.2, max_iter = 200, prec = 1.0e-5):
    # Impleid Black-Scholes volatility from call option prices
    iv = iv0
    for i in range(0,max_iter):
        price = bs_price(S, K, T, r, iv, type)
        vega = bs_vega(S, K, T, r, iv)
        diff = C - price
        if abs(diff) < prec:
            return iv
        iv += diff/vega  # f(x) / f'(x)
    return iv

###############
### Caplets ###
###############
def black_caplet_price(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    if type == 'put':
        price = alpha*p*(R*ndtr(-d2) - L*ndtr(-d1))
    else:
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
    return price

def black_caplet_delta(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    if type == "call":
        # p_prev = p*(1+alpha*L)
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
        delta = alpha*p*ndtr(d1) - alpha/(1+alpha*L)*price
    return delta

def black_caplet_gamma(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    if type == "call":
        gamma = alpha*p*(norm.pdf(d1)/(L*sigma*np.sqrt(T-alpha))-(2*alpha)/((1+alpha*L)**2)*(alpha*R*ndtr(d2) + ndtr(d1)))
    return gamma

def black_caplet_vega(sigma,T,R,alpha,p,L,type = "call"):
    if type == "call":
        d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
        vega = alpha*p*L*norm.pdf(d1) * np.sqrt(T)
    return vega

def black_caplet_theta(sigma,T,r,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
    if type == "call":
        # p_prev = p*(1+alpha*L)
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
        theta = r*price - alpha*p*(sigma*L*norm.pdf(d1))/(2*np.sqrt(T))
    return theta

def black_caplet_iv(C,T,R,alpha,p,L, iv0 = 0.2, max_iter = 200, prec = 1.0e-5):
    iv = iv0
    for i in range(0,max_iter):
        price = black_caplet_price(iv,T,R,alpha,p,L,type = "call")
        vega = black_caplet_vega(iv,T,R,alpha,p,L,type = "call")
        diff = C - price
        if abs(diff) < prec:
            return iv
        iv += diff/vega
    return iv

#################
### Swaptions ###
#################
def black_swaption_price(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'put':
        price = S*(K*ndtr(-d2) - R*ndtr(d1))
    else:
        price = S*(R*ndtr(d1) - K*ndtr(d2))
    return price

def black_swaption_delta(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        delta = (S/R)*ndtr(d1) - price/R
    return delta

def black_swaption_gamma(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        gamma = (2/R**2)*price + (S/R)*(norm.pdf(d1)/(sigma*np.sqrt(T)) - 2*ndtr(d1))
    return gamma

def black_swaption_vega(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        vega = S*R*norm.pdf(d1) * np.sqrt(T)
    return vega

def black_swaption_theta(sigma,T,K,S,r,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        theta = r*price - S*R*sigma*norm.pdf(d1)/(2*np.sqrt(T))
    return theta

def black_swaption_iv(C,T,K,S,R,type = "call", iv0 = 0.2, max_iter = 1000, prec = 1.0e-10):
    iv = iv0
    for i in range(0,max_iter):
        price = black_swaption_price(iv,T,K,S,R,type = "call")
        vega = black_swaption_vega(iv,T,K,S,R,type = "call")
        diff = C - price
        if abs(diff) < prec:
            return iv
        iv += diff/vega
    return iv

def black_swaption_smm_price(sigma,K,S,R,type = "call"):
    # This function computes the black price of a swaption in the SWAP market model
    d1 = (np.log(R/K) + 0.5*sigma**2)/sigma
    d2 = (np.log(R/K) - 0.5*sigma**2)/sigma
    if type == 'put':
        price = S*(K*ndtr(-d2) - R*ndtr(d1))
    else:
        price = S*(R*ndtr(d1) - K*ndtr(d2))
    return price

##################
### SABR model ###
##################
def sigma_sabr(K,T,F_0,sigma_0,beta,upsilon,rho,type = "call"):
    if abs(F_0-K) < 1e-8:    # SABR ATM formula
        sigma = sigma_0*F_0**(beta-1)*(1+(((1-beta)**2/24)*(sigma_0**2*(F_0)**(2*beta-2)) + (rho*beta*upsilon*sigma_0/4)*(F_0)**(beta-1) + (2-3*rho**2)/24*upsilon**2)*T)
    else:
        z = (upsilon/sigma_0)*(F_0*K)**((1-beta)/2)*np.log(F_0/K)
        x = np.log((np.sqrt(1-2*rho*z+z**2) + z - rho)/(1-rho))
        D = (F_0*K)**((1-beta)/(2))*(1 + ((1-beta)**2/24)*np.log2(F_0/K) + ((1-beta)**4/1920)*np.emath.logn(4,F_0/K))
        # print(z,x,D)
        A = 1 + (((1-beta)**2/24)*sigma_0**2*(F_0*K)**(beta-1) + (rho*beta*upsilon*sigma_0/4)*(F_0*K)**((beta-1)/2) + ((2-3*rho**2)/24)*upsilon**2)*T
        sigma = (sigma_0/D)*(z/x)*A
    return sigma

def sabr_simul(F_0,sigma_0,beta,upsilon,rho,M,T):
    sigma, F = np.zeros([M+1]), np.zeros([M+1])
    sigma[0], F[0] = sigma_0, F_0
    delta = T/M
    Z = np.random.standard_normal([2,M])
    # print(np.average(Z[1,:]),np.var(Z[1,:]))
    delta_sqrt = np.sqrt(delta)
    rho_sqrt = np.sqrt(1-rho**2)
    for m in range(1,M+1):
        F[m] = F[m-1] + sigma[m-1]*F[m-1]**beta*delta_sqrt*Z[0,m-1]
        if F[m] < 0:
            F[m] = F[m-1]
        sigma[m] = sigma[m-1] + upsilon*sigma[m-1]*delta_sqrt*(rho*Z[0,m-1] + rho_sqrt*Z[1,m-1])
        if sigma[m] < 0:
            sigma[m] = sigma[m-1]
    return F, sigma

def fit_sabr_obj(param,sigma_market,K,T,R):
    sigma_0, beta, upsilon, rho = param
    N = len(sigma_market)
    sse = 0
    for n in range(0,N):
        sigma_model = sigma_sabr(K[n],T,R,sigma_0,beta,upsilon,rho,type = "call")
        sse += (sigma_market[n]-sigma_model)**2
    return sse

###########################
#### ZCB curve fitting ####
###########################
def zcb_curve_fit(data_input,interpolation_options = {"method": "linear"},scaling = 1):
    data = copy.deepcopy(data_input)
    data_known = []
    libor_data, fra_data, swap_data = [], [], []
    # Separating the data and constructing data_known from fixings
    for item in data:
        if item["instrument"] == "libor":
            libor_data.append(item)
            data_known.append({"maturity":item["maturity"],"rate":np.log(1+item["rate"]*item["maturity"])/item["maturity"]})
        elif item["instrument"] == "fra":
            fra_data.append(item)
        elif item["instrument"] == "swap":
            swap_data.append(item)
    # Adding elements to data_known based on FRAs
    I_done = False
    while I_done == False:
        for fra in fra_data:
            I_exer, known_exer = value_in_list_of_dict_returns_I_idx(fra["exercise"],data_known,"maturity")
            I_mat, known_mat = value_in_list_of_dict_returns_I_idx(fra["maturity"],data_known,"maturity")
            # FIX the last few instances!!!!
            if I_exer == True and I_mat == False:
                data_known.append({"maturity":fra["maturity"],"rate":(known_exer["rate"]*known_exer["maturity"]+np.log(1+(fra["maturity"]-fra["exercise"])*fra["rate"]))/fra["maturity"]})
                I_done = False
                break
            if I_exer == False and I_mat == True:
                pass
            if I_exer == True and I_mat == True:
                pass
            else:
                I_done = True
    T_known, T_swap, T_knot = [], [], []
    R_known = []
    # Finding T's and corresponding R's where there is some known data
    for known in data_known:
        T_known.append(known["maturity"])
        R_known.append(known["rate"])
    for swap in swap_data:
        T_knot.append(swap["maturity"])
        if swap["float_freq"] == "quarterly":
            if value_in_list_returns_I_idx(0.25,T_known)[0] == False and value_in_list_returns_I_idx(0.25,T_swap)[0] == False:
                T_swap.append(0.25)
        elif swap["float_freq"] == "semiannual":
            if value_in_list_returns_I_idx(0.5,T_known)[0] == False and value_in_list_returns_I_idx(0.5,T_swap)[0] == False:
                T_swap.append(0.5)
        elif swap["float_freq"] == "annual":
            if value_in_list_returns_I_idx(1,T_known)[0] == False and value_in_list_returns_I_idx(1,T_swap)[0] == False:
                T_swap.append(1)
        if swap["fixed_freq"] == "quarterly":
            for i in range(1,4*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.25,T_known)[0] == False and value_in_list_returns_I_idx(i*0.25,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.25,T_swap)[0] == False:
                    T_swap.append(i*0.25)
        elif swap["fixed_freq"] == "semiannual":
            for i in range(1,2*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.5,T_known)[0] == False and value_in_list_returns_I_idx(i*0.5,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.5,T_swap)[0] == False:
                    T_swap.append(i*0.5)
        elif swap["fixed_freq"] == "annual":
            for i in range(1,swap["maturity"]):
                if value_in_list_returns_I_idx(i,T_known)[0] == False and value_in_list_returns_I_idx(i*1,T_knot)[0] == False and value_in_list_returns_I_idx(i,T_swap)[0] == False:
                    T_swap.append(i)
    # Finding T_fra and T_endo
    T_endo, T_fra = [], []
    fra_data.reverse()
    for fra in fra_data:
        if value_in_list_returns_I_idx(fra["maturity"],T_known)[0] == False:
            I_fra_mat, idx_fra_mat = value_in_list_returns_I_idx(fra["maturity"],T_fra)
            I_endo_mat, idx_endo_mat = value_in_list_returns_I_idx(fra["maturity"],T_endo)
            if I_fra_mat is False and I_endo_mat is False:
                T_fra.append(fra["maturity"])
            elif I_fra_mat is True and I_endo_mat is False:
                pass
            elif I_fra_mat is False and I_endo_mat is True:
                pass
            elif I_fra_mat is True and I_endo_mat is True:
                T_fra.pop(idx_fra_mat)
        if value_in_list_returns_I_idx(fra["exercise"],T_known)[0] == False:
            I_fra_exer, idx_fra_exer = value_in_list_returns_I_idx(fra["exercise"],T_fra)
            I_endo_exer, idx_endo_exer = value_in_list_returns_I_idx(fra["exercise"],T_endo)
            if I_fra_exer is False and I_endo_exer is False:
                T_endo.append(fra["exercise"])
            elif I_fra_exer is True and I_endo_exer is False:
                T_fra.pop(idx_fra_exer)
                T_endo.append(fra["exercise"])
            elif I_fra_exer is False and I_endo_exer is True:
                pass
            elif I_fra_exer is True and I_endo_exer is True:
                T_fra.pop(idx_fra_exer)
    fra_data.reverse()
    # Fitting the swap portion of the curve
    T_swap_fit = T_known + T_swap + T_knot
    T_swap_fit.sort(), T_fra.sort(), T_endo.sort()
    R_knot_init = [None]*len(swap_data)
    for i, swap in enumerate(swap_data):
        indices = []
        if swap["fixed_freq"] == "quarterly":
            for j in range(1,4*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.25,T_swap_fit)[1])
        elif swap["fixed_freq"] == "semiannual":
            for j in range(1,2*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.5,T_swap_fit)[1])
        elif swap["fixed_freq"] == "annual":
            for j in range(1,swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j,T_swap_fit)[1])
        swap["indices"] = indices
        R_knot_init[i] = swap["rate"]
        i += 1
    args = (T_known,T_knot,T_swap_fit,R_known,swap_data,interpolation_options,1)
    result = minimize(zcb_curve_swap_fit_obj,R_knot_init,args = args,options={'disp': False})
    T_swap_curve, R_swap_curve = T_known + T_knot, R_known + list(result.x)
    T_fra_fit = T_swap_curve + T_fra + T_endo
    T_fra_fit.sort()
    R_fra_fit = interpolate(T_swap_curve,R_swap_curve,T_fra_fit,interpolation_options)
    R_fra_init = [None]*len(T_fra)
    for i in range(0,len(T_fra)):
        R_fra_init[i] = R_fra_fit[value_in_list_returns_I_idx(T_fra[i],T_fra_fit)[1]]
    args = (T_fra,T_known,T_endo,T_fra_fit,R_fra_fit,fra_data,interpolation_options,scaling)
    result = minimize(zcb_curve_fra_fit_obj,R_fra_init,args = args,options={'disp': False})
    R_fra = list(result.x)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra_fit)):
        I_fra, idx_fra = value_in_list_returns_I_idx(T_fra_fit[i],T_fra)
        if I_fra is True:
            R_fra_fit[i] = R_fra[idx_fra]
        elif I_fra is False:
            I_endo, idx_endo = value_in_list_returns_I_idx(T_fra_fit[i],T_endo)
            if I_endo is True:
                R_fra_fit[i] = R_endo[idx_endo]
    return np.array(T_fra_fit), np.array(R_fra_fit)

def zcb_curve_interpolate(T,R,interpolation_options = {"method":"linear"},resolution = 1):
    T_inter = np.array([i*(1/(resolution*12)) for i in range(0,int(T[-1])*12*resolution + 1)])
    # T_inter = np.array([i*(1/(res*12)) for i in range(0,2*12*res + 1)])
    N = len(T_inter)
    p_inter = np.ones([N])
    R_inter = np.zeros([N])
    f_inter = np.zeros([N])
    if interpolation_options["method"] == "linear":
        for n in range(1,N):
            I_known, idx = value_in_list_returns_I_idx(T_inter[n],T)
            if I_known == True:
                R_inter[n] = R[idx]
            else:
                idx_before_x, idx_after_x = idx_before_after_in_list(T_inter[n],T)
                R_inter[n] = ((T[idx_after_x]-T_inter[n])/(T[idx_after_x]-T[idx_before_x]))*R[idx_before_x] + ((T_inter[n]-T[idx_before_x])/(T[idx_after_x]-T[idx_before_x]))*R[idx_after_x]
            p_inter[n] = np.exp(-R_inter[n]*T_inter[n])
        for n in range(1,N):
            if n < N-1:
                f_inter[n] = (np.log(p_inter[n-1])-np.log(p_inter[n+1]))/(T_inter[n+1]-T_inter[n-1])
            elif n == N - 1:
                f_inter[n] = (np.log(p_inter[n-1])-np.log(p_inter[n]))/(T_inter[n]-T_inter[n-1])
    elif interpolation_options["method"] == "hermite":
        indices = []
        for item in T:
            I_idx, idx = value_in_list_returns_I_idx(item,T_inter)
            if I_idx is True:
                indices.append(value_in_list_returns_I_idx(item,T_inter)[1])
        l, r = -int((interpolation_options["degree"]+1)/2),int(interpolation_options["degree"]/2)
        for i in range(-l,-r+len(T)):
            coef = hermfit(T[i+l:i+r+1],R[i+l:i+r+1],interpolation_options["degree"])
            coef_der = hermder(coef)
            for idx in range(indices[i+l],indices[i+r]+1):
                R_inter[idx] = hermval(T_inter[idx],coef)
                f_inter[idx] = hermval(T_inter[idx],coef_der)*T_inter[idx] + R_inter[idx]
                p_inter[idx] = np.exp(-R_inter[idx]*T_inter[idx])
    return p_inter, R_inter, f_inter, T_inter

def interpolate(x,y,x_inter,interpolation_options = {"method":"linear"}):
    N = len(x_inter)
    y_inter = [None]*N
    if interpolation_options["method"] == "linear":
        for n in range(0,N):
            I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
            if I_known == True:
                y_inter[n] = y[idx]
            else:
                idx_before_x, idx_after_x = idx_before_after_in_list(x_inter[n],x)
                y_inter[n] = ((x[idx_after_x]-x_inter[n])/(x[idx_after_x]-x[idx_before_x]))*y[idx_before_x] + ((x_inter[n]-x[idx_before_x])/(x[idx_after_x]-x[idx_before_x]))*y[idx_after_x]
    elif interpolation_options["method"] == "hermite":
        indices = []
        for item in x:
            I_idx, idx = value_in_list_returns_I_idx(item,x_inter)
            if I_idx is True:
                indices.append(value_in_list_returns_I_idx(item,x_inter)[1])
        l, r = -int((interpolation_options["degree"]+1)/2),int(interpolation_options["degree"]/2)
        for i in range(-l,-r+len(x)):
            coef = hermfit(x[i+l:i+r+1],y[i+l:i+r+1],interpolation_options["degree"])
            for idx in range(indices[i+l],indices[i+r]+1):
                y_inter[idx] = hermval(x_inter[idx],coef)
    return y_inter

def swap_indices(data,T):
    # Finding the swap indices
    for item in data:
        if item["instrument"] == "swap":
            indices = []
            if item["fixed_freq"] == "quarterly":
                for i in range(1,4*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.25,T)[1])
            elif item["fixed_freq"] == "semiannual":
                for i in range(1,2*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.5,T)[1])
            elif item["fixed_freq"] == "annual":
                for i in range(1,item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i,T)[1])
            item["indices"] = indices
    return data

def R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data):
    R_fra.reverse(), T_fra.reverse()
    R_endo = [None]*len(T_endo)
    for i in range(0,len(T_fra)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_fra[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_fra[i]*T_fra[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_fra.reverse(), T_fra.reverse()
    R_endo.reverse(), T_endo.reverse()
    for i in range(0,len(T_endo)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_endo[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_endo[i]*T_endo[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_endo.reverse(), T_endo.reverse()
    return R_endo

def zcb_curve_fra_fit_obj(R_fra,T_fra,T_known,T_endo,T_all,R_all,fra_data,interpolation_options,scaling = 1):
    sse = 0
    R_fra = list(R_fra)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra)):
        if T_fra[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_fra[i],T_all)[1]] - R_fra[i])**2
    for i in range(0,len(T_endo)):
        if T_endo[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_endo[i],T_all)[1]] - R_endo[i])**2
    return sse

def zcb_curve_swap_fit_obj(R_knot,T_known,T_knot,T_all,R_known,swap_data,interpolation_options,scaling = 1):
    sse = 0
    R_knot = list(R_knot)
    R_all = interpolate(T_known + T_knot,R_known + R_knot,T_all,interpolation_options)
    p = spot_rates_to_zcb(T_all,R_all)
    for n, swap in enumerate(swap_data):
        if swap["fixed_freq"] == "quarterly":
            alpha = 0.25
        if swap["fixed_freq"] == "semiannual":
            alpha = 0.5
        if swap["fixed_freq"] == "annual":
            alpha = 1
        S_swap = 0
        for idx in swap["indices"]:
            S_swap += alpha*p[idx]
        R_swap = (1 - p[swap["indices"][-1]])/S_swap
        sse += (R_swap - swap["rate"])**2
    sse *= scaling
    return sse

def value_in_list_returns_I_idx(value,list):
    output = False, None
    for i, item in enumerate(list):
        if abs(value-item) < 1e-6:
            output = True, i
            break
    return output

def idx_before_after_in_list(value,list):
    idx_before, idx_after = None, None
    if value < list[0]:
        idx_before, idx_after = 0, 1
    elif list[-1] < value:
        idx_before, idx_after = len(list)-2, len(list) - 1
    else:
        for i in range(0,len(list)):
            if list[i] < value:
                idx_before = i
            elif list[i] > value:
                idx_after = i
                break
    return idx_before, idx_after

def value_in_list_of_dict_returns_I_idx(value,L,name):
    output = False, None
    for item in L:
        if abs(value-item[name]) < 1e-6:
            output = True, item
            break
    return output
