import numpy as np
import fixed_income_derivatives as fid
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint

# Problem 1
def fit_vasicek_obj(param,p_star,T):
    r0, a, b, sigma = param
    N = len(p_star)
    p_fit = fid.zcb_price_vasicek(r0,a,b,sigma,T)
    y = 0
    for i in range(0,N):
        y += 1e16*(p_fit[i] - p_star[i])**2
    return y

# Generating the data for the minimization
N = 100
T = np.array([0.1*i for i in range(0,N+1)])
M = len(T)
r0_star, a_star, b_star, sigma_star = 0.025, 2, 0.1, 0.02
p_star = fid.zcb_price_vasicek(r0_star,a_star,b_star,sigma_star,T)

# print(f"Unconstrained optimization")
param_0 = (0.03, 1.8, 0.12, 0.03)
y = fit_vasicek_obj(param_0,p_star,T)
result = minimize(fit_vasicek_obj,param_0,method = 'nelder-mead',args = (p_star,T),options={'xatol': 1e-8,'disp': True})
r0_hat, a_hat, b_hat, sigma_hat, fct_value = result.x[0], result.x[1], result.x[2], result.x[3], result.x
print(f"r0_hat: {r0_hat}, a_hat: {a_hat}, b_hat: {b_hat}, sigma_hat: {sigma_hat}, opt: {result.x}")
p_fit = fid.zcb_price_vasicek(r0_hat,a_hat,b_hat,sigma_hat,T)
r_star, r_fit = np.zeros([N]), np.zeros([N])
for i in range(0,N):
    r_star[i] = - np.log(p_star[i+1])/(T[i+1])
    r_fit[i] = - np.log(p_fit[i+1])/(T[i+1])

# print(f"Bounded optimization")
param_0 = (0.03, 1.8, 0.12, 0.03)
bounds = Bounds([0.,0,0,0],[0.2,1.8,0.08,0.1])
result = minimize(fit_vasicek_obj,param_0,method = 'trust-constr',bounds = bounds,args = (p_star,T),options={'disp': True})
r0_hat, a_hat, b_hat, sigma_hat, fct_value = result.x[0], result.x[1], result.x[2], result.x[3], result.x
print(f"r0_hat: {r0_hat}, a_hat: {a_hat}, b_hat: {b_hat}, sigma_hat: {sigma_hat}, opt: {result.x}")
p_fit = fid.zcb_price_vasicek(r0_hat,a_hat,b_hat,sigma_hat,T)
r_star, r_fit = np.zeros([N]), np.zeros([N])
for i in range(0,N):
    r_star[i] = - np.log(p_star[i+1])/(T[i+1])
    r_fit[i] = - np.log(p_fit[i+1])/(T[i+1])

# print(f"optimization with non-linear constraint")

def cons_f(param):
    return [2*param[1]*param[2] - param[3]**2]

def cons_J(param):
    return [[0,2*param[2],2*param[1],-2*param[3]]]

def cons_H(param,v):
    return v[0]*np.array([[0,0,0,0],[0,0,2,0],[0,2,0,0],[0,0,0,-2]])

param_0 = (0.03, 1.8, 0.12, 0.015)
bounds = Bounds([0,0,0,0],[0.1,1.8,0.08,0.1])  # bounds = [(0,0.05),(0,3),(0,0.2),(0,0.05)]
nonlinear_constraint = NonlinearConstraint(cons_f, 0, np.inf, jac=cons_J, hess=cons_H)
result = minimize(fit_vasicek_obj,param_0,method = 'trust-constr',args = (p_star,T),bounds = bounds,constraints=[nonlinear_constraint],options={'disp': True})
r0_hat, a_hat, b_hat, sigma_hat, fct_value = result.x[0], result.x[1], result.x[2], result.x[3], result.x
print(f"r0_hat: {r0_hat}, a_hat: {a_hat}, b_hat: {b_hat}, sigma_hat: {sigma_hat}, opt: {result.x}")
p_fit = fid.zcb_price_vasicek(r0_hat,a_hat,b_hat,sigma_hat,T)
r_star, r_fit = np.zeros([N]), np.zeros([N])
for i in range(0,N):
    r_star[i] = - np.log(p_star[i+1])/(T[i+1])
    r_fit[i] = - np.log(p_fit[i+1])/(T[i+1])

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"Empirical and fitted ZCB prices", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.2,xticks[-1]+0.2])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.02,0.04,0.06,0.08,0.1])
ax.set_yticklabels([0,0.02,0.04,0.06,0.08,0.1],fontsize = 6)
ax.set_ylim([0,0.105])
ax.set_ylabel(f"ZCB prices",fontsize = 6)
p1 = ax.scatter(T[1:], r_star, s = 1, color = 'black', marker = ".",label="r_star")
p2 = ax.scatter(T[1:], r_fit, s = 1, color = 'red', marker = ".",label="r_fit")

plots = [p1,p2]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 6)

plt.show()

# Problem 2
def fit_forward_rate_ns_obj(param,f_star,T,scaling):
    K = int((len(param)-1)/2 - 1)
    f_inf, a, b = param[0], param[1:1+K+1], param[K+2:K+2+K+1]
    param_fit = f_inf, a, b
    M = len(T)
    f_fit = fid.f_ns(param_fit,T)
    y = 0
    for m in range(0,M):
        y += scaling*(f_fit[m] - f_star[m])**2
    return y

def fit_forward_rate_ns_jac(param,f_star,T,scaling):
    N = int((len(param)-1)/2 - 1)
    param_fit = param[0], param[1:1+N+1], param[N+2:N+2+N+1]
    M = len(T)
    f_fit = fid.f_ns(param_fit,T)
    jac = np.zeros([len(param)])
    for m in range(0,M):
        # f_deriv = fid.f_ns_jac(param_fit,T[m])
        jac += scaling*2*(f_fit[m] - f_star[m])*fid.f_ns_jac(param_fit,T[m])
    return jac

def fit_forward_rate_ns_hess(param,f_star,T,scaling):
    N = int((len(param)-1)/2 - 1)
    param_fit = param[0], param[1:1+N+1], param[N+2:N+2+N+1]
    M = len(T)
    f_fit = fid.f_ns(param_fit,T)
    hess = np.zeros([len(param),len(param)])
    for m in range(0,M):
        # f_deriv = fid.f_ns_jac(param_fit,T[m])
        f_deriv = fid.f_ns_jac(param_fit,T[m])
        hess += scaling*2*(f_fit[m] - f_star[m])*fid.f_ns_hess(param_fit,T[m]) + 2*np.outer(f_deriv,f_deriv)
    return hess

M = 101
T = np.array([0.1*i for i in range(0,M)])

# # Parameters used to generate the data
# f_inf_star, a_star, b_star, sigma_star = 0.04, [-0.02,0.01,0.005], [0.5,0.2,0.8], 0.02
# param = f_inf_star, a_star, b_star
# f_star_data = fid.f_ns(param,T)

f_star = np.array([0.02, 0.02200177, 0.02399526, 0.02596512, 0.02789877,
0.02978607, 0.03161897, 0.03339121, 0.03509809, 0.03673622, 0.03830334,
0.03979815, 0.04122013, 0.04256945, 0.04384681, 0.04505338, 0.04619068, 0.04726055,
0.04826501, 0.0492063, 0.05008674, 0.05090877, 0.05167487, 0.05238751,
0.0530492, 0.0536624, 0.05422954, 0.054753, 0.05523509, 0.05567805,
0.05608405, 0.05645518, 0.05679343, 0.05710072, 0.05737887, 0.05762962,
0.05785464, 0.05805548, 0.05823363, 0.05839051, 0.05852743, 0.05864565,
0.05874636, 0.05883067, 0.05889961, 0.05895418, 0.05899529, 0.05902381,
0.05904056, 0.0590463, 0.05904173, 0.05902752, 0.05900431, 0.05897268,
0.05893317, 0.0588863, 0.05883255, 0.05877236, 0.05870616, 0.05863434,
0.05855727, 0.05847529, 0.05838872, 0.05829788, 0.05820303, 0.05810446,
0.0580024, 0.05789709, 0.05778876, 0.0576776, 0.05756382, 0.05744759,
0.05732909, 0.05720849, 0.05708593, 0.05696156, 0.05683551, 0.05670793,
0.05657892, 0.05644861, 0.05631711, 0.05618451, 0.05605093, 0.05591645,
0.05578117, 0.05564517, 0.05550853, 0.05537132, 0.05523364, 0.05509554,
0.05495709, 0.05481836, 0.05467941, 0.05454029, 0.05440108, 0.05426181,
0.05412255, 0.05398334, 0.05384423, 0.05370527, 0.0535665])

# Initial parameter values
f_inf_0 = (0.03,)
a_0 = -0.05, 0.03, 0.001
b_0 = 0.35, 0.35, 0.7
scaling = 10e12

K = len(a_0) - 1
param_0 = f_inf_0 + a_0 + b_0
# print(f"Unconstrained optimization")
result = minimize(fit_forward_rate_ns_obj,param_0,method = 'nelder-mead',args = (f_star,T,scaling),options={'xatol': 1e-8,'disp': True})
f_inf_hat, a_hat, b_hat, fct_value = result.x[0], result.x[1:1+K+1], result.x[K+2:K+2+K+1], result.fun
print(f"f_inf_hat: {f_inf_hat}, a_hat: {a_hat}, b_hat: {b_hat}, opt: {result.fun}")
f_fit = fid.f_ns((f_inf_hat,a_hat,b_hat),T)
print(f"r0: {f_fit[0]}")

# # print(f"Optimization using both the Jacobian and the Hessian")
# result = minimize(fit_forward_rate_ns_obj, param_0, method='Newton-CG',args = (f_star,T,scaling),jac = fit_forward_rate_ns_jac, hess=fit_forward_rate_ns_hess,options={'maxiter': 5000, 'xtol': 1e-12, 'disp': False})
# f_inf_hat, a_hat, b_hat, fct_value = result.x[0], result.x[1:1+K+1], result.x[K+2:K+2+K+1], result.fun
# print(f"f_inf_hat: {f_inf_hat}, a_hat: {a_hat}, b_hat: {b_hat}, opt: {result.fun}")
# f_fit = fid.f_ns((f_inf_hat,a_hat,b_hat),T)
# print(f"ro: {f_fit[0]}")

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"Empirical and fitted forward rates", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.2,xticks[-1]+0.2])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.02,0.04,0.06,0.08,0.1])
ax.set_yticklabels([0,0.02,0.04,0.06,0.08,0.1],fontsize = 6)
ax.set_ylim([0,0.105])
ax.set_ylabel(f"forward rates",fontsize = 6)
p1 = ax.scatter(T, f_star, s = 1, color = 'black', marker = ".",label="f_star")
p2 = ax.scatter(T, f_fit, s = 1, color = 'red', marker = ".",label="f_fit")

plots = [p1,p2]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 6)

plt.show()

# Problem 3
f_inf_hat = 0.0470857723425845
r0 = 0.01946854287371834
a_hat = [-0.02761198, 0.01251315, 0.00146974]
b_hat = [0.38773133, 0.28801503, 0.68312069]
sigma = 0.03
N = 1000
M, T = 1000, 1

param = f_inf_hat, a_hat, b_hat, sigma
chi = 0
for n in range(0,N):
    r = fid.short_rate_simul(r0,param,M,T,method = "ho_lee_ns")
    chi += np.exp(-sum(r)/N)*(sum(r)/(M+1),0)  
pi = chi/N

print(f"Price of the Asian derivative: {pi}")
