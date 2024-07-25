import numpy as np
from scipy.optimize import lsq_linear  

# low pass filter to make plots smoother 
def windowed_average(input_array, window_size = 15):
    output_array = np.convolve(input_array, np.ones(window_size)/window_size, mode='valid')
    return np.append(input_array[0], output_array) # append used to match input and output size

# fit curve to O(T^{2/3})
def fit_regret_curve(T_horizon_list, dpop_regret, start_index = 0):
    # create T^{2/3} for given values of T
    X = np.ones([T_horizon_list.shape[0], 2])
    X[:,1] = T_horizon_list**(2/3)

    # fit given regret to O(T^{2/3})
    regret_fit_dpop = lsq_linear(X[start_index:,:], dpop_regret[start_index:], bounds=([-np.inf,0], [np.inf,np.inf]))
    theoretical_dpop_regret = X@regret_fit_dpop.x

    # print co efficients and return
    print('Fit co-effs [1 T^{2/3}] = ' + np.array2string(regret_fit_dpop.x, precision=3, suppress_small=True))
    return theoretical_dpop_regret

# function to plot regret curve
def plot_regret_curve(ax, arrival_rate, noise_variance_list, show_theoretical, sweep_results_folder, plot_style, show_ylabel, label_font_size):
    # iterate for given values of noise variances
    for jj, noise_variance in enumerate(noise_variance_list):
        current_result = np.load(sweep_results_folder + '/regret-lambda-' + str(arrival_rate).replace('.','_') + '-var-' + str(noise_variance).replace('.','_') + '.npy')     
        ax.plot(current_result[0,:], current_result[1,:] - current_result[2,:], plot_style[jj], label = '$\sigma^2$ = ' + str(noise_variance), fillstyle = 'none', markeredgewidth=2, ms=8)  

        if(show_theoretical[jj]):
            theoretical_regret = fit_regret_curve(current_result[0,:], current_result[1,:] - current_result[2,:], start_index = 6)
            ax.plot(current_result[0,:], theoretical_regret, '--', label = '$O(T^{2/3})$', linewidth=3)

    if(show_ylabel): ax.set_ylabel('Regret')
    ax.set_xlabel('Time horizon')
    
    ax.set_ylim([0,8000])
    ax.set_xlim([1000,20000])

    # show values in scientific notation and show exponent near axes
    ax.set_xticks(ticks=2000*np.arange(1,11), labels=['{:1.0f}'.format(s) for s in 2*np.arange(1,11)])
    ax.text(18500, -1125, '$\\times 10^3$', fontdict=None, size=label_font_size)
    
    # show values in scientific notation and show exponent near axes
    ax.set_yticks(ticks=1000*np.arange(0,9), labels=['{:1.0f}'.format(s) for s in np.arange(0,9)])
    ax.text(960, 8075, '$\\times 10^3$', fontdict=None, size=label_font_size)
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2,3,4]
    order.reverse()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left')
    
    ax.grid()

# function to plot total queue backlogs
def plot_backlog_curve(ax, unknownT_backlog_at_tt, knownT_backlog_at_tt, oracle_backlog_at_tt, label_font_size):
    ax.plot(unknownT_backlog_at_tt, color = 'C0', label = 'DPOP (doubling)')
    ax.plot(knownT_backlog_at_tt, color = 'C1', label = 'DPOP (given T)')
    ax.plot(oracle_backlog_at_tt, color = 'C2', label = 'Oracle policy')

    # show values in scientific notation and show exponent near axes
    ax.set_xticks(ticks=1000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])
    ax.text(9100, -7.9, '$\\times 10^3$', fontdict=None, size=label_font_size)

    ax.set_xlim([-250,10000])
    ax.set_ylim([0,65])

    ax.set_xlabel('Time-slot')
    ax.set_ylabel('Total queue backlog')

    ax.legend(loc = 'lower right')
    ax.grid()

# function to plot total transmission costs
def plot_transmission_cost_curve(ax, unknownT_tran_cost_at_tt, knownT_tran_cost_at_tt, oracle_tran_cost_at_tt, label_font_size):
    ax.plot(windowed_average(oracle_tran_cost_at_tt), color = 'C2', label = 'Oracle policy')
    ax.plot(windowed_average(knownT_tran_cost_at_tt), color = 'C1', label = 'DPOP (given T)')
    ax.plot(windowed_average(unknownT_tran_cost_at_tt), color = 'C0', label = 'DPOP (doubling)')

    # show values in scientific notation and show exponent near axes
    ax.set_xticks(ticks=1000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])
    ax.text(9100, -0.61, '$\\times 10^3$', fontdict=None, size=label_font_size)

    ax.set_xlim([-250,10000])
    ax.set_ylim([0,5])

    ax.set_xlabel('Time-slot')
    ax.set_ylabel('Instantaneous transmission cost')

    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2]
    order.reverse()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'lower right')

    ax.grid()