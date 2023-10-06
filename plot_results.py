import numpy as np
import matplotlib
import matplotlib.pyplot as plt


u = np.load('u.npy')
x = np.load('x.npy')
z = np.load('z.npy')

plt.plot(4.8 * np.ones((30, )))
plt.plot(3.6 * np.ones((30, )))
plt.plot(2.4 * np.ones((30, )))
plt.plot(1.2 * np.ones((30, )))
plt.plot(0 * np.ones((30, )))

plt.plot()


plt.xlim([0, SAFETY[1]])
plt.ylim([0, SAFETY[3]])
plt.xlabel('x')
plt.ylabel('y')
plt.legend([ps, p1, p2, p3, pe],
           ['Initial position', 'Trajectory stage 1', 'Trajectory stage 2', 'Trajectory stage 3', 'Ending position'],
           loc='lower left')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(2)

axs[0].plot(np.arange(0, kappa[1]+1), x_stage_1[0], color=STATE_1_COLOR, linewidth=2.5, linestyle='solid')
axs[1].plot(np.arange(0, kappa[1]+1), x_stage_1[1], color=STATE_1_COLOR, linewidth=2.5, linestyle='solid')

axs[0].plot(np.arange(kappa[1], kappa[2]+1), x_stage_2[0], color=STATE_2_COLOR, linewidth=2.5, linestyle='solid')
axs[1].plot(np.arange(kappa[1], kappa[2]+1), x_stage_2[1], color=STATE_2_COLOR, linewidth=2.5, linestyle='solid')

axs[0].plot(np.arange(kappa[2], kappa[3]+1), x_stage_3[0], color=STATE_3_COLOR, linewidth=2.5, linestyle='solid')
axs[1].plot(np.arange(kappa[2], kappa[3]+1), x_stage_3[1], color=STATE_3_COLOR, linewidth=2.5, linestyle='solid')

x_complete = np.concatenate((x_stage_1.T, x_stage_2.T[1:], x_stage_3.T[1:]))

for i in range(kappa[3]):
    if gamma_t.robustness(y=np.array([x_complete[i]]).T, t=0) >= 0:
        axs[0].fill(*get_coordinates((i, i + 1, 0, 12)), color=TARGET_COLOR)
        axs[1].fill(*get_coordinates((i, i + 1, 0, 12)), color=TARGET_COLOR)
    elif gamma_h.robustness(y=np.array([x_complete[i]]).T, t=0) >= 0:
        axs[0].fill(*get_coordinates((i, i + 1, 0, 12)), color=HOME_COLOR)
        axs[1].fill(*get_coordinates((i, i + 1, 0, 12)), color=HOME_COLOR)
    elif gamma_c.robustness(y=np.array([x_complete[i]]).T, t=0) >= 0:
        axs[0].fill(*get_coordinates((i, i + 1, 0, 12)), color=CHARGER_COLOR)
        axs[1].fill(*get_coordinates((i, i + 1, 0, 12)), color=CHARGER_COLOR)

axs[0].set_xlim([0, kappa[3]])
axs[0].set_ylim([SAFETY[2], SAFETY[3]-1])
axs[1].set_xlim([0, kappa[3]])
axs[1].set_ylim([SAFETY[2], SAFETY[3]-1])

axs[0].set_ylabel(r'$x$')
axs[1].set_ylabel(r'$y$')
axs[1].set_xlabel(r'$k$')

plt.show()
