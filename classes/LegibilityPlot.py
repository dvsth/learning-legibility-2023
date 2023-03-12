import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class LegibilityPlot:
    def __init__(self):
        fe = fm.FontEntry(fname='unifont.ttf', name='unifont')
        fm.fontManager.ttflist.append(fe)

    @staticmethod
    def gradient_crim_to_darkg(mix):
        c1 = '#3B724B'
        c2 = '#9A2C44'
        # increase saturation
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((mix)*c1 + (1-mix)*c2)

    def plot(self, scores, perturbations):
        # convert the raw scores to probabilities
        scores = [1 / (1 + np.exp(-x)) for x in scores]
        fig, ax = plt.subplots(1, 1, figsize=(7, 10), dpi=600)

        # add a horizontal bar for each probability, showing the difference between the probability and 0.5
        # center the bar at 0.5
        # make the bar 0.25 high
        # make the bar color a gradient from crimson to darkgreen
        # with opacity proportional to the probability
        ax.barh(range(len(scores)), [x-0.5 for x in scores], height=0.1, color=[self.gradient_crim_to_darkg(x) for x in scores], left=0.5, alpha=0.9)

        ax.scatter(scores, range(len(scores)), s=40, color='white', marker='v', alpha=1.0, zorder=10000, edgecolors='black', linewidths=0.5)
        # place the image under the point
        for i in range(len(perturbations)):
            # place the image to the right of the point
            ax.text(scores[i], i-0.25, perturbations[i], horizontalalignment='center', verticalalignment='top', fontsize=26, fontfamily='unifont', bbox=dict(facecolor=self.gradient_crim_to_darkg(scores[i]), edgecolor='white', alpha=0.3, boxstyle='round,pad=0.3'))
            
        ax.set_xlim(0, 1)
        # make ticks from 0 to 1 spaced by 0.1
        ax.set_xticks(np.arange(0, 1.1, 0.20))
        ax.set_ylim(-1, len(scores)-0.5)
        ax.set_yticks([])
        # show a line at 0.5
        ax.axvline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
        # disable grid lines
        ax.grid(False)
        # make the background white
        ax.set_facecolor('white')
        # show the x axis line
        ax.spines['bottom'].set_visible(True)
        # make the x axis line black with 0.5 alpha
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_alpha(0.5)
        # add x axis label
        ax.set_xlabel('Legibility Score', fontsize=22)
        # axis ticklabel font size
        ax.tick_params(axis='x', labelsize=18)
        # aspect ratio
        ax.set_aspect(0.25)
        # x tick marks
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        # embolden the x axis label
        ax.xaxis.label.set_fontweight('bold')
        # add some padding above the x axis label
        # ax.xaxis.labelpad = 15
        
        # remove the left and right spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # remove the top spine
        ax.spines['top'].set_visible(False)
        
        return fig