import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pageRankSim

# fname should be .mp4!
def hist_viz(rankmatrix, fps, fname):
    fig, ax = plt.subplots()

    xpos = np.arange(1,rankmatrix[0].shape[0]+1)
    ax.set_xticks( xpos, [ str(x) for x in xpos ]) # name them 1 - 7

    hist_ylim = np.min([ np.around(np.max(rankmatrix), decimals=1) + 0.05, 1.0 ])

    # animation
    def update_hist(row, data):
        ax.clear()
        ax.set_ylim(0, hist_ylim )
        ax.bar(xpos, data[row,:])
        for x in xpos:
            ax.text( x - 0.325, data[row,x-1] + 0.005, "%.3f" % data[row,x-1] )

    update_hist(0, rankmatrix)

    anim = animation.FuncAnimation(fig, update_hist, rankmatrix.shape[0], fargs=(rankmatrix, ) )

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps)
    anim.save(fname, writer=writer)

# %% codecell
#Example simulation:
    
    
A = np.array([  [0,1,0,0,1,1,0,0], #Defining adjacency matrix for A
                 [0,0,0,0,0,0,0,0],
                 [1,1,0,0,0,0,1,0],
                 [0,0,1,0,1,1,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,1,0,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,0,0]])
A=(A+A.T)>0 #Making sure A is symmetric (this makes it a non-directed graph)

rmat = pageRankSim.beginsim(A,50) #creating a rankmatrix with graph A, 50 animation frames

fps = 2 #setting fps

fname = "7nodedirect.mp4" #naming animation output file

hist_viz(rmat, fps, fname) #Calling visualization function to make the animation
