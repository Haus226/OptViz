import re
from graphviz import Digraph
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

matplotlib.use('TkAgg')

def binop(op, operand_1, operand_2):
    if op == "+":
        return operand_1 + operand_2
    elif op == "-":
        return operand_1 - operand_2
    elif op == '*':
        return operand_1 * operand_2
    elif op == '/':
        return operand_1 / operand_2
    elif op == '^':
        return operand_1 ** operand_2
    
def tokenize(expression: str):
    pattern = r"\s*(-*\d*\.*\d+)|(\w+)|([-+*/^])|([(),])"
    tokens = re.findall(pattern, expression)
    return [t[0] or t[1] or t[2] or t[3] for t in tokens]

def draw_dag(root, filename, rankdir='RL'):
    assert rankdir in ['LR', 'TB', 'RL']
    nodes, edges = root.trace()

    dot = Digraph(format='png', graph_attr={'rankdir': rankdir})
    
    for n in nodes: 
        label = "{f : %.3f} | {d : %.3f}" % (n.v, n.grad)
        if n.type == "number":
            label = "{v : %.3f}" % (n.v)
        if n.type == "constant":
            label = n.info
        if n.type == "var":
            label += f" | {n.info}"        
        dot.node(name=str(id(n)), label=label, shape='record')
        if n.type == "op" or n.type == "func":
            dot.node(name=str(id(n)) + n.info, label=n.info)
            dot.edge(str(id(n)), str(id(n)) + n.info)  

    for n1, n2 in edges:
        dot.edge(str(id(n2)) + n2.info, str(id(n1)))
    dot.render(filename=filename, directory='').replace('\\', '/')
    return dot

def Animate2D(coords, f, 
                xlim, ylim,   
                title="Visualization", filename="animate2d.gif", fps=2):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=160)
    frames = list(range(len(coords)))
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    x, y = np.meshgrid(x, y)
    z = f(x=x, y=y)
    levels = np.linspace(z.min(), z.max(), 10)  
    contour = ax.contour(x, y, z, levels=levels)
    ax.set_xlabel("x", fontsize=30)
    ax.set_ylabel("y", fontsize=30)

    def update(frame):
        ax.clear()
        ax.clabel(contour, inline=True, fontsize=15)
        ax.plot(coords[:frame + 1, 0], coords[:frame + 1, 1], 
                color='red', linewidth=3, label='Path')        
        ax.scatter([coords[frame, 0]], [coords[frame, 1]], 
                color='yellow', s=200, label='Ball')

        ax.set_title(f"{title} (Iteration: {frame})", fontsize=40)


    # Check the file extension and save accordingly
    if filename.endswith(".png"):
        for idx in range(len(frames)):
            update(idx)
        plt.savefig(filename)
    elif filename.endswith(".gif"):
        anim = FuncAnimation(fig, update, frames=frames)
        anim.save(filename,
                writer='pillow',
                fps=fps,
                dpi=100,
                progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
        
def Animate3D(coords, f, xlim, ylim, title="Visualization", filename="animate3d.gif", fps=30):
    # Set up the figure and subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=16)
    
    # Create subplots - all with 3D projection
    ax_main = fig.add_subplot(221, projection='3d')  # Main 3D view
    ax_xy = fig.add_subplot(222, projection='3d')    # Top view (XY)
    ax_xz = fig.add_subplot(223, projection='3d')    # Side view (XZ)
    ax_yz = fig.add_subplot(224, projection='3d')    # Side view (YZ)
    
    # Define the grid for the surface
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = f(x=X, y=Y)  # Calculate function values for the surface
    
    # Pre-compute the path z-values for the ball
    z = f(x=coords[:, 0], y=coords[:, 1])
    
    def plot(ax, title, elev, azim, alpha, frame, reduced_axis=None):
        ax.plot_surface(X, Y, Z, cmap='jet', alpha=alpha)
        ax.plot(coords[:frame + 1, 0], coords[:frame + 1, 1], z[:frame + 1],
                    color='red', linewidth=3, label='Path')
        ax.scatter([coords[frame, 0]], [coords[frame, 1]], [z[frame]],
                    color='yellow', s=200, label='Ball')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        match reduced_axis:
            case "x":
                ax.set_xticks([])
            case "y":
                ax.set_yticks([])
            case "z":
                ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)  # Rotating view

    def update(frame):
        ax_main.cla()
        ax_xy.cla()
        ax_xz.cla()
        ax_yz.cla()
        
        plot(ax_main, "Main 3D View", 30, frame * 0.5, 0.7, frame)
        plot(ax_xy, "Top View (XY)", 90, -90, 0.7, frame, "z")
        plot(ax_yz, "Side View (YZ)", 0, 0, 0.3, frame, "x")
        plot(ax_xz, "Side View (XZ)", 0, 90, 0.3, frame, "y")       
        return []
    
    anim = FuncAnimation(
        fig,
        update,
        frames=len(coords),
        interval=50,
        blit=False  # Blitting doesn't work well with 3D plots
    )
    
    # Save the animation as a GIF
    anim.save(filename, writer='pillow', fps=fps, dpi=100,
              progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
    plt.close(fig)  # Close the figure to avoid display in Jupyter environments
