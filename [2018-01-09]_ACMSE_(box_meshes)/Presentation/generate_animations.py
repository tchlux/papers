from util.plotly import Plot
import numpy as np
from scipy.spatial import Voronoi

# help(Plot)
# exit()

# Generate an interesting random set of points
points = np.array([
    [0.48, 0.52],
    [0.40, 0.24],
    [0.65, 0.93],
    [0.16, 0.56],
    [0.93, 0.68],
    [0.70, 0.16],
])

# ============================
#      Iterative Box Mesh     
# ============================

p = Plot()
# Define an additive matrix for shifting the corners of boxes outwards
shift = np.array([
    [-1, 1],
    [ 1, 1],
    [ 1,-1],
    [-1,-1],
], dtype=float)
# Shifts per box
shift_0 = shift.copy()
shift_5 = shift.copy()
shift_1 = shift.copy()
# Corners of the box
box_0 = np.ones((4,2)) * points[0] + (.05 * shift)
box_5 = np.ones((4,2)) * points[5] + (.05 * shift)
box_1 = np.ones((4,2)) * points[1] + (.05 * shift)
# Multiplier (for stretching frames)
m = 1

# Add all points to the frame "f" of plot "p"
def add_all_points(f, faded=.2):
    for i,pt in enumerate(points):
        if (i == 0):
            a = 1 if (0 <= f) else faded
            p.add_node(f"pt-{i}", *pt, white=False,
                       color=p.color(i,alpha=a),
                       marker_line_color=p.color(color="rgb(0,0,0)",alpha=a),
                       size=10, frame=f)
        elif (i == 5):
            a = 1 if (2 <= f) else faded
            p.add_node(f"pt-{i}", *pt, white=False,
                       color=p.color(i,alpha=a), 
                       marker_line_color=p.color(color="rgb(0,0,0)",alpha=a),
                       size=10, frame=f)
        elif (i == 1):
            a = 1 if (5 <= f) else faded
            p.add_node(f"pt-{i}", *pt, white=False,
                       color=p.color(i,alpha=a), 
                       marker_line_color=p.color(color="rgb(0,0,0)",alpha=a),
                       size=10, frame=f)
        else:
            p.add_node(f"pt-{i}", *pt, white=False,
                       color=p.color(i,alpha=faded), 
                       marker_line_color=p.color(color="rgb(0,0,0)",alpha=faded),
                       size=10, frame=f)

# Manage the box around p0 through the animations
def add_p0_box(f, n=0):
    if ((f/m) < 0):
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        fill_color = p.color(0, alpha=0)
        edge = []
        # Add the nodes for the corners
        for i,pt in enumerate(box_0):
            p.add_node(f"p0-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p0-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=0),
                   fill_color=fill_color, frame=f)
    if (0 <= (f/m) < 1):
        p.add_node(f"p{n}-highlighter", *points[n], frame=f)
        fill_color = p.color(0, alpha=0)
        edge = []
        # Add the nodes for the corners
        for i,pt in enumerate(box_0):
            p.add_node(f"p0-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p0-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=0),
                   fill_color=fill_color, frame=f)
    elif (1 <= (f/m) < 4):
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        fill_color = p.color(0, alpha=.1)
        edge = []
        # Add the nodes for the corners
        for i,pt in enumerate(points[0]+shift_0):
            p.add_node(f"p0-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p0-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)        
    elif (4 <= (f/m) < 8):
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        fill_color = p.color(0, alpha=.1)
        edge = []
        # Set the lower boundary
        shift_0[2,1] = points[5,1] - points[0,1]
        shift_0[3,1] = points[5,1] - points[0,1]
        # Add the nodes for the corners
        for i,pt in enumerate(points[0]+shift_0):
            p.add_node(f"p0-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p0-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)        
    elif (8 <= (f/m)):
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        fill_color = p.color(0, alpha=.1)
        edge = []
        # Set the lower boundary
        shift_0[2,1] = -abs(points[1,1] - points[0,1])
        shift_0[3,1] = -abs(points[1,1] - points[0,1])
        # Add the nodes for the corners
        for i,pt in enumerate(points[0]+shift_0):
            p.add_node(f"p0-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p0-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)        


def add_p5_box(f, n=5):
    if ((f/m) < 3):
        # Add a highlighter before the box is built
        if (2 <= (f/m)): 
            p.add_node(f"p{n}-highlighter", *points[n], frame=f)
        else:
            p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=0)
        edge = []
        # Add the nodes for the corners
        for i,pt in enumerate(globals()[f"box_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=0),
                   fill_color=fill_color, frame=f)
    elif (3 <= (f/m) < 4):
        # Add a highlighter before the box is built
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=.1)
        edge = []
        # Add the nodes for the corners
        for i,pt in enumerate(points[n]+globals()[f"shift_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)        
    elif (4 <= (f/m) < 7):
        # Add a highlighter before the box is built
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=.1)
        edge = []
        # Set the lower boundary
        globals()[f"shift_{n}"][0,1] = points[0,1] - points[n,1]
        globals()[f"shift_{n}"][1,1] = points[0,1] - points[n,1]
        # Add the nodes for the corners
        for i,pt in enumerate(points[n]+globals()[f"shift_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)        
    elif (7 <= (f/m)):
        # Add a highlighter before the box is built
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=.1)
        edge = []
        # Set the left boundary
        globals()[f"shift_{n}"][0,0] = points[1,0] - points[n,0]
        globals()[f"shift_{n}"][3,0] = points[1,0] - points[n,0]
        # Add the nodes for the corners
        for i,pt in enumerate(points[n]+globals()[f"shift_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)


# handle the animation of the box around p2
def add_p1_box(f, n=1):
    if ((f/m) < 6):
        # Add a highlighter before the box is built
        if (5 <= (f/m)): 
            p.add_node(f"p{n}-highlighter", *points[n], frame=f)
        else:
            p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=0)
        edge = []
        # Add the nodes for the corners
        for i,pt in enumerate(globals()[f"box_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=0),
                   fill_color=fill_color, frame=f)
    elif (6 <= (f/m) < 7):
        # Add a highlighter before the box is built
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=.1)
        edge = []
        # Add the nodes for the corners
        for i,pt in enumerate(points[n]+globals()[f"shift_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)        
    elif (7 <= (f/m) < 8):
        # Add a highlighter before the box is built
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=.1)
        edge = []
        # Set the lower boundary
        globals()[f"shift_{n}"][1,0] = points[5,0] - points[n,0]
        globals()[f"shift_{n}"][2,0] = points[5,0] - points[n,0]
        # Add the nodes for the corners
        for i,pt in enumerate(points[n]+globals()[f"shift_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)        
    elif (8 <= (f/m)):
        # Add a highlighter before the box is built
        p.add_node(f"p{n}-highlighter", *points[n], display=False, frame=f)
        # Add the box
        fill_color = p.color(n, alpha=.1)
        edge = []
        # Set the left boundary
        globals()[f"shift_{n}"][0,1] = points[0,1] - points[n,1]
        globals()[f"shift_{n}"][1,1] = points[0,1] - points[n,1]
        # Add the nodes for the corners
        for i,pt in enumerate(points[n]+globals()[f"shift_{n}"]):
            p.add_node(f"p{n}-corner-{i}", *pt, display=False, frame=f)
            edge.append(f"p{n}-corner-{i}")
        # Finish the loop and add the background fill for the box
        edge += [edge[0]]
        p.add_edge(edge, line_color=p.color(n,alpha=1),
                   fill_color=fill_color, frame=f)



# 0) Just points
# 1) Box around p0
# 2) Point at p5
# 3) Box at p5 (big)
# 4) Shrinking p0 box and p5 box
# 5) Point at p1
# 6) Box at p1 (big)
# 7) Shrinking p1+p5
# 8) Srhining p1+p0
for f in range(-1,9):
    add_p1_box(f)
    add_p5_box(f)
    add_p0_box(f)
    add_all_points(f)


# Generate the plot
size = 700
p.graph(width=1.2*size, height=size, x_range=[0,1.2], y_range=[0,1],
        show_slider_labels=False, data_easing=True, bounce=False,
        loop_duration=10, loop_animation=True,
        file_name="IBM_Construction.html")
exit()



# ======================
#      Max Box Mesh     
# ======================

p = Plot()

shift = np.array([
    [-1, 1],
    [ 1, 1],
    [ 1,-1],
    [-1,-1],
], dtype=float)
box = np.ones((4,2)) * points[0] + (.05 * shift)


# Add all points to the frame "f" of plot "p"
def add_all_points(f):
    for i,pt in enumerate(points):
        p.add_node(f"pt-{i}", *pt, white=False, color=p.color(i), size=10, frame=f)

# Add the box corners
def add_box(f):
    display = f > 1
    fill_color = "rgba(0,0,0,.05)" if display else "rgba(0,0,0,0)"
    edge = []
    # Add the nodes for the corners
    for i,pt in enumerate(box):
        p.add_node(f"corner-{i}", *pt, color=p.color(0), size=5,
                   display=display, white=False, frame=f)
        edge.append(f"corner-{i}")
    # Finish the loop and add the background fill for the box
    edge += [edge[0]]
    p.add_edge(edge, line_color="rgba(0,0,0,0)", 
               fill_color=fill_color, frame=f)
    # Add solid line edges for sides of the box that are complete
    for i in range(len(edge)-1):
        first = edge[:-1][i-1]
        second = edge[i]
        if (shift[i][ shift[i]==shift[i-1] ][0] == 0):
            line_color = "rgba(0,0,0,1)"
        else:
            line_color = "rgba(0,0,0,0)"
        # Add the edges
        p.add_edge([first,second], line_color=line_color, frame=f)


intersects = [ 2, 3, 1, 4]
dimensions = [ 1, 0, 1, 0]
to_stop    = [[2,3], [0,3], [0,1], [1,2]]

for step in range(0, 13):
    f = step // 3 + 1
    add_box(step)
    add_all_points(step)
    if (step > 0) and (f <= 4):
        # Extract the info about growth at this step
        i = intersects[f-1]
        d = dimensions[f-1]
        t = to_stop[f-1]
        # Shift the box to meet that point
        box += shift*(abs(points[i,d] - box[t[0],d]))
        # Stop growth along that dimension
        shift[t[0],d] = 0
        shift[t[1],d] = 0

# Generate the plot
size = 700
p.graph(width=1.2*size, height=size, x_range=[0,1.2], y_range=[0,1],
        show_slider_labels=False, data_easing=True, bounce=False,
        loop_duration=10, loop_animation=True,
        file_name="MBM_Construction.html")
exit()


# ======================
#      Voronoi Mesh     
# ======================

p = Plot()
# Add the voronoi region around the main point
vor = Voronoi(points)
# Get the center
center = points[0]
# Get the boundary of the center (in terms of outwards vector)
boundary = np.array([vor.vertices[v] for v in vor.regions[3]]) - center
# Set an interpolation point
interp_pt = np.array([.2, .42])
# Calculate the projection of the interpolation point
projection = (points[3] - points[0])
project_ratio = (
    (np.dot(interp_pt,projection) - np.dot(points[0],projection)) /
    (np.dot(points[3],projection) - np.dot(points[0],projection))
)
interp_project = (project_ratio * (points[3]-points[0])) + points[0]
interp_bound = ((interp_pt - points[0]) / project_ratio) + points[0]
     
# Function for adding all voronoi cells to a frame of Plot "p"
def add_all_cells(f):
    # Add the background voronoi cells web
    vor_cell_nodes = []
    for pair in vor.ridge_points:
        if (0 in pair): continue
        # Get the mid point and the corresponding voronoi vertex
        mid_pt = (points[pair[0]] + points[pair[1]]) / 2
        v_pt = vor.vertices[vor.ridge_dict[tuple(pair)]][1]
        # Shift the mid point far away so that it is off the graph
        vector = mid_pt - v_pt
        if np.dot(vector,mid_pt - points[0]) < 0: vector *= -1
        mid_pt += 20*vector
        # Add the nodes and edge between the edge cells
        node_1 = f"cell-{len(vor_cell_nodes)}"
        node_2 = f"boundary-{len(vor_cell_nodes)}"
        p.add_node(node_1, *v_pt, display=False, frame=f)
        p.add_node(node_2, *mid_pt, display=False, frame=f)
        p.add_edge([node_1, node_2], line_width=1, dash="dash", frame=f)
        # Add the voronoi vertex to the primary cell edge definition.
        vor_cell_nodes += [node_1]
    # Add the trace around the central voronoi cell
    p.add_edge(vor_cell_nodes+[vor_cell_nodes[0]], line_width=1,
               dash="dash", frame=f)

# Add all points to the frame "f" of plot "p"
def add_all_points(f):
    for i,pt in enumerate(points):
        p.add_node(f"pt-{i}", *pt, white=False, color=p.color(i), size=10, frame=f)

# Add the growing voronoi cell based on the constant multiplier "c" to Plot "p"
def add_growing_cell(f):
    # Set the 'c' value
    c = 2 if f >= 2*3 else 1
    # Add the boundary of the interesting cell to the frame
    support = center + c*boundary
    nodes = []
    for i,pt in enumerate(support):
        nodes += [f"{i}"]
        p.add_node(f"{i}", *(pt), display=False, frame=f)
    nodes += [nodes[0]]
    p.add_edge(nodes, fill_color="rgba(0,0,0,.05)", frame=f)

# Add the interpolation point
def add_interp_point(f):
    display = f >= 3*3
    p.add_node("interp", *(interp_pt), display=display,
               marker_line_color="rgba(255,50,50,1)", 
               frame=f, size=10, symbol="x")

# Add the line that the interpolation point will project onto
def add_projection_line(f):
    display = f >= 4*3
    if display: color = "rgba(0,0,0,.8)"
    else:       color = "rgba(0,0,0,0)"
    # Add node for the projected interpolation point
    p.add_node("interp-project", *interp_project, display=display,
               size=5, frame=f)
    # Add edges for projection and projected point
    p.add_edge(["interp", "interp-project"], line_color=color,
               dash="dot", frame=f)
    p.add_edge(["pt-0", "pt-3"], line_color=color, dash="dot",
               frame=f)

# Add the line that the interpolation point will project onto
def add_bound_line(f):
    display = f >= 5*3
    if display: color = "rgba(0,0,0,.8)"
    else:       color = "rgba(0,0,0,0)"
    # Add node for the projected interpolation point's bound
    p.add_node("interp-bound", *interp_bound, display=display, size=5, frame=f)
    # Add line from center to bound
    p.add_edge(["pt-0", "interp-bound"], line_color=color, dash="dot", frame=f)

# 1) initial setup
# 2) grown voronoi cell
# 3) introduced intopolation point
# 4) introduced projection line
# 5) show ratio by coloring parts
for f in range(1,19):
    add_all_cells(f)
    add_growing_cell(f)
    add_all_points(f)
    add_interp_point(f)
    add_projection_line(f)
    add_bound_line(f)

# Generate the plot
size = 700
p.graph(width=1.2*size, height=size, x_range=[0,1.2], y_range=[0,1],
        show_slider_labels=False, data_easing=True, bounce=False,
        loop_duration=10, loop_animation=True,
        file_name="VM_Construction.html")
exit()
