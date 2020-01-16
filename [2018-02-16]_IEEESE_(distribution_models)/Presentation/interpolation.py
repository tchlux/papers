from util.plotly import Plot

p = Plot("","System Paramter","File I/O Throughput (kb/s) Mean")
n1 = "Config 1"
n2 = "Config 2"
n3 = "Config 3"
p.add_node(n1, 0, 0, color=p.color(0), size=10)
p.add_node(n2, 1, 1, color=p.color(1), size=10)
p.add_node(n3, 2, .3, color=p.color(2), size=10)
p.add_edge([n1,n2,n3])
p.graph(file_name="interpolating_values.html", show_titles=True,
        y_range=[-.15,1.15], x_range=[-.2,2.2])

p = Plot("","System Paramter","File I/O Throughput (kb/s) Distribution")
n1 = "Config 1"
n2 = "Config 2"
n3 = "Config 3"
p.add_node(n1, 0, 0, color=p.color(0), size=10, label=True, label_y_offset=-.08)
p.add_node(n2, 1, 1, color=p.color(1), size=10, label=True, label_y_offset=.08)
p.add_node(n3, 2, .3, color=p.color(2), size=10, label=True, label_y_offset=-.08)
p.add_edge([n1,n2,n3])
p.graph(file_name="interpolating_functions.html", show_titles=True,
        y_range=[-.15,1.15], x_range=[-.2,2.2])
