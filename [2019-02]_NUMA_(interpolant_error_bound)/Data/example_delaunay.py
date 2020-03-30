from util.plot import Plot, multiplot

p = Plot()
p.add_node("top center", 2, -.45, size=335,
           marker_line_width=1,
           marker_line_color="rgba(0,0,0,.2)",
           color='rgba(0,0,0,.05)')
p.add_node("a", 0, 1, size=10, color=p.color(0))
p.add_node("b", 2, 0, size=10, color=p.color(1))
p.add_node("c", 2, 2, size=10, color=p.color(2))
p.add_node("d", 4, 1, size=10, color=p.color(3))
p.add_node("", 2, 3, display=False)
p.add_node("", 2, -1, display=False)
p.add_edge(['a','b','d','a'])
p.add_edge(['a','c','d','a'])
p1 = p.graph(width=400, height=400, file_name="example_not_delaunay.html")


p = Plot()
p.add_node("left center", 1.25, 1, size=170,
           marker_line_width=1,
           marker_line_color="rgba(0,0,0,.2)",
           color='rgba(0,0,0,.05)')
p.add_node("a", 0, 1, size=10, color=p.color(0))
p.add_node("b", 2, 0, size=10, color=p.color(1))
p.add_node("c", 2, 2, size=10, color=p.color(2))
p.add_node("d", 4, 1, size=10, color=p.color(3))
p.add_node("", 2, 3, display=False)
p.add_node("", 2, -1, display=False)
p.add_edge(['a','b','c','a'])
p.add_edge(['b','c','d','b'])
p2 = p.graph(width=400, height=400, file_name="example_delaunay.html")
# p2 = p.graph(html=False)

# multiplot([p1,p2], file_name="example_delaunay.html")
