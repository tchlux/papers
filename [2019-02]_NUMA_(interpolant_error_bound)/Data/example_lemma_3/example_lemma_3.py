from util.plot import Plot, multiplot
from polynomial import Spline


ANY_FUNCTION = True
LAST_ALLOWABLE = True
NOT_ALLOWED = True

if ANY_FUNCTION:
    # Construct the "last allowable" function, any larger first derivative
    # would cause a prohibatively large slope of the function.
    # g = Spline([0, .1, .3, .5, .8, 1], [[0,.3], [.05, .2], [.1,.04], [.2,0], [-.1,0], [0,0]])
    # Dg = g.derivative()
    Dg = Spline([0, .6, 1], [[.4], [-.4], [.4]])
    g = Dg.integral()
    # DDg = Dg.derivative()
    p = Plot(font_size=24, font_family="times")

    # Add the line for "w"
    boundary = Spline([0, 1], [[Dg(0)], [Dg(0)-2]])
    p.add_func("Bound", boundary, [0, 1], color=(0,0,0,.2), dash="dot",)

    # Add endpoints, regular function, and derivative function.
    p.add("g points", [0, 1], [0, 0], color=1,)
    p.add_func("g", g, [0, 1], color=1,)
               # fill="tozeroy", fill_color="rgba(0,0,0,.1)")
    p.add_func("Dg", Dg, [0, 1], color=p.color(2,alpha=.8), dash="dash",
               fill="tozeroy", fill_color="rgba(0,0,0,.1)")
    # p.add_func("DDg", DDg, [0, 1], color=p.color(3,alpha=.8), dash="dot")
    # Add annotations (in place of legend)
    p.add_annotation("g'(t)", .14, Dg(.14),
                     font_family="times", font_size=17)
    p.add_annotation("g(t)", .5, g(.5),
                     font_family="times", font_size=17)
    p.add_annotation(" w ", .5, boundary(.5), ax=-30, ay=40,
                     font_family="times", font_size=17)
    p.add_annotation(" t ", .2, 0, ax=0, ay=56,
                     font_family="times", font_size=17,)
    p.add_annotation(" ~ ", .2, 0, ax=0, ay=46,
                     font_family="times", font_size=12,
                     bg_color="rgba(0,0,0,0)",
                     arrow_color="rgba(0,0,0,0)")

    # Set the axis settings to have custom labels.
    x_axis_settings = dict(
        tickmode='array',
        tickvals=[0, 1/2, 1],
        ticktext=["0", "0.5", "1"],
    )
    y_axis_settings = dict(
        tickmode='array',
        tickvals=[-1, -1/2, 0, 1/2, 1],
        ticktext=["-𝛾<sub>g </sub>/ 2", "", "0", "", "𝛾<sub>g </sub>/ 2"],
    )

    # Make plot.
    p_any = p.graph(y_range=[-1.1,1.2], show_grid=True, show_ticks=True,
                    show_line=False, show_zero_line=True,
                    show_legend=False, show_titles=False,
                    width=450, height=400,
                    y_axis_settings=y_axis_settings,
                    x_axis_settings=x_axis_settings,
                    file_name="example_lemma_3.html",
    )
    if not (LAST_ALLOWABLE or NOT_ALLOWED): exit()

if LAST_ALLOWABLE:
    # Construct the "last allowable" function, any larger first derivative
    # would cause a prohibatively large slope of the function.
    # Dg = Spline([0, 1/2-.1, 1/2+.1, 1], [[.8], [0], [0], [-.8]])
    Dg = Spline([0, 1/2, 1], [[1], [0], [-1]])
    g = Dg.integral()
    # DDg = Dg.derivative()
    p = Plot(font_size=24, font_family="times")

    # Add the line for "w"
    boundary = Spline([0, 1], [[Dg(0)], [Dg(0)-2]])
    p.add_func("Bound", boundary, [0, 1], color=(0,0,0,.2), dash="dot",)

    # Add endpoints, regular function, and derivative function.
    p.add("g points", [0, 1], [0, 0], color=1,)
    p.add_func("g", g, [0, 1], color=1,)
               # fill="tozeroy", fill_color="rgba(0,0,0,.1)")
    p.add_func("Dg", Dg, [0, 1], color=p.color(2,alpha=.8), dash="dash",
               fill="tozeroy", fill_color="rgba(0,0,0,.1)")
    # p.add_func("DDg", DDg, [0, 1], color=p.color(3,alpha=.8), dash="dot")
    # Add annotations (in place of legend)
    p.add_annotation("g'(t)", .14, Dg(.14),
                     font_family="times", font_size=17)
    p.add_annotation("g(t)", .65, g(.65),
                     font_family="times", font_size=17)
    p.add_annotation(" w ", .76, boundary(.76), ax=-30, ay=40,
                     font_family="times", font_size=17)
    p.add_annotation(" t ", .5, 0, ax=0, ay=56,
                     font_family="times", font_size=17,)
    p.add_annotation(" ~ ", .5, 0, ax=0, ay=46,
                     font_family="times", font_size=12,
                     bg_color="rgba(0,0,0,0)",
                     arrow_color="rgba(0,0,0,0)")

    # Set the axis settings to have custom labels.
    x_axis_settings = dict(
        tickmode='array',
        tickvals=[0, 1/2, 1],
        ticktext=["0", "0.5", "1"]
    )
    y_axis_settings = dict(
        tickmode='array',
        tickvals=[-1, -1/2, 0, 1/2, 1],
        ticktext=["", "", "", "", ""]
    )

    # Make plot.
    p_last = p.graph(y_range=[-1.1,1.2],
                     show_grid=True, show_ticks=True,
                     show_line=False, show_zero_line=True,
                     show_legend=False, show_titles=False,
                     width=450, height=400,
                     y_axis_settings=y_axis_settings,
                     x_axis_settings=x_axis_settings,
                     file_name="example_lemma_3.html"
    )
    if not (ANY_FUNCTION or NOT_ALLOWED): exit()


if NOT_ALLOWED:
    # Construct the "last allowable" function, any larger first derivative
    # would cause a prohibatively large slope of the function.
    Dg = Spline([0, 1/2, 1], [[1.1], [.1], [-.9]])
    g = Dg.integral()
    # DDg = Dg.derivative()
    p = Plot(font_size=24, font_family="times")

    # Add the line for "w"
    boundary = Spline([0, 1], [[Dg(0)], [Dg(0)-2]])
    p.add_func("Bound", boundary, [0, 1], color=(0,0,0,.2), dash="dot",)

    # Add the line for "last allowable"
    # last_down = Spline([0, 1], [[1], [-1]])
    # p.add_func("Last down", last_down, [0, 1], color=p.color(0, alpha=.8),
    #            line_width=.8, dash="dot")

    # Add endpoints, regular function, and derivative function.
    p.add("g points", [0, 1], [0, 0], color=1,)
    p.add_func("g", g, [0, 1], color=1,)
               # fill="tozeroy", fill_color="rgba(0,0,0,.1)")
    p.add_func("Dg", Dg, [0, 1], color=p.color(2,alpha=.8), dash="dash",
               fill="tozeroy", fill_color="rgba(0,0,0,.1)")

    # p.add_func("DDg", DDg, [0, 1], color=p.color(3,alpha=.8), dash="dot")
    # Add annotations (in place of legend)
    p.add_annotation("g'(t)", .14, Dg(.14),
                     font_family="times", font_size=17)
    p.add_annotation("g(t)", .65, g(.65),
                     font_family="times", font_size=17)
    p.add_annotation(" w ", .816, boundary(.816), ax=-30, ay=40,
                     font_family="times", font_size=17)
    p.add_annotation(" t ", .55, 0, ax=0, ay=56,
                     font_family="times", font_size=17,)
    p.add_annotation(" ~ ", .55, 0, ax=0, ay=46,
                     font_family="times", font_size=12,
                     bg_color="rgba(0,0,0,0)",
                     arrow_color="rgba(0,0,0,0)")


    p.add_node("", 1-.03, g(1)-.01, size=35,
               marker_line_color=p.color(0), color=p.color(0, alpha=0))

    # Set the axis settings to have custom labels.
    x_axis_settings = dict(
        tickmode='array',
        tickvals=[0, 1/2, 1],
        ticktext=["0", "0.5", "1"]
    )
    y_axis_settings = dict(
        tickmode='array',
        tickvals=[-1, -1/2, 0, 1/2, 1],
        ticktext=["", "", "", "", ""],
        # ticktext=["-𝛾<sub>g </sub>/ 2", "", "0", "", "𝛾<sub>g </sub>/ 2"],
        # position=1,
    )

    # Make plot.
    p_bad = p.graph(y_range=[-1.1,1.2],
                    show_grid=True, show_ticks=True,
                    show_line=False, show_zero_line=True,
                    show_legend=False, show_titles=False,
                    width=450, height=400,
                    y_axis_settings=y_axis_settings,
                    x_axis_settings=x_axis_settings,
                    file_name="example_lemma_3.html"
    )
    if not (ANY_FUNCTION or LAST_ALLOWABLE): exit()



# Construct a multiplot holding all the generated figures.
to_show = []
if ANY_FUNCTION: to_show.append( p_any )
if LAST_ALLOWABLE: to_show.append( p_last )
if NOT_ALLOWED: to_show.append( p_bad )

multiplot(to_show, file_name="example_lemma_3.html", gap=.04, 
          width=800, height=200)
