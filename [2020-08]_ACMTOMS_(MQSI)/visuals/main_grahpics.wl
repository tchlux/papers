(* Set colors for styles. *)
red = RGBColor[0.8,0.2,0.2];
blue = RGBColor[0.5,0.75,0.9];
green = RGBColor[0.25,0.7,0.4];
purple = RGBColor[0.7,0.5,0.75];
darkGray = RGBColor[0.35,0.35,0.45];
lightGray = RGBColor[0.8, 0.8, 0.8];
font = {FontSize->10, FontFamily->"Liberation Serif", FontColor->Black};
markerSize = .03;
edgeThickness = .15;
circleMarker = Graphics[{Black, Thickness[2*edgeThickness], Circle[]}];
dZeroMarker  = Graphics[{FaceForm[blue],  EdgeForm[{Thickness[edgeThickness],Black}], Disk[]}];
ddZeroMarker = Graphics[{FaceForm[purple],EdgeForm[{Thickness[edgeThickness],Black}], Rectangle[]}];
dPosMarker   = Graphics[{FaceForm[green], EdgeForm[{Thickness[edgeThickness],Black}], Triangle[{{-.5,-.5},{0,.5},{.5,-.5}}]}];
dNegMarker   = Graphics[{FaceForm[red],   EdgeForm[{Thickness[edgeThickness],Black}], Triangle[{{-.5,.5},{0,-.5},{.5,.5}}]}];
fileExtension = ".eps";


(* Make a plot of the points where the constraints are styled with
   the plot markers (increasing -> up triangle, decreasing -> down
   triangle, extreme point -> circle, flat point -> square) *)
StyledPoints[x_, y_, ms_:markerSize] := {
  ClearAll[dZero, ddZero, dPos, dNeg, left, right, points, series, markers, styles];
  (* Break up all points into increasing, decreasing, extreme, or flat. *)
  dZero={}; ddZero={}; dPos={}; dNeg={};
  For[i=1,i<=Length[x],i++,{
    left  = If[i>1,            y[[i]]-y[[i-1]], y[[2]]-y[[1]]];
    right = If[i+1<=Length[x], y[[i+1]]-y[[i]], y[[Length[y]]]-y[[Length[y]-1]]];
    Which[
      (Abs[left] < 2^(-26) || Abs[right] < (2^-26)),AppendTo[ddZero,i],
      (left right)<0, AppendTo[dZero,i],
      (left>0),       AppendTo[dPos,i],
      (left<0),       AppendTo[dNeg,i],
      True, Print["ERROR: An error occurred in StyledPoints."]
    ];
  }];
  (* Convert all indices to actual (x,y) points. *)
  toPoints[i_] := {x[[i]], y[[i]]};
  (* Organize all the points into a list of points, marker styles, and series colors. *)
  series = {}; markers = {};
  If[Length[dZero] > 0, {
    AppendTo[series,  Map[toPoints,dZero]];
    AppendTo[markers, {dZeroMarker,ms}];
  }];
  If[Length[ddZero] > 0, {
    AppendTo[series,  Map[toPoints,ddZero]];
    AppendTo[markers, {ddZeroMarker,ms}];
  }];
  If[Length[dPos] > 0, {
    AppendTo[series,  Map[toPoints,dPos]];
    AppendTo[markers, {dPosMarker,ms}];
  }];  
  If[Length[dNeg] > 0, {
    AppendTo[series,  Map[toPoints,dNeg]];
    AppendTo[markers, {dNegMarker,ms}];
  }];  
  (* Create the graphic with all stylized points, return. *)
  points = ListPlot[series, PlotMarkers->markers, PlotRange->Full]
}[[1]];


(* Use the MQSI command line interface to fit a quintic spline to x and y,
   then evaluate the spline (derivative D) at points u and return results. *)
mqsi[x_,y_,u_,D_:0] := {
  ClearAll[n, m, qu];
  (* Read the data that was given to make an MQSI. *)
  n = Length[x];
  m = Length[u];
  (* Export a null output file, to make sure one exists. *)
  Export["temp.out", {0}, "Table", "FieldSeparators" -> " "]; 
  (* Export a data file and points file for the MQSI command line interface. *)
  Export["temp.data", {n,N[x],N[y]}, "Table", "FieldSeparators" -> " "];
  Export["temp.pts", {m,N[u]}, "Table", "FieldSeparators" -> " "];
  (* Write a file containing all the points to get the MQSI
     evaluations by using the MQSI command line interface. *)
  Run["rm temp.out"]; (* Remove the existing file. *)
  Run[StringJoin[{"./mqsi -d ", ToString[D], " temp.data temp.pts temp.out"}]];
  (* Read the values of the MQSI at all points. *)
  qu = Map[Internal`StringToMReal,StringSplit[Import["temp.out","Text"],Whitespace]];
  Run["rm temp.data temp.pts temp.out "]; (* Remove the files *)
  (* Return the result of the MQSI. *)
  qu
}[[1]];

(* Make a plot of the MQSI to data, including stylized markers for all
   data determined by local function change conditions. *)
makePlot[name_, x_, y_, m_:100, width_:400, height_:130, markSize_:markerSize] := {
  ClearAll[u, qu, points, graphic];
  Print[""];
  Print["Making '"<>name<>"'.."];
  (* Get the output of a MQSI with equally spaced approximation points. *)
  Print["  constructing MQSI interpolant.."];
  u = Subdivide[x[[1]],x[[-1]],m-1];
  qu = mqsi[x,y,u];
  (* Add stylized data points to graphic. *)
  Print["  adding stylized points to graphic.."];
  points = StyledPoints[x,y,ms=markSize];
  (* Add the MQSI approximation to the graphic. *)
  Print["  adding MQSI predictions at ",m," equally spaced points to graphic.."];
  graphic = Show[
    ListLinePlot[Transpose[{u,qu}], LabelStyle->font, PlotRange->Full,
		 PlotStyle->{Thickness[.003],darkGray}],
    points, ImageSize->{width,height}, AspectRatio->Full,
    Method->{"AxesInFront"->False}, PlotRange->Full];
  Print["  exporting graphic to '"<>name<>"'.."];
  Export[name, graphic];
  Print[""];
  (* Return the graphic for later use (if desired). *)
  graphic
}[[1]];


(* Make a plot of the MQSI to data, including stylized markers for all
   data determined by local function change conditions. *)
quadraticSensitivity[name_, offset_, q1_, q2_, m_:100, width_:400, height_:150] := {
  ClearAll[center, x];
  Print[""];
  Print["Making '"<>name<>"'.."];
  (* Get the location that the two quadratics intercept each other. *)
  center = Solve[q1[z] == q2[z], {z}][[1,1]][[2]];
  x = offset + center;
  Print["  x:  ",x];
  (* Get the indices of the left and right sides. *)
  left[i_] := x[[i]] <= center;
  right[i_] := x[[i]] > center;
  leftSide = Select[Range[1,Length[x]], left];
  rightSide = Select[Range[1,Length[x]], right];
  (* Construct the MQSI for each set of points, one of them with a
     slightly decreased curvature. *)
  u = Subdivide[x[[1]], x[[-1]], m-1];
  y1 = Join[q1[x[[leftSide]]], q2[x[[rightSide]]]];
  y2 = Join[q1[x[[leftSide]]], q2[x[[rightSide]]] - (2^(-48))*(offset[[rightSide]])^2];
  Print["  y1: ",y1];
  Print["  y2: ",y2];
  qu1 = mqsi[x, y1, u];
  qu2 = mqsi[x, y2, u];
  (* Plot both MQSI to see the difference. *)
  data1 = Transpose[{u,qu1}];
  data2 = Transpose[{u,qu2}];
  Print["  adding stylized points to graphic.."];
  points = ListPlot[Transpose[{x,y1}], LabelStyle->font, PlotRange->Full,
		    PlotMarkers->{circleMarker,.03}, PlotStyle->Black, 
		    PlotRange->Full];
  Print["  adding MQSI predictions at ",m," equally spaced points to graphic.."];
  graphic = Show[
    ListLinePlot[{data1,data2}, LabelStyle->font,
		 PlotStyle->{{Thickness[.003],blue},{Thickness[.003],red,Dashed}}],
    points, ImageSize->{width,height}, AspectRatio->Full,
    Method->{"AxesInFront" -> False}, PlotRange->Full];
  Print["  exporting graphic to '"<>name<>"'.."];
  Export[name, graphic];
  Print[""];
}


(* Construct the specific zoomed plot for the paper. This is not a 
   reusable function, it is only here to compartmentalize code. *)
makeZoomedPlot[name_, x_, y_, m_:1000, width_:450, height_:350] := {
  g = makePlot[name, x, y, m, width, height, markSize=.015];
  (* Make a subplot zoomed in on a "sharp" looking corner. *)
  windowRadius = .001;
  u = Subdivide[x[[-2]]-windowRadius, x[[-2]]+windowRadius, 100];
  qu = mqsi[x,y,u];
  subplot = Show[
    ListLinePlot[Transpose[{u,qu}],PlotRange->Full,LabelStyle->font,
  		 PlotStyle->{Thickness[.012],darkGray}],
    ListPlot[{{x[[-2]],y[[-2]]}}, PlotMarkers->{dPosMarker,.1}, PlotRange->Full],
    Axes->False
  ];
  (* Make the lines showing the "magnifier" location. *)
  ratio = 1.43; (* y/x aspect ratio of existing visual. *)
  dRadius = .133; (* Radius (in x) of the destination circle. *)
  tRadius = .01; (* Radius (in x) of the target circle. *)
  dCenter = {.25, .58}; (* Center of the destination circle. *)
  dOffset = {.004, 0.0}; (* Offset of the destination circle center. *)
  tyOffset = -.005; (* Target circle y offset. *)
  thck = .003; (* Thickness of the target circle line (other thicknesses are chosen from it). *)
  botOffset = {.8*dRadius, .5*dRadius}; (* Buttom line destination offset. *)
  (* Target circle. *)
  target = {Black, Thickness[thck], Circle[{x[[-2]],y[[-2]]+tyOffset}, {tRadius, ratio*tRadius}]};
  (* Top line from target to destination. *)
  topLine = {lightGray, Dashed, Thickness[thck/2], Line[{dCenter+{0,ratio*dRadius},           {x[[-2]], y[[-2]]+ratio*tRadius+tyOffset}}]};
  (* Bottom line from target to destination. *)
  botLine = {lightGray, Dashed, Thickness[thck/2], Line[{dCenter-{0,ratio*dRadius}+botOffset, {x[[-2]], y[[-2]]-ratio*tRadius+tyOffset}}]};
  (* Destination circle. *)
  destination = {Black, Thickness[1.5*thck], Circle[dCenter+dOffset, {dRadius, ratio*dRadius}]};
  (* Combine all the graphics into one final output. *)
  g2 = Show[g, Epilog->{Inset[subplot, dCenter, Automatic, Scaled[.30]],
  		      topLine, botLine, target, destination}];
  Export[name, g2];
  g2
};


(* Construct a stacked bar chart (data needs to be stacked already,
   { {<col 1>}, {<col 2>}, ... {<col N>} }, where there are N xLabels. *)
stackedBarChart[data_, xlabels_, colors_, yAxisName_, xAxisName_,
		    showLegend_:True, width_:450, height_:150] := {
  legend = If[showLegend, Placed[SwatchLegend[
		  colors, names, LabelStyle->font,
		  LegendLayout->"Column"], {.2,.7}], None];
  Show[BarChart[data,ChartLayout->"Stacked",
		ChartStyle->colors,
		ChartLegends->legend
       ],
       Frame -> {{True, None}, {True, None}},
       FrameLabel -> {{yAxisName, None}, {xAxisName , None}},
       FrameTicks -> {{Automatic, Automatic},
		      {Transpose[{Range[Length[data]],xlabels}], None}},
       BaseStyle -> font,
       ImageSize->{width,height},
       AspectRatio->Full]
  }[[1]];


(* ------------------------------------------------------------------ *)


(* Sensitivity demonstration of quadratic facet model. *)
q1[x_] := x^2;
q2[x_] := (x-2)^2 + 6;
quadraticSensitivity["1-sensitivity"<>fileExtension, {-1.5,-0.5,0.0,0.5,1.5}, q1, q2]


(* ------------------------------------------------------------------ *)


(* Piecewise polynomial. *)
y = {0.0, 1.0, 1.0, 1.0, 0.0, 20.0, 19.0, 18.0, 17.0,  0.0,  0.0,  3.0,  0.0,  1.0,  6.0, 16.0, 16.1, 1.0};
x = Subdivide[0, 1, Length[y]-1];
makePlot["2-piecewise-polynomial"<>fileExtension, x, y, m=1000];

(* Large tangent *)
x = { 0.0, 0.2098765432098766, 0.39506172839506193, 0.5555555555555556, 0.691358024691358, 0.8024691358024691, 0.8888888888888888, 0.9506172839506173, 0.9876543209876543, 1.0 };
y = { -0.00990099009900991, 0.249807128529548, 0.626179482031721, 1.2004889975550124, 2.138318481208833, 3.818560380725758, 7.25688073394495, 15.839916839916839, 43.751381215469486, 98.99999999999991 };
makePlot["3-large-tangent"<>fileExtension, x, y, m=1000];

(* Signal decay *)
x = { 0.0, 0.05263157894736842, 0.10526315789473684, 0.15789473684210525, 0.21052631578947367, 0.2631578947368421, 0.3157894736842105, 0.3684210526315789, 0.42105263157894735, 0.47368421052631576, 0.5263157894736842, 0.5789473684210527, 0.631578947368421, 0.6842105263157894, 0.7368421052631579, 0.7894736842105263, 0.8421052631578947, 0.894736842105263, 0.9473684210526315, 1.0 };
y = { 0.0, 9.432708787172459, 4.284713438563181, -5.8890539191352635, -5.800712066272032, 1.9184371257428916, 4.989833591891926, 0.6982214699335515, -3.3026390701546022, -1.8935165608964157, 1.6291755274134456, 2.104355002689216, -0.3299203058930831, -1.754105324112179, -0.5050258870439092, 1.1574764406463904, 0.9092650864532725, -0.5285051026957308, -0.9718230935964967, -8.906522175617114*10^(-16) };
makePlot["4-signal-decay"<>fileExtension, x, y, m=1000];


(* ------------------------------------------------------------------ *)


(* Real VarSys data used to demonstrate the CDF approximation. *)
data = Transpose[Import["3.2GHz-readers.csv"]][[1,2;;]];
(* Set the random seed for repeatability. *)
SeedRandom[1];
sample = Sort[RandomSample[data,100]];
(* Plot MQSI and the 'truth' in the same visual. *)
(*   generate approximation of CDF with MQSI using equally spaced percentiles. *)
groups = 13;
smoothed = sample[[ 1 ;; Length[sample] ;; Floor[Length[sample]/groups] ]];
x = Join[{data[[1]]}, smoothed, {data[[-1]]}];
y = Join[{0}, Subdivide[0,1,Length[x]-3], {1}];
approx = makePlot["5-real-data"<>fileExtension, x, y, m=1000];
(*   generate 'truth' as an empirical CDF *)
trueEDF = EmpiricalDistribution[data];
trueCDF[x_] := CDF[trueEDF,x];
truth = Plot[trueCDF[x], {x,data[[1]],data[[-1]]}, PlotStyle->{Thickness[.003],red,Dashed}];
vis = Show[{truth,approx},
	   ImageSize->{450,220}, AspectRatio->Full,
	   Method->{"AxesInFront"->False}, PlotRange->Full];
Export["5-real-data"<>fileExtension,vis];


(* ------------------------------------------------------------------ *)


(* Random monotone, with sub-window showing smoothness. *)
x = { 0.025926231827891333, 0.13457994534493356, 0.18443986564691528, 0.2046486340378425, 0.26682727510286663, 0.29965467367452314, 0.3303348210038741, 0.42036780208748903, 0.4353223926182769, 0.43599490214200376, 0.5135781212657464, 0.5291420942770391, 0.5496624778787091, 0.6192709663506637, 0.6211338327692949, 0.7853351478166735 };
y = { 0.06528650438687811, 0.079645477009061, 0.09653091566061256, 0.10694568430998297, 0.12715997170127746, 0.20174322626496533, 0.2203062070705597, 0.22601200060423587, 0.34982628500329926, 0.42812232759738944, 0.46778748458230024, 0.4942368373819278, 0.505246090121704, 0.5967453089785958, 0.846561485357468, 0.8539752926394888 };
makeZoomedPlot["6-random-monotone"<>fileExtension, x, y, m=1000, width=450, height=350];


(* (\* ------------------------------------------------------------------ *\) *)


(* Print["Making timing graphics '7-runtimes-10' and '8-runtimes-100'.."] *)
(* (\* Timing data stacked bar chart. *\) *)
(* data = Import["timing_data.csv","csv"]; *)
(* sizes = data[[2;;,1]]; *)
(* names = data[[1,2;;]] *)
(* data = data[[2;;,2;;]]; *)

(* (\* 10 -- 100 *\) *)
(* tempSizes = sizes[[;;10]]; *)
(* tempData = data[[;;10]]; *)
(* chart = stackedBarChart[tempData, tempSizes, {red, green, blue}, *)
(* 			"Time (seconds)", ""]; *)
(* Export["7-runtimes-10"<>fileExtension,chart]; *)

(* (\* 100 -- 1000  *\) *)
(* tempSizes = sizes[[10;;]]; *)
(* tempData = data[[10;;]]; *)
(* chart = stackedBarChart[tempData, tempSizes, {red, green, blue}, *)
(* 			"Time (seconds)", "Data Size", False]; *)
(* Export["8-runtimes-100"<>fileExtension,chart]; *)


