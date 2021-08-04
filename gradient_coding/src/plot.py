import numpy as np
import matplotlib.pyplot as plt

def setDisplay( xlabel, title, xticks ):
	plt.xlabel( xlabel )
	plt.ylabel( 'Time (s)' )
	plt.title( title )
	plt.xticks( group_spacing, xticks )
	plt.tight_layout()
	plt.legend()

def plotGroup( bar1, bar1Title, bar2, bar2Title, bar3, bar3Title, 
	# bar4, bar4Title, bar5, bar5Title, bar6, bar6Title, bar7, bar7Title, bar8, bar8Title, bar9, bar9Title 
	):
	edgecolor = "k"
	linewidth = 0.8

	plt.bar( group_spacing + -1 * bar_width, bar1, bar_width, alpha = opacity, color = "r", 		label = bar1Title, edgecolor = edgecolor, linewidth = linewidth )
	plt.bar( group_spacing +  0 * bar_width, bar2, bar_width, alpha = opacity, color = "g", 		label = bar2Title, edgecolor = edgecolor, linewidth = linewidth )
	plt.bar( group_spacing +  1 * bar_width, bar3, bar_width, alpha = opacity, color = "b", 		label = bar3Title, edgecolor = edgecolor, linewidth = linewidth )

	# plt.bar( group_spacing + -4 * bar_width, bar1, bar_width, alpha = opacity, color = "r", 		label = bar1Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing + -3 * bar_width, bar2, bar_width, alpha = opacity, color = "g", 		label = bar2Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing + -2 * bar_width, bar3, bar_width, alpha = opacity, color = "b", 		label = bar3Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing + -1 * bar_width, bar4, bar_width, alpha = opacity, color = "c", 		label = bar4Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing +  0 * bar_width, bar5, bar_width, alpha = opacity, color = "m", 		label = bar5Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing +  1 * bar_width, bar6, bar_width, alpha = opacity, color = "y", 		label = bar6Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing +  2 * bar_width, bar7, bar_width, alpha = opacity, color = "olive",		label = bar7Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing +  3 * bar_width, bar8, bar_width, alpha = opacity, color = "tab:pink", 	label = bar8Title, edgecolor = edgecolor, linewidth = linewidth )
	# plt.bar( group_spacing +  4 * bar_width, bar9, bar_width, alpha = opacity, color = "tab:gray", 	label = bar9Title, edgecolor = edgecolor, linewidth = linewidth )

# Number of data groups. One for each delay count
n_groups = 4


# Number of bars per group
bars_per_group = 3


# Index covering each of the groups for the delay values
index = np.arange( n_groups )


# Width of the bars
bar_width = 0.35


# Spacing between ticks
group_spacing = index * 3 * bars_per_group * bar_width


# Testing Data
	# s = 1, 100 iterations
	# Delay         0s       2s     3.5s       5s
# naive_s1 =   ( 0.22188, 2.16036, 3.65394, 5.15602 )
# cyclic_s1 =  ( 0.32262, 0.32408, 0.31988, 0.32040 )
# fracrep_s1 = ( 0.30644, 0.29682, 0.29366, 0.28853 )

# 100 iteraation average
#				    0s   0.222s   0.327s   0.459s
xticks_s1 =  		( "Delay = 0 s", 	"Delay = 0.171 s", 	"Delay = 0.245 s", 	"Delay = 0.336 s" )
naive_s1 =   		( 0.17089, 			0.24484, 			0.33604, 			0.43672 )
cyclic_s1 =  		( 0.16841, 			0.17541, 			0.18917, 			0.18615 )
fracrep_s1 = 		( 0.12534, 			0.14691, 			0.15527, 			0.15421 )
	
	# s = 2         0s     0.4s     0.6s    0.8s
xticks_s2 =  		( "Delay = 0 s", 	"Delay = 0.34 s", 	"Delay = 0.51 s", 	"Delay = 0.68 s" )
naive_s2 = 	 		( 0.17127, 			0.45003,  			0.62336, 			0.80555 )
cyclic_s2 =  		( 0.16949, 			0.18518, 			0.18466, 			0.17990 )
fracrep_s2 = 		( 0.12463, 			0.12582, 			0.11798, 			0.12190 )

# Paper Data
	# s = 1
	# Delay          0s   2s  3.5s   5s
xticks_s1_paper =  	( "Delay = 0 s", 	"Delay = 2 s", 		"Delay = 3.5 s", 	"Delay = 5 s" )
naive_s1_paper =   	( 1.8, 				3.7, 				5.2, 				6.6 )
cyclic_s1_paper =  	( 3.2, 				3.9, 				3.8, 				4.2 )
fracrep_s1_paper = 	( 3.1, 				3.2, 				3.2, 				3.3 )

	# s = 2
	# Delay          0s   3s  4.5s   6s
xticks_s2_paper =  	( "Delay = 0 s", 	"Delay = 3 s", 		"Delay = 4.5 s", 	"Delay = 6 s" )
naive_s2_paper =   	( 1.6, 				4.7, 				6.1, 				7.6 )
cyclic_s2_paper =  	( 4.8, 				5.9, 				6.0, 				5.9 )
fracrep_s2_paper = 	( 4.7, 				4.8, 				4.9, 				5.0 )


# Create plot
fig, ax = plt.subplots()
opacity = 0.8

# Plot test data with s = 1
	# Select top left plot
plt.subplot( 2, 2, 1 )

	# Plot the data
plotGroup( 	naive_s1,  "Naive",
			cyclic_s1, "Cyclic",
			fracrep_s1, "Fractional"
		)
	
	# Set display parameters for plot
setDisplay( xlabel = "s = 1 Straggler", title = "Test Data", xticks = xticks_s1 )


# Plot paper data with s = 1
	# Select top right plot
plt.subplot( 2, 2, 2 )

	# Plot the data
plotGroup( 	naive_s1_paper,  "Naive",
			cyclic_s1_paper, "Cyclic",
			fracrep_s1_paper, "Fractional"
		)
	
	# Set display parameters for plot
setDisplay( xlabel = "s = 1 Straggler", title = "Paper Data", xticks = xticks_s1_paper )



# Plot test data with s = 2
	# Select bottom left plot
plt.subplot( 2, 2, 3 )

	# Plot the data
plotGroup( 	naive_s2,  "Naive",
			cyclic_s2, "Cyclic",
			fracrep_s2, "Fractional"
		)
	
	# Set display parameters for plot
setDisplay( xlabel = "s = 2 Straggler", title = "Test Data", xticks = xticks_s2 )


# Plot paper data with s = 2
	# Select bottom right plot
plt.subplot( 2, 2, 4 )

	# Plot the data
plotGroup( 	naive_s2_paper,  "Naive",
			cyclic_s2_paper, "Cyclic",
			fracrep_s2_paper, "Fractional"
		)
	
	# Set display parameters for plot
setDisplay( xlabel = "s = 2 Straggler", title = "Paper Data", xticks = xticks_s2_paper )

ax.autoscale( tight = True )
plt.show()
