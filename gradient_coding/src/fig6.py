import numpy as np
import matplotlib.pyplot as plt


# Width of the bars
bar_width = 0.8

# Opacity of the bars
opacity = 0.8

# Edge colors for the bars
edgecolor = "k"


def getAvgTime( loc ):
	time_file = open( loc + "timeset.dat" )
	times = time_file.read().split( " \n" )

	del times[ len( times ) - 1 ]
	avgTime = [ float( x ) for x in times ]

	return np.average( avgTime )



topLevelDirs = 	[ 	
	"data/fig6/a/",
	"data/fig6/b/",
	"data/fig6/c/",
	"data/fig6/a_new/",
]

titles = 	[	
	"Average Time per Iteration on n = 10 Workers",
	"Average Time per Iteration on n = 20 Workers",
	"Average Time per Iteration on n = 30 Workers",
	"Average Time per Iteration on n = 10 Workers",
]

xticks = [
	[ 1, 2, 3, 4 ],
	[ 1, 2, 3, 4, 5, 6, 7 ],
	[ 1, 2, 3, 4, 5, 6, 7 ],
	[ 1, 2, 3, 4 ],
]

filenames = [ 
	[ "naive_acc_0_", 	"replication_acc_1_", 	"coded_acc_1_", 		"avoidstragg_acc_1_" ],															# 6a
	[ "naive_acc_0_",	"replication_acc_3_",	"replication_acc_4_",	"coded_acc_3_",	"coded_acc_4_", "avoidstragg_acc_3_",	"avoidstragg_acc_4_" ], # 6b
	[ "naive_acc_0_",	"replication_acc_5_",	"replication_acc_9_",	"coded_acc_5_",	"coded_acc_9_", "avoidstragg_acc_5_",	"avoidstragg_acc_9_" ],	# 6c
	[ "naive_acc_1_", 	"replication_acc_1_", 	"coded_acc_1_", 		"avoidstragg_acc_1_" ],															# 6a_new
]

labels = [
	[ "Naive",			"FracRep\ns=1",			"CycRep\ns=1",			"Ignore\nStragg\ns=1"  ],															# 6a
	[ "Naive",			"FracRep\ns=3",			"FracRep\ns=4",			"CycRep\ns=3",	"CycRep\ns=4",	"Ignore\nStragg\ns=3",	"Ignore\nStragg\ns=4"  ],	# 6b
	[ "Naive",			"FracRep\ns=5",			"FracRep\ns=9",			"CycRep\ns=5",	"CycRep\ns=9",	"Ignore\nStragg\ns=5",	"Ignore\nStragg\ns=9"  ],	# 6c
	[ "Naive",			"FracRep\ns=1",			"CycRep\ns=1",			"Ignore\nStragg\ns=1"  ],															# 6a_new
]


save_location_dir = "data/fig6/"
save_location_fname = [
	"fig6a.pdf",
	"fig6b.pdf",
	"fig6c.pdf",
	"fig6a_new.pdf"
]

subfigure_sel = 3
rows = 1
cols = 1


filenames[ subfigure_sel ] = [ ( topLevelDirs[ subfigure_sel ] + s ) for s in filenames[ subfigure_sel ] ]


plt.rcParams["font.size"] = 12
# plt.rcParams["font.weight"] = "bold"
plt.subplot( rows, cols, 1 )
for i in range( len( filenames[ subfigure_sel ] ) ):
	avgTime = getAvgTime( filenames[ subfigure_sel ][ i ] )
	plt.bar( x = ( i + 1 ), height = avgTime, width = bar_width, alpha = opacity, color = "b", label = labels[ subfigure_sel ][ i ], edgecolor = edgecolor )

ax = plt.gca()
ax.set_xticks( xticks[ subfigure_sel ] )
ax.set_xticklabels( labels[ subfigure_sel ] )

plt.ylabel( "Average Time per Iteration (s)" )
plt.title( titles[ subfigure_sel ] )

plt.tight_layout()

plt.savefig( save_location_dir + save_location_fname[ subfigure_sel ] )

# plt.show()
