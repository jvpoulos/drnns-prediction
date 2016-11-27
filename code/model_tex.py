## Convert dot object visualization of model to tex/tikz format 

import pydot
import dot2tex as d2t

dot_graph = pydot.graph_from_dot_file('results/final_model.dot')
texcode = d2t.dot2tex(dot_graph.to_string(), format='tikz', crop=True)

fout = open("results/final_model.tex", 'w')
fout.write(texcode)
fout.close()