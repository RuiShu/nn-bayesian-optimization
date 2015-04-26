import os
import re
from learning_objective.hidden_function import true_evaluate, get_settings

lim_domain = get_settings(lim_domain_only=True)

scribe = open("./data/regret_analysis/gp_hm.csv", 'w')

for f in os.listdir("./data/regret_analysis"):
    if f.startswith("gp_hm"):
        print f
        f = "data/regret_analysis/" + f
        for line in open(f, 'r'):
            r = re.compile('Tasks done:(.*?). New')
            m = r.search(line)
            if m:
                print line,
                r = re.compile('\[ (.*?)\]')
                n = r.search(line)
                print n.group(1).split()

                val = n.group(1).split()
                val[0] = val[0].replace("[", "")
                print val

                query = [float(elem) for elem in val[0:4]]
                print query

                tasknum = int(m.group(1))
                y_val = true_evaluate(query, lim_domain)[0, -1]
                scribe.write(str(tasknum) + "," + str(y_val) + "\n")
                

scribe.close()
