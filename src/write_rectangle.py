import re

f = open('rectangles_train.amat','r')
g = open('rectangles.csv','w')

for line in f:
    csv_line = re.sub('\ \ \ ',',',line)
    g.write(csv_line)

f.close()
g.close()

