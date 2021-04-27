# -*- coding: utf-8 -*-
"""
Created on Wed Jun 05 00:06:59 2013

@author: Harold
"""

import lg
g = lg.Lg('Tests/2p2ShortC.lg')
print g.csv()
print "ORIGINAL one"
g2 = lg.Lg('Tests/2p2.lg')
print g2.csv()

print str(g.compare(g2))
