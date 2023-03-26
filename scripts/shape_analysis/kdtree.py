import numpy as numpy

data=numpy.array([ [1,9] , [2,3] , [4,1] , [3,7]  , [5,4]  , [6,8]  ,[7,2]   , [8,8]   , [7,9]   , [9,6] ] )


#print data.shape
x,y=numpy.median(data, 0)
#print numpy.median(data, 0)
#print numpy.median(data, 1)



print (data [   numpy.where( data[:,0]<x  )   ])

subset1_smaller=data [   numpy.where( data[:,0]<x  )   ]
subset1_larger= data [   numpy.where( data[:,0]>=x  )   ]

print (data [   numpy.where( data[:,0]>x  )   ])
#print data[:,0]



x,y=numpy.median(subset1_smaller, 0)


print ("-----------------------")
print ( subset1_smaller[ numpy.where( subset1_smaller[:,1]<y  )  ])
print ("-----------------------")

print (subset1_smaller[ numpy.where( subset1_smaller[:,1]>=y  )  ])



x,y=numpy.median(subset1_larger, 0)
print ("-----------------------")
print (subset1_larger[ numpy.where( subset1_larger[:,1]<y )  ])


print ("-----------------------")
print (subset1_larger[ numpy.where( subset1_larger[:,1]>=y )  ])