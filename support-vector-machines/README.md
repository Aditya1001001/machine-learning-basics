Initial accuracy without setting the parameters, i.e. without defining a kernel function or setting a soft margin : 90.7

With a linear kernel: 97.6

​	with a soft margin of 2 : 95.3

With a polynomial kernel: 93 (it also takes a long time to train, maybe because of computation complexity) 

​	soft margin of 2: 87.2 

​   if we restrict the degree to to two, the model      takes lesser times to train but also has a lower    accuracy of 83.7

KNN 
    with 9 n_neighbours: 89.5
    with 12 n_neighbours: 94.1
    with 13 n_neighbours: 95.3