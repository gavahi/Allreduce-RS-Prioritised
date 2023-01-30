In this work, I consider the priority of elements in reduce-scatter operation in MPI library.

The new scheduling proposed in this work has been designed in which elements with higher priority process sooner and to eliminate unnecessary delays to handle them.

The main implementation has been done in the Allreduce files located at src/mpi/coll/


You can find more details about this work in here:

https://github.com/gavahi/Allreduce-RS-Prioritised/blob/main/Priority-aware%20MPI_Reduce-Scatter.pdf
