/*  Copyright (c) 2012 Matthew Anderson                                          */
/*                                                                               */
/*  Distributed under the Boost Software License, Version 1.0. (See accompanying */
/*  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int main(int argc,char** argv)
{
   int          taskid, ntasks;
   MPI::Status  status;
   int          ierr,i,j,itask,jtask;
   int          buffsize;
   double       *sendbuff,**recvbuff,buffsum;
   double       inittime,totaltime;

   /*===============================================================*/
   /* MPI Initialisation. Its important to put this call at the     */
   /* begining of the program, after variable declarations.         */
   MPI::Init(argc, argv);

   /*===============================================================*/
   /* Get the number of MPI tasks and the taskid of this task.      */
   taskid = MPI::COMM_WORLD.Get_rank();
   ntasks = MPI::COMM_WORLD.Get_size();

   /*===============================================================*/
   /* Get buffsize value from program arguments.                    */
   buffsize=atoi(argv[1]);

   /*===============================================================*/
   /* Printing out the description of the example.                  */
   if ( taskid == 0 ){
     printf("\n\n\n");
     printf("##########################################################\n\n");
     printf(" Example 10 \n\n");
     printf(" Collective Communication : MPI::COMM_WORLD.Allgather \n\n");
     printf(" Vector size: %d\n",buffsize);
     printf(" Number of tasks: %d\n\n",ntasks);
     printf("##########################################################\n\n");
     printf("                --> BEFORE COMMUNICATION <--\n\n");
   }

   /*=============================================================*/
   /* Memory allocation.                                          */
   sendbuff = new double[buffsize];
   recvbuff = new double*[ntasks];
   recvbuff[0] = new double[ntasks*buffsize];
   for(i=1;i<ntasks;i++)recvbuff[i]=recvbuff[i-1]+buffsize;

   /*=============================================================*/
   /* Vectors and/or matrices initalisation.                      */
   srand((unsigned)time( nullptr ) + taskid);
   for(i=0;i<buffsize;i++){
       sendbuff[i]=(double)rand()/RAND_MAX;
   }

   /*==============================================================*/
   /* Print out before communication.                              */

   MPI::COMM_WORLD.Barrier();

   buffsum=0.0;
   for(i=0;i<buffsize;i++){
     buffsum=buffsum+sendbuff[i];
   }
   printf("Task %d : Sum of vector = %e \n",taskid,buffsum);

   /*===============================================================*/
   /* Communication.                                                */

   inittime = MPI::Wtime();

   MPI::COMM_WORLD.Allgather(sendbuff,buffsize,MPI::DOUBLE,
                             recvbuff[0],buffsize,MPI::DOUBLE);

   totaltime = MPI::Wtime() - inittime;

   /*===============================================================*/
   /* Print out after communication.                                */

   if ( taskid == 0 ){
     printf("\n");
     printf("##########################################################\n\n");
     printf("                --> AFTER COMMUNICATION <-- \n\n");
   }

   for(jtask=0;jtask<ntasks;jtask++){
     MPI::COMM_WORLD.Barrier();
     if ( taskid == jtask ){
       printf("\n");
       for(itask=0;itask<ntasks;itask++){
         buffsum=0.0;
         for(i=0;i<buffsize;i++){
           buffsum=buffsum+recvbuff[itask][i];
         }
         printf("Task %d : Sum of vector received from %d -> %e \n",
               taskid,itask,buffsum);
       }
     }
   }

   MPI::COMM_WORLD.Barrier();

   if(taskid==0){
     printf("\n");
     printf("##########################################################\n\n");
     printf(" Communication time : %f seconds\n\n",totaltime);
     printf("##########################################################\n\n");
   }

   /*===============================================================*/
   /* Free the allocated memory.                                    */
     delete [] recvbuff[0];
     delete [] recvbuff;
     delete [] sendbuff;

   /*===============================================================*/
   /* MPI finalisation.                                             */
   MPI::Finalize();

}
