#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "helper.h"
#include <time.h> 


float square(float num)
{
    return num * num;
}

float sum(float num1, float num2)
{
    return num1 + num2;
}

enum bool isSqrtInCircle(float num)
{
    if (sqrt(num) <= 1)
    {
        return true;
    }
    return false;
}
  
int main(int argc, char **argv) 
{  
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i;
    int expCount = 0;
    int status = 0;
    float *random_variables;
    float *square_root_variables;
    clock_t begin;

    if (rank == 0)
    {
        if (argc != 2)
        {
            printf("Please only enter number of experiments.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        expCount = atoi(argv[1]);
        if (expCount <= 0)
        {   
            printf("Please enter positive integer for number of experiments\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        if ((expCount * 2) % numprocs == 0)
        {
            status = 1;
        }
        random_variables = (float *)calloc(expCount * 2, sizeof(float));
        square_root_variables = (float *)calloc(expCount, sizeof(float));
        //Start timer
        begin = clock();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *) &expCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *) &status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //init functions and rand
    float (*sum_func)(float, float);
    sum_func = &sum;
    float (*square_func)(float);
    square_func = &square;
    enum bool (*sqrtInCircle_func)(float);
    sqrtInCircle_func = &isSqrtInCircle;
    srand(time(NULL) + rank);
    int exp_variable_count = expCount * 2;
    MPI_Barrier(MPI_COMM_WORLD);
    //---------Create random data parallel and init needed variables------------

    float *chunk_of_random_variables;
    int chunk_size;
    int divider;
    int excess; 

    if (status == 1)
    {
        chunk_size = exp_variable_count / numprocs;
    }
    else
    {
        divider = exp_variable_count / numprocs;
        excess = exp_variable_count % numprocs;
        chunk_size = divider;
        if (rank < excess)
        {
            chunk_size++;
        }
    }

    chunk_of_random_variables = (float *)calloc(chunk_size, sizeof(float));
    for (i = 0; i < chunk_size; i++)
    {
        chunk_of_random_variables[i] = (float) rand() / (float) RAND_MAX;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (status == 1)
    {
        MPI_Gather(&chunk_of_random_variables[0], chunk_size, MPI_FLOAT, &random_variables[0], chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        if (rank == 0)
        {
            //Recv Data
            for (i = 0; i < chunk_size; i++)
            {
                random_variables[i] = chunk_of_random_variables[i];
            }

            int dataCursor = chunk_size;
            for (i = 1; i < numprocs; i++)
            {
                int recvDataSize = divider;

                if (i < excess)
                {
                    recvDataSize++;
                }

                MPI_Recv((void *) &random_variables[dataCursor], recvDataSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                dataCursor += recvDataSize;
            }
        }
        else
        {
            MPI_Send((void *) &chunk_of_random_variables[0], chunk_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }
    //------------Random data created------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    free(chunk_of_random_variables);
    //Call map function to calculate square of random variables
    random_variables = MPI_Map_Func(random_variables, exp_variable_count, square_func);
    MPI_Barrier(MPI_COMM_WORLD);
    //-------------Square values adding-----------------------
    divider = expCount / numprocs;
    excess = expCount % numprocs;
    int new_chunk_size = divider;
    if (rank < excess)
    {
        new_chunk_size++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        for (i = 0; i < new_chunk_size * 2; i += 2)
        {
            square_root_variables[i / 2] = random_variables[i] + random_variables[i + 1];
        }
        int dataCursor = new_chunk_size * 2;

        for (i = 1; i < numprocs; i++)
        {
            int sendDataSize = divider;

            if (i < excess)
            {
                sendDataSize++;
            }

            sendDataSize = sendDataSize * 2;

            MPI_Send((void *) &random_variables[dataCursor], sendDataSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD);

            dataCursor += sendDataSize;
        }

        dataCursor = new_chunk_size;
        for (i = 1; i < numprocs; i++)
        {
            int recvDataSize = divider;

            if (i < excess)
            {
                recvDataSize++;
            }

            MPI_Recv((void *) &square_root_variables[dataCursor], recvDataSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            dataCursor += recvDataSize;
        }
        free(random_variables);
    }
    else
    {
        int recvDataSize = new_chunk_size * 2;
        float *recvData = (float *)calloc(recvDataSize, sizeof(float));
        float *sendData = (float *)calloc(new_chunk_size, sizeof(float));

        MPI_Recv((void *) &recvData[0], recvDataSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (i = 0; i < recvDataSize; i += 2)
        {
            sendData[i / 2] = recvData[i] + recvData[i + 1];
        }

        MPI_Send((void *) &sendData[0], new_chunk_size, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);

        free(recvData);
        free(sendData);
    }
    MPI_Barrier(MPI_COMM_WORLD);  
    //-------------Square values added-----------------------
    //MPI_Filter_Func to detect if the value is in the circle or not
    square_root_variables = MPI_Filter_Func(square_root_variables, expCount, sqrtInCircle_func);
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Fold_Func add values to calculate pi
    float pi = MPI_Fold_Func(square_root_variables, expCount, 0, sum_func) / (float) expCount * 4.00;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        //timer start
        clock_t end = clock();
        double timeSpent = (double)(end - begin) * 1000.0 / CLOCKS_PER_SEC;
        free(square_root_variables);

        printf("----------main-parallel----------\nNumber of experiment: %d\nEstimated pi: %f\nTime spent in ms: %f\n---------------------------------\n", expCount, pi, timeSpent);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}