#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

enum bool
{
    false = 0,
    true = 1
};

//mpi initilazations
int rank, numprocs;

float *MPI_Map_Func(float *arr, int size, float (*func)(float))
{
    int i;
    //Calculate which implementation will be used
    int status = 0;
    if (rank == 0){
        if (size % numprocs == 0){
            status = 1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *) &status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //Scatter&Gather
    if (status == 1)
    {
        int chunkSize;
        if (rank == 0)
        {
            chunkSize = size / numprocs;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast((void *) &chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        float *chunkData = (float *)calloc(chunkSize, sizeof(float));

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter((void *)&arr[0], chunkSize, MPI_FLOAT, &chunkData[0], chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

        //Calculations
        for (i = 0; i < chunkSize; i++)
        {
            chunkData[i] = func(chunkData[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(&chunkData[0], chunkSize, MPI_FLOAT, &arr[0], chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

        free(chunkData);
        

    }
    //Send&Recv
    else
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            int divider = size / numprocs;
            int excess = size % numprocs;
            int masterSize = divider;

            if (rank < excess)
            {
                masterSize++;
            }

            int dataCursor = masterSize;
            //Master Calculation
            for (i = 0; i < masterSize; i++)
            {
                arr[i] = func(arr[i]);
            }
            //Send Data
            for (i = 1; i < numprocs; i++)
            {
                int sendDataSize = divider;

                if (i < excess)
                {
                    sendDataSize++;
                }
                MPI_Send((void *) &sendDataSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                MPI_Send((void *) &arr[dataCursor], sendDataSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD);

                dataCursor += sendDataSize;
            }
            //Recv Data
            dataCursor = masterSize;
            for (i = 1; i < numprocs; i++)
            {
                int recvDataSize = divider;

                if (i < excess)
                {
                    recvDataSize++;
                }
                MPI_Recv((void *) &arr[dataCursor], recvDataSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                dataCursor += recvDataSize;
            }
        }
        else 
        {
            int recvSize;

            MPI_Recv((void *) &recvSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            float *chunkData = (float *)calloc(recvSize, sizeof(float));

            MPI_Recv((void *) &chunkData[0], recvSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (i = 0; i < recvSize; i++)
            {
                chunkData[i] = func(chunkData[i]);
            }

            MPI_Send((void *) &chunkData[0], recvSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);

            free(chunkData);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return &arr[0];
}

float MPI_Fold_Func(float *arr, int size, float initial_value, float (*func)(float, float))
{
    int i;
    //Calculate which implementation will be used
    int status = 0;
    if (rank == 0){
        if (size % numprocs == 0){
            status = 1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *) &status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void *) &initial_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //Scatter&Gather
    if (status == 1)
    {
        int chunkSize;
        if (rank == 0)
        {
            chunkSize = size / numprocs;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast((void *) &chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        float *chunkData = (float *)calloc(chunkSize, sizeof(float));

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter((void *)&arr[0], chunkSize, MPI_FLOAT, &chunkData[0], chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

        //Calculations
        for (i = 0; i < chunkSize; i++)
        {
            initial_value = func(initial_value, chunkData[i]);
        }
        float *calculated_values = NULL;
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            calculated_values = (float *)calloc(numprocs, sizeof(float));
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Gather(&initial_value, 1, MPI_FLOAT, &calculated_values[0], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        //Calculate local values
        if (rank == 0)
        {
            initial_value = calculated_values[0];
            for (i = 1; i < numprocs; i++)
            {
                initial_value = func(initial_value, calculated_values[i]);
            }

            free(calculated_values);
        }
        free(chunkData);
        

    }
    //Send&Recv
    else
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            int divider = size / numprocs;
            int excess = size % numprocs;
            int masterSize = divider;

            if (rank < excess)
            {
                masterSize++;
            }

            int dataCursor = masterSize;
            //Master Calculation
            for (i = 0; i < masterSize; i++)
            {
                initial_value = func(initial_value, arr[i]);
            }
            //Send Data
            for (i = 1; i < numprocs; i++)
            {
                int sendDataSize = divider;

                if (i < excess)
                {
                    sendDataSize++;
                }
                MPI_Send((void *) &sendDataSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                MPI_Send((void *) &arr[dataCursor], sendDataSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD);

                dataCursor += sendDataSize;
            }
        }
        else 
        {
            int recvSize;

            MPI_Recv((void *) &recvSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            float *chunkData = (float *)calloc(recvSize, sizeof(float));

            MPI_Recv((void *) &chunkData[0], recvSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (i = 0; i < recvSize; i++)
            {
                initial_value = func(initial_value, chunkData[i]);
            }

            free(chunkData);
        }
        //Recv local calculated values
        float *calculated_values = NULL;

        if (rank == 0)
        {
            calculated_values = (float *)calloc(numprocs, sizeof(float));
        }
        
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Gather(&initial_value, 1, MPI_FLOAT, &calculated_values[0], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        //Calculate local values
        if (rank == 0)
        {
            initial_value = calculated_values[0];
            for (i = 1; i < numprocs; i++)
            {
                initial_value = func(initial_value, calculated_values[i]);
            }

            free(calculated_values);
        }


    }

    MPI_Barrier(MPI_COMM_WORLD);
    return initial_value;
}

float *MPI_Filter_Func(float *arr, int size, enum bool (*pred)(float))
{
    int i;
    //Calculate which implementation will be used
    int status = 0;
    if (rank == 0){
        if (size % numprocs == 0){
            status = 1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *) &status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //Scatter&Gather
    if (status == 1)
    {
        int chunkSize;
        if (rank == 0)
        {
            chunkSize = size / numprocs;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast((void *) &chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        float *chunkData = (float *)calloc(chunkSize, sizeof(float));

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter((void *)&arr[0], chunkSize, MPI_FLOAT, &chunkData[0], chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

        //Calculations
        for (i = 0; i < chunkSize; i++)
        {
            chunkData[i] = pred(chunkData[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(&chunkData[0], chunkSize, MPI_FLOAT, &arr[0], chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

        free(chunkData);
        

    }
    //Send&Recv
    else
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            int divider = size / numprocs;
            int excess = size % numprocs;
            int masterSize = divider;

            if (rank < excess)
            {
                masterSize++;
            }

            int dataCursor = masterSize;
            //Master Calculation
            for (i = 0; i < masterSize; i++)
            {
                arr[i] = pred(arr[i]);
            }
            //Send Data
            for (i = 1; i < numprocs; i++)
            {
                int sendDataSize = divider;

                if (i < excess)
                {
                    sendDataSize++;
                }
                MPI_Send((void *) &sendDataSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                MPI_Send((void *) &arr[dataCursor], sendDataSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD);

                dataCursor += sendDataSize;
            }
            //Recv Data
            dataCursor = masterSize;
            for (i = 1; i < numprocs; i++)
            {
                int recvDataSize = divider;

                if (i < excess)
                {
                    recvDataSize++;
                }
                MPI_Recv((void *) &arr[dataCursor], recvDataSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                dataCursor += recvDataSize;
            }
        }
        else 
        {
            int recvSize;

            MPI_Recv((void *) &recvSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            float *chunkData = (float *)calloc(recvSize, sizeof(float));

            MPI_Recv((void *) &chunkData[0], recvSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (i = 0; i < recvSize; i++)
            {
                chunkData[i] = pred(chunkData[i]);
            }

            MPI_Send((void *) &chunkData[0], recvSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);

            free(chunkData);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return &arr[0];
}
