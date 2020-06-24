#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 

int main(int argc, char **argv) 
{
    if (argc != 2)
    {
        printf("Please only enter number of experiments.\n");
        return 0;
    }
    int expNum = atoi(argv[1]);
    if (expNum <= 0)
    {
        printf("Please enter positive integer for number of experiments\n");
        return 0;
    }
    srand(time(NULL));
    int i;
    //timer start
    clock_t begin = clock();
    int inCount = 0;
    for (i = 0; i < expNum; i++)
    {
        float x = (float) rand() / (float) RAND_MAX;
        float y = (float) rand() / (float) RAND_MAX;
        if ((float)sqrt((x * x) + (y * y)) <= 1)
        {
            inCount++;
        }
    }
    float pi = (float) inCount / (float) expNum * 4.00;
    clock_t end = clock();
    double timeSpent = (double)(end - begin) * 1000.0 / CLOCKS_PER_SEC;
    printf("---------main-serial---------\nNumber of experiment: %d\nEstimated pi: %f\nTime spent in ms: %f\n-----------------------------\n", expNum, pi, timeSpent);
    return 0;
}