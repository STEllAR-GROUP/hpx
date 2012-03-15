#include <omp.h>

int main()
{
    #ifdef _OPENMP
        return 0; 
    #else
        #error OpenMP support not found.
    #endif
}

