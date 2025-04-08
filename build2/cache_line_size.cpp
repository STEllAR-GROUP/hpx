
    #include <iostream>
    #include <new>
    int main()
    {
#if defined(HPX_HAVE_CXX17_HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE)
        std::cout << std::hardware_destructive_interference_size;
#else
#if defined(__s390__) || defined(__s390x__)
        std::cout << 256;    // assume 256 byte cache-line size
#elif defined(powerpc) || defined(__powerpc__) || defined(__ppc__)
        std::cout << 128;    // assume 128 byte cache-line size
#else
        std::cout << 64;     // assume 64 byte cache-line size
#endif
#endif
    }
