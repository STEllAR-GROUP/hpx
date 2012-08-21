
//  Copyright (c) 2012 Andrew Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <iomanip>

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <oclm/oclm.hpp>
#include <oclm/util/high_resolution_timer.hpp>
#include <oclm/kernelUtil/readCL.hpp>

void runFunction(const oclm::program& p, const int& numThreads, const std::string& function)
{
        // Select a kernel
        oclm::kernel k(p, function);

        // build a function object out of the kernel
        oclm::function f = k[oclm::global(numThreads), oclm::local(1)];

        std::vector<int> A(numThreads, 1);
        std::vector<int> B(numThreads, 2);
        std::vector<int> C(numThreads, 0);
        
        // create a command queue with a device type and a platform ... context and
        // platform etc is selected in the background ... this will be managed as
        // global state
        oclm::command_queue queue(oclm::device::default_, false);


        // asynchronously fire the opencl function on the command queue, the
        // std::vector's will get copied back and forth transparantly, policy classes
        // to come ...
        oclm::event e1 = async(queue, f(A, B, C));

        // wait until everything is completed ...
        e1.get();
}

void foutWrite(const std::vector<double> timeLocal, const std::vector<double> timeGlobal, const std::vector<double> timeVecAdd, int iters, std::ofstream& fout, int precision, int width)
{
    for (int i = 0; i < iters; i++)
    {
        int numThreads = (int)std::pow((double)2, (double)i);
        fout << numThreads << " thread execution time:\n";
        fout << std::setw(width) << std::setiosflags(std::ios::fixed) << std::setprecision(precision) << timeLocal[i];
        fout << std::setw(width) << std::setiosflags(std::ios::fixed) << std::setprecision(precision) << timeGlobal[i];
        fout << std::setw(width) << std::setiosflags(std::ios::fixed) << std::setprecision(precision) << timeVecAdd[i];
        fout << std::endl;
        for (int i = 0; i < width * 3; i++)
            fout << "=";
        fout << std::endl;
    }
}

int hpx_main()
{
    oclm::get_platform();
    //Open oclm/kernelUtil/kernels/speedTest.cl
    const std::string src = readCLFile("/kernels/speedTest.cl");
    // create a program from source ... possibly a vector of sources ...
    oclm::program p(src);
    
    std::cout << "Initializing Part 1 of the speed test: basic memory read/write and arithmetic operations.\n";
    std::vector<double> timeLocal;
    std::vector<double> timeGlobal;
    std::vector<double> timeVecAdd;
    const int iters = 19;
    for (int i = 0; i < iters; i++)
    {
        int numThreads = (int)std::pow((double)2, (double)i);
        std::string function = "assignLocal";
        std::cout << "Running basic tests with " << numThreads << " threads.\n";

        oclm::util::high_resolution_timer t;

        runFunction(p, numThreads, function);
        timeLocal.push_back(t.elapsed());
        
        function = "assignGlobal";
        t.restart();
        
        runFunction(p, numThreads, function);
        timeGlobal.push_back(t.elapsed());

        
        t.restart();
        
        function = "vecAdd";
        runFunction(p, numThreads, function);
        timeVecAdd.push_back(t.elapsed());
    }
    std::ofstream fout;
    fout.open("logSimple.txt");
    fout << "BASIC TESTS\n";
    const int width = 40;
    const int precision = 16;
    fout << "Basic (1x operation) tests to determine the I/O speed of the OpenCL device.\n";
    for (int i = 1; i < width * 3; i++)
        fout << "="; 
    fout << std::setiosflags(std::ios::right);
    fout << std::resetiosflags(std::ios::left);
    fout << std::endl;
    fout << std::setw(width) << "Local 1x Read, Global 1x Write";
    fout << std::setw(width) << "Global 1x Read/Write";
    fout << std::setw(width) << "Global 2x Read/1x Write";
    fout << std::endl;
    fout <<  "Function name in kernels/speedTest.cl:" << std::endl;
    fout << std::setw(width) << "assignLocal";
    fout << std::setw(width) << "assignGlobal";
    fout << std::setw(width) << "vecAdd";
    fout << std::endl;
    for (int i = 0; i < width * 3; i++)
        fout << "=";
    fout << std::endl;
    foutWrite(timeLocal, timeGlobal, timeVecAdd, iters, fout, precision, width);
    fout.close();
    std::cout << "Basic performance information output to logSimple.txt\n";
    
    timeLocal.clear();
    timeGlobal.clear();
    timeVecAdd.clear();

    std::cout << "Initializing Part 2 of the speed test: complex (1024x kernel repetitions) memory read/write and arithmetic operations.\n";
    for (int i = 0; i < iters; i++)
    {
        int numThreads = (int)std::pow((double)2, (double)i);
        std::string function = "assignLocalMany";
        std::cout << "Running complex tests with " << numThreads << " threads.\n";

        oclm::util::high_resolution_timer t;
        
        runFunction(p, numThreads, function);
        timeLocal.push_back(t.elapsed());
        
        function = "assignGlobalMany";
        t.restart();
        
        runFunction(p, numThreads, function);
        timeGlobal.push_back(t.elapsed());
        
        t.restart();
        
        function = "vecAddMany";
        runFunction(p, numThreads, function);
        timeVecAdd.push_back(t.elapsed());
    }
    fout.open("logComplex.txt");
    fout << "COMPLEX TESTS\n";
    fout << "Complex (1024x operation) tests to determine the memory read/write speed of the OpenCL device.\n";
    for (int i = 1; i < 90; i++)
        fout << "="; 
    fout << std::setiosflags(std::ios::right);
    fout << std::resetiosflags(std::ios::left);
    fout << std::endl;
    fout << std::setw(width) << "Local 1024x Read, Global 1024x Write";
    fout << std::setw(width) << "Global 1024x Read/Write";
    fout << std::setw(width) << "Global 2048x Read/1024x Write";
    fout << std::endl;
    fout <<  "Function name in kernels/speedTest.cl:" << std::endl;
    fout << std::setw(width) << "assignLocalMany";
    fout << std::setw(width) << "assignGlobalMany";
    fout << std::setw(width) << "vecAddMany";
    fout << std::endl;
    for (int i = 0; i < width * 3; i++)
        fout << "=";
    fout << std::endl;
    foutWrite(timeLocal, timeGlobal, timeVecAdd, iters, fout, precision, width);
    fout.close();
    std::cout << "Basic performance information output to logComplex.txt\n";
    std::cout << "Note: The basic tests are more to check the I/O rate of your OpenCL device. However, the complex tests can be very informative about the nature of memory access.\n"
        << "The first three functions in the .cl file are the basic tests, while the remaining three are the complex tests. Note how much longer it takes to access global memory rather than local memory.\n";
    //check if we're using Windows

    return hpx::finalize(); // Handles HPX shutdown
}

int main()
{
    int code = hpx::init();    
    #if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)
        //if so, pause because the console usually closes at the end of execution
        system("Pause");
    #endif
    return code;
}
