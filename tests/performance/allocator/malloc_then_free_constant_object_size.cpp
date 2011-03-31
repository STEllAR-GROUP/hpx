////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Bryce Lelbach 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>

#include <iostream>

#include <boost/config.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/program_options.hpp>

#include <hpx/util/high_resolution_timer.hpp>

inline void
worker(std::size_t count, std::size_t size, void*** bin,
       boost::barrier* bar)
{
    for (register std::size_t i = 0; i != count; ++i)
        (*bin)[i] = ::malloc(size);

    bar->wait(); 

    for (register std::size_t i = 0; i != count; ++i)
        ::free((*bin)[i]);
}

int main(int argc, char** argv)
{
    using boost::program_options::variables_map;
    using boost::program_options::options_description;
    using boost::program_options::value;
    using boost::program_options::store;
    using boost::program_options::command_line_parser;
    using boost::program_options::notify;

    variables_map vm;

    options_description
        desc_cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
   
    std::size_t threads, count, object_size;
 
    desc_cmdline.add_options()
        ("help,h", "print out program usage (this message)")
        ("threads,t", value<std::size_t>(&threads)->default_value(1 << 2), 
         "the number of worker threads") 
        ("count,c", value<std::size_t>(&count)->default_value(1 << 20), 
         "the number of objects to allocate per thread") 
        ("object-size,s", value<std::size_t>
          (&object_size)->default_value(1 << 6), 
         "the size, in bytes, of each object") 
    ;

    store(command_line_parser(argc, argv).options(desc_cmdline).run(), vm);

    notify(vm);

    ///////////////////////////////////////////////////////////////////////////
    // print help screen
    if (vm.count("help"))
    {
        std::cout << desc_cmdline;
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // make the bins
    void*** bins = new void**[threads];

    for (std::size_t i = 0; i != threads; ++i)
        bins[i] = new void*[count]; 

    ///////////////////////////////////////////////////////////////////////////
    // run the test
    boost::thread_group tg;
    boost::barrier bar(threads + 1);

    hpx::util::high_resolution_timer t;

    for (std::size_t i = 0; i != threads; ++i)
        tg.add_thread
            (new boost::thread(worker, count, object_size, &bins[i], &bar));

    bar.wait();

    tg.join_all();

    double elapsed = t.elapsed();

    ///////////////////////////////////////////////////////////////////////////
    // free the bins
    for (std::size_t i = 0; i != threads; ++i)
        delete[] bins[i];

    delete[] bins;

    ///////////////////////////////////////////////////////////////////////////
    // output results
    std::cout
        << "(((threads " << threads << ") "
             "(count " << count << ") "
             "(object-size " << object_size << ")) "
            "((allocation-total " << (threads * count * object_size) << ") "
             "(wall-time " << elapsed << ")))" << std::endl;
}

