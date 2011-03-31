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
#include <boost/lockfree/fifo.hpp>
#include <boost/program_options.hpp>

#include <hpx/util/high_resolution_timer.hpp>

volatile bool done = false;

inline void
producer(std::size_t count, std::size_t size, boost::lockfree::fifo<int*>* bin)
{
    for (register std::size_t i = 0; i != count; ++i)
        bin->enqueue(reinterpret_cast<int*>(operator new (size)));
}

inline void consumer(boost::lockfree::fifo<int*>* bin)
{
    int* value = 0;

    while (!done) {
        while (bin->dequeue(&value))
            delete value;
    }

    while (bin->dequeue(&value))
        delete value; 
}

int main(int argc, char** argv)
{
    using boost::program_options::variables_map;
    using boost::program_options::options_description;
    using boost::program_options::value;
    using boost::program_options::store;
    using boost::program_options::command_line_parser;
    using boost::program_options::notify;

    using boost::lockfree::fifo;

    variables_map vm;

    options_description
        desc_cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
   
    std::size_t threads, count, object_size;
 
    desc_cmdline.add_options()
        ("help,h", "print out program usage (this message)")
        ("threads,t", value<std::size_t>(&threads)->default_value(1 << 2), 
         "the number of worker threads")
        ("count,c", value<std::size_t>(&count)->default_value(1 << 20), 
         "the number of objects to allocate per producer thread") 
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
    // make the bins (which, for this test, are lockfree fifos)
    fifo<int*>* bins = 0;
        
    bins = reinterpret_cast<fifo<int*>*>   
        (::calloc(threads, sizeof(fifo<int*>)));

    for (std::size_t i = 0; i != threads; ++i)
        new (&bins[i]) fifo<int*>(count); 

    ///////////////////////////////////////////////////////////////////////////
    // run the test
    boost::thread_group producers, consumers;

    hpx::util::high_resolution_timer t;

    for (std::size_t i = 0; i != threads; ++i)
        producers.add_thread(new boost::thread
            (producer, count, object_size, &bins[i]));
 
    for (std::size_t i = 0; i != threads; ++i)
        consumers.add_thread(new boost::thread(consumer, &bins[i]));

    producers.join_all();
    done = true;

    consumers.join_all();

    double elapsed = t.elapsed();

    ///////////////////////////////////////////////////////////////////////////
    // free the fifo array
    ::free(bins);

    ///////////////////////////////////////////////////////////////////////////
    // output results
    std::cout
        << "(((threads " << threads << ") "
             "(count " << count << ") "
             "(object-size " << object_size << ")) "
            "((allocation-total " << (threads * count * object_size) << ") "
             "(wall-time " << elapsed << ")))" << std::endl;
}

