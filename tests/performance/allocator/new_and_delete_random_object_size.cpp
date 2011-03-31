////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Bryce Lelbach 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>

#include <vector>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/config.hpp>
#include <boost/thread/thread.hpp>
#include <boost/lockfree/fifo.hpp>
#include <boost/program_options.hpp>

#include <hpx/util/high_resolution_timer.hpp>

volatile bool done = false;

inline void
producer(std::vector<boost::uint32_t>* size, boost::lockfree::fifo<int*>* bin)
{
    const std::size_t elements = size->size();

    for (register std::size_t i = 0; i != elements; ++i)
        bin->enqueue(reinterpret_cast<int*>(operator new ((*size)[i])));
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

    using boost::mt19937;
    using boost::uniform_int;

    using boost::lockfree::fifo;

    variables_map vm;

    options_description
        desc_cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
   
    boost::uint32_t seed, min_object_size, max_object_size; 
    std::size_t threads, alloc_limit;
 
    desc_cmdline.add_options()
        ("help,h", "print out program usage (this message)")
        ("threads,t", value<std::size_t>(&threads)->default_value(1 << 3), 
         "the number of worker threads")
        // older versions of Boost.Random generators don't provide the static
        // member default_seed, so we just use the literal integer here
        ("seed,s", value<boost::uint32_t>(&seed)->default_value(5489), 
         "the seed for the pseudo-random number generator") 
        ("min-object-size,m", value<boost::uint32_t>
          (&min_object_size)->default_value(1 << 2), 
         "the minimum size, in bytes, of each object") 
        ("max-object-size,M", value<boost::uint32_t>
          (&max_object_size)->default_value(1 << 12), 
         "the maximum size, in bytes, of each object") 
        ("allocation-limit,l", value<std::size_t>
          (&alloc_limit)->default_value(1 << 24), 
         "the maximum amount of memory to allocate per thread, in bytes") 
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
    // make the bins (which, for this test, are lockfree fifos) and generate
    // random sizes
    mt19937 rng(seed);
    uniform_int<boost::uint32_t> dist(min_object_size, max_object_size);
        
    fifo<int*>* bins = reinterpret_cast<fifo<int*>*>   
        (::calloc(threads, sizeof(fifo<int*>)));
    std::vector<boost::uint32_t>* sizes
        = new std::vector<boost::uint32_t>[threads];

    // we don't randomly generate the last size, instead, we just compute it
    // so that we don't go over alloc_limit. 
    const std::size_t upper_limit = alloc_limit - max_object_size;

    for (std::size_t i = 0; i != threads; ++i)
    {
        register std::size_t accum = 0;

        while (accum <= upper_limit)
        {
            sizes[i].push_back(dist(rng));
            accum += sizes[i].back();
        }

        sizes[i].push_back(static_cast<boost::uint32_t>(alloc_limit - accum));

        new (&bins[i]) fifo<int*>(sizes[i].size()); 
    }

    ///////////////////////////////////////////////////////////////////////////
    // run the test
    boost::thread_group producers, consumers;

    hpx::util::high_resolution_timer t;

    for (std::size_t i = 0; i != threads; ++i)
        producers.add_thread(new boost::thread(producer, &sizes[i], &bins[i]));
 
    for (std::size_t i = 0; i != threads; ++i)
        consumers.add_thread(new boost::thread(consumer, &bins[i]));

    producers.join_all();
    done = true;

    consumers.join_all();

    double elapsed = t.elapsed();

    ///////////////////////////////////////////////////////////////////////////
    // free the fifo array
    ::free(bins);
    ::free(sizes);

    ///////////////////////////////////////////////////////////////////////////
    // output results
    std::cout
        << "(((threads " << threads << ") "
             "(seed " << seed << ") "
             "(min-object-size " << min_object_size << ") "
             "(max-object-size " << max_object_size << ") "
             "(allocation-limit " << alloc_limit << ")) "
            "((allocation-total " << (threads * alloc_limit) << ") "
             "(wall-time " << elapsed << ")))" << std::endl;
}

