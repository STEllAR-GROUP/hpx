//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/memory_block.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <boost/format.hpp>

#include <algorithm>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::naming::get_locality_from_id;

using hpx::components::memory_block;
using hpx::components::access_memory_block;

using hpx::actions::manage_object_action;
using hpx::actions::plain_action3;

using hpx::lcos::eager_future;
using hpx::lcos::wait;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;
using hpx::endl;

using hpx::find_here;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
std::size_t partition(boost::uint32_t* data, std::size_t begin, std::size_t end)
{
    boost::uint32_t* const first = data + begin;
    boost::uint32_t* const last = first + (end - begin);

    std::less<boost::uint32_t> const less_; 

    boost::uint32_t* const middle =
        std::partition(first, last, std::bind2nd(less_, *first));

    return middle - data;
}

void quicksort(id_type const& d, std::size_t begin, std::size_t end);

typedef plain_action3<
    id_type const&,
    std::size_t,
    std::size_t,
    &quicksort
> quicksort_action;

HPX_REGISTER_PLAIN_ACTION(quicksort_action);

typedef eager_future<quicksort_action> quicksort_future;

void quicksort(id_type const& d, std::size_t begin, std::size_t end)
{
    if (begin != end)
    {
        memory_block mb(d);
        access_memory_block<boost::uint32_t> data(mb.get());

        std::size_t middle = partition(data.get_ptr(), begin, end);
        id_type prefix = get_locality_from_id(d);

        // Always spawn the larger part in a new thread.
        if (2 * middle < end - begin)
        {
            quicksort_future n(prefix, d, (std::max)(begin + 1, middle), end);
            quicksort(d, begin, middle);
            wait(n);
        }

        else
        {
            quicksort_future n(prefix, d, begin, middle);
            quicksort(d, (std::max)(begin + 1, middle), end);
            wait(n);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        // Retrieve the command line options. 
        std::size_t const elements = vm["elements"].as<std::size_t>();
        std::size_t seed = vm["seed"].as<std::size_t>();

        // If the specified seed is 0, then we pick a random seed.
        if (!seed)
            seed = std::size_t(std::time(0));

        // Seed the C standard libraries random number facilities.
        std::srand(seed);

        cout << "Seed: " << seed << endl; 

        // Create a memory block.
        memory_block mb;

        mb.create<boost::uint32_t, boost::uint8_t>(find_here(), elements);
        access_memory_block<boost::uint32_t> data(mb.get());

//        int* it = data.get_ptr();
//        int* end = data.get_ptr() + elements;

//        for (; it < end; ++it)
//            std::cout << *it << "\n";

        std::generate(data.get_ptr(), data.get_ptr() + elements, std::rand);

        high_resolution_timer t;
    
        quicksort_future n(find_here(), mb.get_gid(), 0, elements);

        wait(n);

        char const* const fmt = "sorted %1% items in %2% [s]\n";

        cout << (boost::format(fmt) % elements % t.elapsed()) << flush;

//        it = data.get_ptr();
//        end = data.get_ptr() + elements;

//        for (; it < end; ++it)
//            std::cout << *it << "\n";

        mb.free();
    }

    // initiate shutdown of the runtime systems on all localities
    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("elements", value<std::size_t>()->default_value(1024),
            "the number of elements to generate and sort")
        ("seed", value<std::size_t>()->default_value(0),
            "the seed for the pseudo random number generator (if 0, a seed "
            "is choosen based on the current system time)")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

