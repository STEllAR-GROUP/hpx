//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/bind.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/lcos/object_semaphore.hpp>

typedef hpx::lcos::object_semaphore<int> object_semaphore_type;

///////////////////////////////////////////////////////////////////////////////
void worker(object_semaphore_type os)
{
    try
    {
        int value = os.wait_sync(); // retrieve one value, might throw
        std::cout << value << std::endl;
    }

    catch (hpx::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
}

void break_semaphore(object_semaphore_type os)
{
    os.abort_pending();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
        // create a new object semaphore 
        object_semaphore_type os
            (object_semaphore_type::create_sync(hpx::find_here()));
    
        // create some threads waiting to pull elements from the queue
        for (int i = 0; i < 5; ++i)
            hpx::applier::register_work(boost::bind(&worker, os));
    
        // add some values to the queue
        for (int i = 0; i < 5; ++i)
            os.signal_sync(i);
    
        // create some threads waiting to pull elements from the queue
        for (int i = 0; i < 5; ++i)
            hpx::applier::register_work(boost::bind(&worker, os));
    
        // abort all pending workers
        hpx::applier::register_work(boost::bind(&break_semaphore, os));
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

