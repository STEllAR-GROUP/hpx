////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/atomic.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::actions::plain_action1;

using hpx::lcos::eager_future;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

boost::atomic<std::size_t> ctor_count;
boost::atomic<std::size_t> copy_ctor_count;
boost::atomic<std::size_t> assignment_count;
boost::atomic<std::size_t> dtor_count;

struct object
{
    object()
    {
        ++ctor_count;
    }

    object(object const& other)
    {
        ++copy_ctor_count;
    }

    ~object()
    {
        ++dtor_count;
    }

    object& operator=(object const& other)
    {
        ++assignment_count;
        return *this;
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // no-op
    } 
};

///////////////////////////////////////////////////////////////////////////////
void pass_object(
    object const& obj  
    )
{
    std::cout << "pass_object: object(" << &obj << ")\n";
}

typedef plain_action1<
    // arguments
    object const&  
    // function
  , pass_object
> pass_object_action;

HPX_REGISTER_PLAIN_ACTION(pass_object_action);

typedef eager_future<pass_object_action> pass_object_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        const id_type prefix = find_here(); 

        object obj;

        std::cout << "hpx_main: object(" << &obj << ")\n";

        pass_object_future f(prefix, obj);

        f.get();
    }

    std::cout << "ctor_count:       " << ctor_count.load() << "\n"
              << "copy_ctor_count:  " << copy_ctor_count.load() << "\n"
              << "assignment_count: " << assignment_count.load() << "\n"
              << "dtor_count:       " << dtor_count.load() << "\n";

    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

