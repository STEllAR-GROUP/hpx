////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>
#include <boost/move/move.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::actions::plain_action1;

using hpx::lcos::eager_future;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

boost::atomic<std::size_t> copy_count;

struct object
{
    object() { }

    // Copy constructor.
    object(object const& other)
    {
        ++copy_count;
    }

    // Move constructor.
    object(BOOST_RV_REF(object) other) { }

    ~object() { }

    // Copy assignment.
    object& operator=(BOOST_COPY_ASSIGN_REF(object) other)
    {
        ++copy_count;
        return *this;
    }

    // Move assignment.
    object& operator=(BOOST_RV_REF(object) other)
    {
        return *this;
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int) { }

  private:
    BOOST_COPYABLE_AND_MOVABLE(object);
};

///////////////////////////////////////////////////////////////////////////////
void pass_object(object const& obj) {}

typedef plain_action1<
    // Arguments.
    object const&
    // Function.
  , pass_object
> pass_object_action;

HPX_REGISTER_PLAIN_ACTION(pass_object_action);

typedef eager_future<pass_object_action> pass_object_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        id_type const here = find_here();

        object obj;

        pass_object_future f(here, obj);

        f.get();

        HPX_TEST_EQ(1U, copy_count.load());
    }

    finalize();

    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

