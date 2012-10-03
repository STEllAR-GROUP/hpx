//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>
#include <hpx/components/dataflow/async_dataflow_wait.hpp>

#include <hpx/components/remote_object/object.hpp>
#include <hpx/components/remote_object/new.hpp>
#include <hpx/components/dataflow/dataflow_object.hpp>

#include <tests/regressions/actions/components/movable_objects.hpp>

#include <hpx/util/lightweight_test.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::test::movable_object;
using hpx::test::non_movable_object;

using hpx::components::object;
using hpx::components::dataflow_object;
using hpx::components::new_;

struct foo : boost::noncopyable {};

template <typename Object>
struct movable_functor
{
    typedef std::size_t result_type;

    Object obj;

    movable_functor(Object const & o) : obj(o) {}

    movable_functor() {}

    movable_functor(movable_functor const & other)
        : obj(other.obj)
    {}

    movable_functor(BOOST_RV_REF(movable_functor) other)
        : obj(boost::move(other.obj))
    {}

    movable_functor& operator=(BOOST_COPY_ASSIGN_REF(movable_functor) other)
    {
        obj = other.obj;
        return *this;
    }

    movable_functor& operator=(BOOST_RV_REF(movable_functor) other)
    {
        obj = boost::move(other.obj);
        return *this;
    }

    result_type operator()(foo&) const
    {
        return obj.get_count();
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & obj;
    }

    private:
        BOOST_COPYABLE_AND_MOVABLE(movable_functor)
};

template <typename Object>
struct non_movable_functor
{
    typedef std::size_t result_type;

    Object obj;

    non_movable_functor(Object const & o) : obj(o) {}

    non_movable_functor() {}

    non_movable_functor(non_movable_functor const & other)
        : obj(other.obj)
    {}

    non_movable_functor& operator=(non_movable_functor const & other)
    {
        obj = other.obj;
        return *this;
    }

    result_type operator()(foo&) const
    {
        return obj.get_count();
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & obj;
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    std::vector<id_type> localities = hpx::find_all_localities();

    BOOST_FOREACH(id_type id, localities)
    {
        bool is_local = (id == hpx::find_here()) ? true : false;
        {
            object<foo> f = new_<foo>(id).get();

            HPX_TEST_EQ((f <= movable_functor<movable_object>()).get(), 0u);
            HPX_TEST_EQ((f <= movable_functor<non_movable_object>()).get(), is_local ? 6u : 6u);

            HPX_TEST_EQ((f <= non_movable_functor<movable_object>()).get(), is_local ? 6u : 6u);
            HPX_TEST_EQ((f <= non_movable_functor<non_movable_object>()).get(), is_local ? 6u : 6u);
        }
        {
            dataflow_object<foo> f(new_<foo>(id).get());

            HPX_TEST_EQ(f.apply(movable_functor<movable_object>()).get_future().get(), 1u);
            HPX_TEST_EQ(f.apply(movable_functor<non_movable_object>()).get_future().get(), is_local ? 5u : 5u);

            HPX_TEST_EQ(f.apply(non_movable_functor<movable_object>()).get_future().get(), is_local ? 5u : 5u);
            HPX_TEST_EQ(f.apply(non_movable_functor<non_movable_object>()).get_future().get(), is_local ? 5u : 5u);
        }
    }

    finalize();

    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

