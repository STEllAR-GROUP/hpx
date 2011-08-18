////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/lockfree/detail/freelist.hpp>
#include <boost/atomic.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/static.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::lockfree::caching_freelist;

using hpx::naming::id_type;

using hpx::applier::get_applier;

using hpx::actions::plain_result_action2;
using hpx::actions::plain_result_action3;

using hpx::lcos::base_lco_with_value;
using hpx::lcos::eager_future;
using hpx::lcos::future_value;

using hpx::util::high_resolution_timer;
using hpx::util::static_;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
typedef future_value<boost::uint64_t> future_value_type;
typedef caching_freelist<future_value<boost::uint64_t> > freelist_type;

struct freelist_tag {};

freelist_type& get_freelist()
{
    static_<freelist_type, freelist_tag> fl;
    return fl.get();
}

struct completion_tag {};

boost::atomic<bool>& get_completion_flag()
{
    static_<boost::atomic<bool>, completion_tag> f;
    return f.get();
}

///////////////////////////////////////////////////////////////////////////////
typedef eager_future<base_lco_with_value<boost::uint64_t>::get_value_action> 
    ackermann_peter_value_future;

HPX_REGISTER_ACTION_EX(
    base_lco_with_value<boost::uint64_t>::get_value_action,
    get_value_action_uint64_t);

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t ackermann_peter_subtract(
    id_type const& x 
  , boost::uint64_t c
);

typedef plain_result_action2<
    // result type
    boost::uint64_t
    // arguments
  , id_type const& 
  , boost::uint64_t
    // function
  , ackermann_peter_subtract
> ackermann_peter_subtract_action;

HPX_REGISTER_PLAIN_ACTION(ackermann_peter_subtract_action);

typedef eager_future<ackermann_peter_subtract_action>
    ackermann_peter_subtract_future;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t ackermann_peter(
    id_type const& prefix 
  , boost::uint64_t m
  , boost::uint64_t n
);

typedef plain_result_action3<
    // result type
    boost::uint64_t
    // arguments
  , id_type const&  
  , boost::uint64_t
  , boost::uint64_t
    // function
  , ackermann_peter
> ackermann_peter_action;

HPX_REGISTER_PLAIN_ACTION(ackermann_peter_action);

typedef eager_future<ackermann_peter_action> ackermann_peter_future;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t ackermann_peter_promised_n(
    id_type const& prefix 
  , boost::uint64_t m
  , id_type const& n 
);

typedef plain_result_action3<
    // result type
    boost::uint64_t
    // arguments
  , id_type const&  
  , boost::uint64_t
  , id_type const&  
    // function
  , ackermann_peter_promised_n
> ackermann_peter_promised_n_action;

HPX_REGISTER_PLAIN_ACTION(ackermann_peter_promised_n_action);

typedef eager_future<ackermann_peter_promised_n_action>
    ackermann_peter_promised_n_future;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t ackermann_peter_subtract(
    id_type const& x 
  , boost::uint64_t c
) {
    if (get_completion_flag().load())
        return -1;

    future_value_type* xf = get_freelist().allocate();
    new (xf) ackermann_peter_value_future(x);
    return xf->get() - c;
}     

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t ackermann_peter(
    id_type const& prefix 
  , boost::uint64_t m
  , boost::uint64_t n
) {
    if (get_completion_flag().load())
        return -1;

    if (HPX_UNLIKELY(m == 0))
        return n + 1;

    else
    {
        if (HPX_UNLIKELY(n == 0))
            return ackermann_peter(prefix, m - 1, 1);

        else
        {
            // No need to start m2 early in this function, as we don't have
            // to wait on m or n.
            future_value_type* m2 = get_freelist().allocate();
            new (m2) ackermann_peter_future(prefix, prefix, m, n - 1);

            future_value_type* m1 = get_freelist().allocate();
            new (m1) ackermann_peter_promised_n_future
                (prefix, prefix, m - 1, m2->get_gid());

            return m1->get();
        }
    } 
}

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t ackermann_peter_promised_n(
    id_type const& prefix 
  , boost::uint64_t m
  , id_type const& n 
) {
    if (get_completion_flag().load())
        return -1;

    if (HPX_UNLIKELY(m == 0))
    {
        future_value_type* nf = get_freelist().allocate();
        new (nf) ackermann_peter_value_future(n);
        return nf->get() + 1;
    }

    else
    {
        // get n
        future_value_type* nf = get_freelist().allocate();
        new (nf) ackermann_peter_value_future(n);

        // We have to wait on n. n == 0 is the less frequent case, so start 
        // the future for m2 (and dependencies) now, allowing it to execute
        // while we wait for n.
        future_value_type* n1 = get_freelist().allocate();
        new (n1) ackermann_peter_subtract_future(prefix, n, 1);

        future_value_type* m2 = get_freelist().allocate();
        new (m2) ackermann_peter_promised_n_future
            (prefix, prefix, m, n1->get_gid());

        if (HPX_UNLIKELY(nf->get() == 0))
            return ackermann_peter(prefix, m - 1, 1);

        else
        {
            future_value_type* m1 = get_freelist().allocate();
            new (m1) ackermann_peter_promised_n_future
                (prefix, prefix, m - 1, m2->get_gid());
            return m1->get();
        }
    } 
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        boost::uint64_t m = vm["m-value"].as<boost::uint64_t>()
                      , n = vm["n-value"].as<boost::uint64_t>();

        const id_type prefix = get_applier().get_runtime_support_gid();

        get_completion_flag().store(false);

        high_resolution_timer t;

        ackermann_peter_future f(prefix, prefix, m, n);

        boost::uint64_t r = f.get();

        double elapsed = t.elapsed();

        get_completion_flag().store(true);

        std::cout
            << ( boost::format("ackermann_peter(%1%, %2%) == %3%\n"
                               "elapsed time == %4%\n")
               % m % n % r % elapsed);

        freelist_type& fl = get_freelist();
        future_value_type* fv = 0;

        while ((fv = fl.get()))
        {
            fv->get();
            fv->~future_value_type();
            fl.deallocate(fv);
        }
    }

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
        ( "m-value"
        , value<boost::uint64_t>()->default_value(2)
        , "m value for Ackermann-Peter function") 

        ( "n-value"
        , value<boost::uint64_t>()->default_value(2)
        , "n value for Ackermann-Peter function") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

