////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436)
#define HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436

#include <list>
#include <cmath>

#include <boost/format.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/iostreams.hpp>

#include <examples/math/distributed/discovery/discovery.hpp>

namespace hpx { namespace balancing { namespace server
{

template <typename T>
inline T absolute_value(T const& t)
{
    if (t < 0)
        return -t;
    else
        return t;
}

template <typename T> 
struct HPX_COMPONENT_EXPORT integrator
    : components::simple_component_base<integrator<T> > 
{
    typedef components::simple_component_base<integrator> base_type; 
   
    struct current_shepherd
    {
        current_shepherd() : prefix(0), shepherd(0) {}

        current_shepherd(boost::uint32_t prefix_, boost::uint32_t shepherd_)
            : prefix(prefix_), shepherd(shepherd_) {}

        boost::uint32_t prefix;
        boost::uint32_t shepherd;
    };
     
  private:
    topology_map* topology_;
    boost::atomic<current_shepherd> current_;
    actions::function<T(T const&)> f_;
    std::vector<naming::id_type> network_;
    T tolerance_;
    boost::uint64_t regrid_segs_;
    T epsilon_;
    const boost::uint32_t here_;

    current_shepherd round_robin()
    {
        current_shepherd expected = current_.load(), desired;

        do {
            // Check if we need to go to the next locality.
            if ((*topology_)[expected.prefix] == (expected.shepherd + 1))
            {
                // Check if we're on the last locality.
                if (topology_->size() == expected.prefix)
                    desired.prefix = 1;
                else
                    desired.prefix = expected.prefix + 1;

                // Reset the shepherd count.
                desired.shepherd = 0;
            }
            
            // Otherwise, just increase the shepherd count.
            else
            {
                desired.prefix = expected.prefix;
                desired.shepherd = expected.shepherd + 1; 
            }
        } while (!current_.compare_exchange_weak(expected, desired));

        return desired;
    }

  public:
    enum actions
    {
        integrator_build_network,
        integrator_deploy,
        integrator_solve_iteration,
        integrator_solve,
    };

    integrator() : here_(applier::get_prefix_id()) {} 

    std::vector<naming::id_type> build_network(
        std::vector<naming::id_type> const& discovery_network
      , actions::function<T(T const&)> const& f 
      , T const& tolerance
      , boost::uint64_t regrid_segs 
      , T const& epsilon 
    ) {
        BOOST_ASSERT(f);

        std::vector<lcos::future_value<naming::id_type, naming::gid_type> >
            results0; 

        const boost::uint32_t root_prefix
            = naming::get_prefix_from_gid(this->get_base_gid());
    
        BOOST_FOREACH(naming::id_type const& node, discovery_network)
        {
            boost::uint32_t current_prefix = naming::get_prefix_from_id(node);
           
            if (root_prefix != current_prefix) 
                results0.push_back
                    (components::stubs::runtime_support::create_component_async
                        ( naming::get_gid_from_prefix(current_prefix)
                        , components::get_component_type<integrator>()));
        }
    
        std::vector<naming::id_type> integrator_network;

        for (std::size_t i = 0; i < discovery_network.size(); ++i)
        {
            if ((i + 1) == root_prefix)
                integrator_network.push_back(this->get_gid());
            else if ((i + 1) > root_prefix)
                integrator_network.push_back(results0[i - 1].get());
            else
                integrator_network.push_back(results0[i].get());
        }

        BOOST_ASSERT(integrator_network.size() == discovery_network.size());
    
        std::list<lcos::future_value<void> > results1;
   
        for (std::size_t i = 0; i < integrator_network.size(); ++i)
        {
            typedef lcos::eager_future<deploy_action> deploy_future; 
            results1.push_back(deploy_future(integrator_network[i]
                                           , discovery_network[i]
                                           , integrator_network
                                           , f
                                           , tolerance
                                           , regrid_segs
                                           , epsilon));
        }
    
        BOOST_FOREACH(lcos::future_value<void> const& r, results1)
        { r.get(); }
    
        return integrator_network;
    }

    void deploy(
        naming::id_type const& discovery_gid
      , std::vector<naming::id_type> const& network
      , actions::function<T(T const&)> const& f 
      , T const& tolerance
      , boost::uint64_t regrid_segs
      , T const& epsilon 
    ) {
        BOOST_ASSERT(f);

        BOOST_ASSERT(here_ == naming::get_prefix_from_id(discovery_gid));

        balancing::discovery disc_client(discovery_gid);

        // DMA shortcut to reduce scheduling overhead.
        topology_ = reinterpret_cast<topology_map*>
            (disc_client.topology_lva_sync());

        network_ = network;

        BOOST_ASSERT(topology_->size() == network_.size());

        f_ = f;
        tolerance_ = tolerance;
        regrid_segs_ = regrid_segs; 
        epsilon_ = epsilon;

        current_.store(current_shepherd(here_, 0)); 
    }

    T solve_iteration(
        T const& i
      , T const& increment
      , boost::uint32_t depth
    ) {
        const T f_i = f_(i);

        // solve() checks the increment size and ensures that the increment we
        // get isn't too small.
        if (absolute_value(f_(i + (increment / 2)) - f_i) < tolerance_)
            // If we're under the tolerance, then we just compute the area.
            return f_i * increment;

        // Regrid.
        else
        {
            lcos::eager_future<solve_action> r
                ( network_[round_robin().prefix - 1], i, i + increment
                , regrid_segs_, 1 + depth);
            return r.get();
        }
    }

    T solve_first(
        T const& f_i 
      , T const& i
      , T const& increment
      , boost::uint32_t depth
    ) {
        // solve() checks the increment size and ensures that the increment we
        // get isn't too small.
        if (absolute_value(f_(i + (increment / 2)) - f_i) < tolerance_)
            // If we're under the tolerance, then we just compute the area.
            return f_i * increment;

        // Regrid.
        else
        {
            lcos::eager_future<solve_action> r
                ( network_[round_robin().prefix - 1], i, i + increment
                , regrid_segs_, 1 + depth);
            return r.get();
        }
    }

    T solve(
        T const& lower_bound
      , T const& upper_bound
      , boost::uint64_t segments
      , boost::uint32_t depth
    ) {
        const T length = (upper_bound - lower_bound);
        const T f_lower = f_(lower_bound);

        // Make sure the range isn't too small. We need 2 * segments values;
        // we have to be able to divide each increment by two to check if we're
        // under the tolerance. If the range is too small, we can't refine any
        // further and our answer is just f_(lower_bound) * length.
        if (length <= (segments * epsilon_ * 2))
            return f_lower * length; 

        std::list<lcos::future_value<T> > results;

        T increment = length / segments;

        BOOST_ASSERT((lower_bound + increment) < upper_bound);

        util::high_resolution_timer t;

        for (boost::uint64_t i = 1; i < segments; ++i)
        {
            if (0 == depth)
                hpx::cout() <<
                    ( boost::format("[%.12f/%.12f] started segment %d at %f")
                    % (lower_bound + (increment * i))
                    % upper_bound
                    % i
                    % t.elapsed()) << hpx::endl;

            const T point = lower_bound + (increment * i);
            results.push_back(lcos::eager_future<solve_iteration_action>
                (this->get_gid(), point, increment, depth)); 
        }

        // Avoid computing f_(lower_bound) another time in the for loop.
        if (0 == depth)
            hpx::cout() <<
                ( boost::format("[%.12f/%.12f] started segment 0 at %f")
                % lower_bound
                % upper_bound
                % t.elapsed()) << hpx::endl;

        T total_area = solve_first(f_lower, lower_bound, increment, depth);  

        if (0 == depth)
            hpx::cout() <<
                ( boost::format("[%.12f/%.12f] completed segment 0 at %f")
                % lower_bound
                % upper_bound
                % t.elapsed()) << hpx::endl;

        typedef typename std::list<lcos::future_value<T> >::iterator iterator;

        iterator it = results.begin(), end = results.end();
 
        // Accumulate final result. TODO: Check for overflow.
        for (boost::uint64_t i = 1; i < segments; ++i, ++it)
        {        
            total_area += it->get();
            if (0 == depth)
                hpx::cout() <<
                    ( boost::format("[%.12f/%.12f] completed segment %d at %f")
                    % (lower_bound + (increment * i))
                    % upper_bound
                    % i
                    % t.elapsed()) << hpx::endl;
        }

        return total_area;
    }

    typedef actions::result_action5<
        // class
        integrator<T>
        // result
      , std::vector<naming::id_type>
        // action value type
      , integrator_build_network
        // arguments 
      , std::vector<naming::id_type> const&
      , actions::function<T(T const&)> const&
      , T const& 
      , boost::uint64_t 
      , T const& 
        // function
      , &integrator<T>::build_network
    > build_network_action;

    typedef actions::action6<
        // class
        integrator<T>
        // action value type
      , integrator_deploy
        // arguments 
      , naming::id_type const&
      , std::vector<naming::id_type> const&
      , actions::function<T(T const&)> const&
      , T const& 
      , boost::uint64_t 
      , T const& 
        // function
      , &integrator<T>::deploy
    > deploy_action;

    typedef actions::result_action3<
        // class
        integrator<T>
        // result
      , T 
        // action value type
      , integrator_solve_iteration
        // arguments 
      , T const& 
      , T const& 
      , boost::uint32_t 
        // function
      , &integrator<T>::solve_iteration
    > solve_iteration_action;
    
    typedef actions::result_action4<
        // class
        integrator<T>
        // result
      , T 
        // action value type
      , integrator_solve
        // arguments 
      , T const& 
      , T const& 
      , boost::uint64_t 
      , boost::uint32_t 
        // function
      , &integrator<T>::solve
    > solve_action;
};

}}}

#endif // HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436

