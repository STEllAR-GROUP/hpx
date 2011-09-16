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
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

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
    boost::uint32_t total_shepherds_;
    boost::atomic<current_shepherd> current_;
    actions::function<T(T const&)> f_;
    std::vector<naming::id_type> network_;
    T tolerance_;
    boost::uint32_t regrid_segs_;
    T epsilon_;
    const boost::uint32_t here_;

/*
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
*/

    boost::uint32_t round_robin(
        current_shepherd& cs
      , boost::uint32_t max_shepherds
    ) {
        boost::uint32_t shepherds = 0;
        current_shepherd expected = current_.load(), desired;

        do {
            // Check if we need to go to the next locality.
            if ((*topology_)[expected.prefix] == expected.shepherd)
            {
                const boost::uint32_t first_ = topology_->begin()->first;

                // Check if we're on the last locality.
                if (topology_->size() == (expected.prefix - first_ + 1))
                    desired.prefix = first_;
                else
                    desired.prefix = expected.prefix + 1;

                const boost::uint32_t remaining = (*topology_)[desired.prefix];

                if (remaining >= max_shepherds)
                {
                    shepherds = max_shepherds;
                    desired.shepherd = max_shepherds;
                }
 
                else
                {
                    shepherds = remaining;
                    desired.shepherd = remaining; 
                }
            }
            
            else
            {
                desired.prefix = expected.prefix;

                const boost::uint32_t remaining =
                    (*topology_)[expected.prefix] - expected.shepherd;

                if (remaining >= max_shepherds)
                {
                    shepherds = max_shepherds;
                    desired.shepherd = expected.shepherd + max_shepherds;
                }
 
                else
                {
                    shepherds = remaining;
                    desired.shepherd = expected.shepherd + remaining; 
                }
            }
        } while (!current_.compare_exchange_weak(expected, desired));

        cs = desired;
        return shepherds;
    }

  public:
    enum actions
    {
        integrator_build_network,
        integrator_deploy,
        integrator_solve_iterations,
        integrator_solve,
    };

    integrator() : here_(applier::get_prefix_id()) {} 

    std::vector<naming::id_type> build_network(
        std::vector<naming::id_type> const& discovery_network
      , hpx::actions::function<T(T const&)> const& f 
      , T const& tolerance
      , boost::uint32_t regrid_segs 
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
            const boost::uint32_t current_prefix =
                naming::get_prefix_from_id(discovery_network[i]);

            if (current_prefix == root_prefix)
                integrator_network.push_back(this->get_gid());
            else if (current_prefix > root_prefix)
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
      , hpx::actions::function<T(T const&)> const& f 
      , T const& tolerance
      , boost::uint32_t regrid_segs
      , T const& epsilon 
    ) {
        BOOST_ASSERT(f);

        BOOST_ASSERT(here_ == naming::get_prefix_from_id(discovery_gid));

        balancing::discovery disc_client(discovery_gid);

        // DMA shortcut to reduce scheduling overhead.
        topology_ = reinterpret_cast<topology_map*>
            (disc_client.topology_lva_sync());

        total_shepherds_ = disc_client.total_shepherds_sync();

        network_ = network;

        BOOST_ASSERT(topology_->size() == network_.size());

        f_ = f;
        tolerance_ = tolerance;
        regrid_segs_ = regrid_segs; 
        epsilon_ = epsilon;

        current_.store(current_shepherd(here_, 0)); 
    }

    T solve_iterations(
        T const& lower_bound
      , T const& increment
      , boost::uint32_t iterations
      , boost::uint32_t depth
    ) {
        T area(0);

        std::list<lcos::future_value<T> > results;

        const boost::uint32_t here_index
            = here_ - naming::get_prefix_from_id(network_[0]);

        for (boost::uint32_t iteration = 0; iteration < iterations; ++iteration)
        { 
            const T i = lower_bound + (increment * iteration);
            const T f_i = f_(i);

            // solve() checks the increment size and ensures that the increment
            // we get isn't too small.
            if (absolute_value(f_(i + (increment / 2)) - f_i) < tolerance_)
                // If we're under the tolerance, then we just compute the area.
                area += f_i * increment;

            // Regrid.
            else
            {
//                area += solve(i, i + increment, regrid_segs_, 1 + depth);
                results.push_back(
                    lcos::eager_future<solve_action>
                        ( network_[here_index], i, i + increment
                        , regrid_segs_, 1 + depth));
            }
        }

        BOOST_FOREACH(lcos::future_value<T> const& r, results)
        { area += r.get(); }

        return area;
    }

    T solve(
        T const& lower_bound
      , T const& upper_bound
      , boost::uint32_t segments
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

        typedef boost::fusion::vector2<
            lcos::future_value<T>
          , current_shepherd
        > result_type; 

        std::list<result_type> results;

        const T increment = length / segments;

        BOOST_ASSERT((lower_bound + increment) < upper_bound);

        util::high_resolution_timer* t = 0;

        if (0 >= depth)
            t = new util::high_resolution_timer;

        boost::uint32_t first_round = 0;

        topology_map::iterator top_it = topology_->begin()
                             , top_end = topology_->end();

        for (std::size_t i = 0; top_it != top_end; ++top_it, ++i)
        {
            const double node_ratio
                = double(top_it->second) / double(total_shepherds_);

            const boost::uint32_t points
                = boost::uint32_t(std::floor(node_ratio * segments));

            if (0 >= depth)
            {
                if (1 == points)
                    hpx::cout << (boost::format(
                        "[%.12f/%.12f:%d] started segment %d at %f on L%d")
                        % (lower_bound + (increment * first_round))
                        % upper_bound
                        % depth
                        % first_round
                        % t->elapsed()
                        % top_it->first) << hpx::endl;
                else 
                    hpx::cout << (boost::format(
                        "[%.12f/%.12f:%d] started segments %d-%d at %f on L%d")
                        % (lower_bound + (increment * first_round))
                        % upper_bound
                        % depth
                        % first_round
                        % (first_round + points)
                        % t->elapsed()
                        % top_it->first) << hpx::endl;
            }

            const T point = lower_bound + (increment * first_round);
            results.push_back(result_type
                (lcos::eager_future<solve_iterations_action>
                    (network_[i]
                    , point, increment, points, depth)
                , current_shepherd(top_it->first, points))); 

            first_round += points;
        }

        for (boost::uint32_t i = first_round; i < segments;)
        {
            const boost::uint32_t max_shepherds = segments - i;
 
            current_shepherd cs; 
            const boost::uint32_t shepherds = round_robin(cs, max_shepherds);

            BOOST_ASSERT(shepherds);

            const boost::uint32_t cs_index
                = cs.prefix - naming::get_prefix_from_id(network_[0]);

            if (0 >= depth)
            {
                if (1 == shepherds)
                    hpx::cout << (boost::format(
                        "[%.12f/%.12f:%d] started segment %d at %f on L%d")
                        % (lower_bound + (increment * i))
                        % upper_bound
                        % depth
                        % i
                        % t->elapsed()
                        % cs.prefix) << hpx::endl;
                else 
                    hpx::cout << (boost::format(
                        "[%.12f/%.12f:%d] started segments %d-%d at %f on L%d")
                        % (lower_bound + (increment * i))
                        % upper_bound
                        % depth
                        % i
                        % (i + shepherds)
                        % t->elapsed()
                        % cs.prefix) << hpx::endl;
            }

            const T point = lower_bound + (increment * i);
            results.push_back(result_type
                (lcos::eager_future<solve_iterations_action>
                    (network_[cs_index]
                    , point, increment, shepherds, depth)
                , current_shepherd(cs.prefix, shepherds))); 

            i += shepherds;
        }

        typename std::list<result_type>::iterator it = results.begin()
                                                , end = results.end();
 
        // Accumulate final result. TODO: Check for overflow.
        T total_area(0);
        for (boost::uint32_t i = 0; it != end; ++it)
        {        
            using boost::fusion::at_c;

            total_area += at_c<0>(*it).get();

            if (0 >= depth)
            {
                if (1 == at_c<1>(*it).shepherd)
                    hpx::cout << (boost::format(
                        "[%.12f/%.12f:%d] segment %d on L%d completed at %f")
                        % (lower_bound + (increment * i))
                        % upper_bound
                        % depth
                        % i
                        % at_c<1>(*it).prefix  
                        % t->elapsed()) << hpx::endl;
                else
                    hpx::cout << (boost::format(
                        "[%.12f/%.12f:%d] segments %d-%d on L%d completed at %f")
                        % (lower_bound + (increment * i))
                        % upper_bound
                        % depth
                        % i
                        % (i + at_c<1>(*it).shepherd)
                        % at_c<1>(*it).prefix 
                        % t->elapsed()) << hpx::endl;
            }

            i += at_c<1>(*it).shepherd;
        }

        if (0 >= depth)
            delete t;

        return total_area;
    }

    typedef hpx::actions::result_action5<
        // class
        integrator<T>
        // result
      , std::vector<naming::id_type>
        // action value type
      , integrator_build_network
        // arguments 
      , std::vector<naming::id_type> const&
      , hpx::actions::function<T(T const&)> const&
      , T const& 
      , boost::uint32_t 
      , T const& 
        // function
      , &integrator<T>::build_network
    > build_network_action;

    typedef hpx::actions::action6<
        // class
        integrator<T>
        // action value type
      , integrator_deploy
        // arguments 
      , naming::id_type const&
      , std::vector<naming::id_type> const&
      , hpx::actions::function<T(T const&)> const&
      , T const& 
      , boost::uint32_t 
      , T const& 
        // function
      , &integrator<T>::deploy
    > deploy_action;

    typedef hpx::actions::result_action4<
        // class
        integrator<T>
        // result
      , T 
        // action value type
      , integrator_solve_iterations
        // arguments 
      , T const& 
      , T const& 
      , boost::uint32_t 
      , boost::uint32_t 
        // function
      , &integrator<T>::solve_iterations
    > solve_iterations_action;
    
    typedef hpx::actions::result_action4<
        // class
        integrator<T>
        // result
      , T 
        // action value type
      , integrator_solve
        // arguments 
      , T const& 
      , T const& 
      , boost::uint32_t 
      , boost::uint32_t 
        // function
      , &integrator<T>::solve
    > solve_action;
};

}}}

#endif // HPX_1438B63A_BA4C_4DB3_B835_C8CDBD79B436

