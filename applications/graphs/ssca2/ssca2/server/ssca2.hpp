//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_SSCA2_AUG_14_2009_1030AM)
#define HPX_COMPONENTS_SERVER_SSCA2_AUG_14_2009_1030AM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <hpx/components/graph/graph.hpp>
#include <hpx/components/distributed_set/distributed_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>
#include <hpx/components/distributed_map/local_map.hpp>

#include "boost/serialization/map.hpp"

#include "../../pbreak/pbreak.hpp"

#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// The ssca2 is an HPX component.
    ///
    class HPX_COMPONENT_EXPORT ssca2
      : public simple_component_base<ssca2>
    {
    private:
        typedef simple_component_base<ssca2> base_type;
        
    public:
        ssca2();
        
        typedef hpx::components::server::ssca2 wrapping_type;
        
        enum actions
        {
            ssca2_large_set = 0,
            ssca2_large_set_local = 1,
            ssca2_extract = 2,
            ssca2_extract_local = 3,
            ssca2_extract_subgraph = 4,
            ssca2_init_props_map = 5,
            ssca2_init_props_map_local = 6
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // This should go somewhere else ... but where?
        struct graph_foo
        {
            graph_foo()
            {}

            graph_foo(naming::id_type const& G)
              : G_(G)
              {}

            bool operator<(const graph_foo& that)
            {
                return this->G_ < that.G_;
            }

            naming::id_type G_;

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & G_;
            }
        };

        struct edge
        {
            edge()
            {}

            edge(naming::id_type const& source,
                 naming::id_type const& target,
                 int label)
              : source_(source), target_(target), label_(label)
            {}

            bool operator<(const edge& that)
            {
                return (this->source_ < that.source_)
                       && (this->target_ < that.target_)
                       && (this->label_ < that.label_);
            }

            naming::id_type source_;
            naming::id_type target_;
            int label_;

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & source_ & target_ & label_;
            }
        };

        struct props
        {
        private:
            struct tag {};
            typedef hpx::util::spinlock_pool<tag> mutex_type;

        public:
            props()
              : color_(0)
            {}

            int color_black()
            {
                mutex_type::scoped_lock l(this);

                if (color_ == 0)
                {
                    color_ = 1;
                    return 1;
                }

                return 0;
            }

        private:
            int color_;

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & color_;
            }

        };

        struct pbreak_closure
        {
        public:
            pbreak_closure()
              : gid_(naming::invalid_id), partial_(0)
            {}

            pbreak_closure(naming::id_type gid, int partial)
              : gid_(gid), partial_(partial)
            {}

            void update(int partial)
            {
                partial_ += partial;
            }

            void signal(void)
            {
                // Asynchronous on signal
                applier::apply<lcos::detail::pbreak::signal_action>(gid_,partial_);
            }

            int wait(void)
            {
                // Synchronous on the wait
                //applier::apply<lcos::detail::pbreak::wait_action>(gid_);
                return lcos::eager_future<lcos::detail::pbreak::wait_action>(gid_).get();
            }

        private:
            naming::id_type gid_;
            int partial_;

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & gid_ & partial_;
            }
        };

        typedef std::vector<edge> edge_set_type;
        typedef distributing_factory::locality_result locality_result;

        typedef distributed_set<edge_set_type> dist_edge_set_type;

        typedef std::vector<graph_foo> graph_set_type;
        typedef distributed_set<graph_set_type> dist_graph_set_type;

        typedef std::map<naming::id_type,naming::id_type> gids_map_type;
        typedef distributed_map<gids_map_type> dist_gids_map_type;
        typedef local_map<gids_map_type> local_gids_map_type;

        int
        large_set(naming::id_type G,
                  naming::id_type dist_edge_set);

        int
        large_set_local(locality_result local_set,
                        naming::id_type edge_set,
                        naming::id_type local_max_lco,
                        naming::id_type global_max_lco);

        int
        extract(naming::id_type edge_set,
                naming::id_type subgraphs);

        int
        extract_local(naming::id_type local_edge_set,
                      naming::id_type subgraphs);

        int
        extract_subgraph(naming::id_type H,
                         naming::id_type pmap,
                         naming::id_type source,
                         naming::id_type vertex,
                         int d);

        int
        init_props_map(naming::id_type P,
                       naming::id_type G);

        int
        init_props_map_local(naming::id_type local_props,
                             locality_result local_vertices);

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_large_set,
            naming::id_type, naming::id_type,
            &ssca2::large_set
        > large_set_action;

        typedef hpx::actions::result_action4<
            ssca2, int, ssca2_large_set_local,
            locality_result, naming::id_type, naming::id_type, naming::id_type,
            &ssca2::large_set_local
        > large_set_local_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_extract,
            naming::id_type, naming::id_type,
            &ssca2::extract
        > extract_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_extract_local,
            naming::id_type, naming::id_type,
            &ssca2::extract_local
        > extract_local_action;

        typedef hpx::actions::result_action5<
            ssca2, int, ssca2_extract_subgraph,
            naming::id_type, naming::id_type, naming::id_type, naming::id_type, int,
            &ssca2::extract_subgraph
        > extract_subgraph_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_init_props_map,
            naming::id_type, naming::id_type,
            &ssca2::init_props_map
        > init_props_map_action;

        typedef hpx::actions::result_action2<
            ssca2, int, ssca2_init_props_map_local,
            naming::id_type, locality_result,
            &ssca2::init_props_map_local
        > init_props_map_local_action;

    };

}}}

#endif
