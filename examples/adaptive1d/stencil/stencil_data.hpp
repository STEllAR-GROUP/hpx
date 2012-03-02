//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2009-2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STENCIL_STENCIL_DATA_AUG_02_2011_0719PM)
#define HPX_COMPONENTS_STENCIL_STENCIL_DATA_AUG_02_2011_0719PM

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <vector>

#include <hpx/lcos/local/mutex.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/valarray.hpp>

#include <examples/adaptive1d/parameter.hpp>
#include <examples/adaptive1d/array1d.hpp>

namespace hpx { namespace components { namespace adaptive1d
{
    namespace server
    {

        ///////////////////////////////////////////////////////////////////////////////
        struct stencil_config_data
        {
            int face_;
            int count_;

        private:
            friend class boost::serialization::access;

            BOOST_SERIALIZATION_SPLIT_MEMBER()

            template<class Archive>
            void save(Archive & ar, const unsigned int version) const
            {
                ar & face_ & count_;
            }

            template<class Archive>
            void load(Archive & ar, const unsigned int version)
            {
                ar & face_ & count_;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // client side representation
    struct stencil_config_data
      : access_memory_block<server::stencil_config_data>
    {
        typedef access_memory_block<server::stencil_config_data> base_type;

        // create a new server::stencil_config_data locally and resolve it
        memory_block_data create_and_resolve_target();

        // create a new server::stencil_config_data locally and initialize it
        stencil_config_data(int face, int size);

        components::memory_block mem_block;
    };

    struct nodedata
    {
      double phi[2][NUM_EQUATIONS];
      double x;
      double error;

      nodedata() {}

      private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & phi & x & error;
        }
    };
    ///////////////////////////////////////////////////////////////////////////
    struct stencil_data
    {
        stencil_data()
          : max_index_(0), index_(0), timestep_(0)
        {}
        ~stencil_data() {}

        stencil_data(stencil_data const& rhs)
          : max_index_(rhs.max_index_), index_(rhs.index_),
            timestep_(rhs.timestep_),
            value_(rhs.value_)
        {
            // intentionally do not copy mutex, new copy will have it's own mutex
        }

        stencil_data& operator=(stencil_data const& rhs)
        {
            if (this != &rhs) {
                max_index_ = rhs.max_index_;
                index_ = rhs.index_;
                timestep_ = rhs.timestep_;
                value_ = rhs.value_;
                // intentionally do not copy mutex, new copy will have it's own mutex
            }
            return *this;
        }

        hpx::lcos::local::mutex mtx_;    // lock for this data block

        std::size_t max_index_;   // overall number of data points
        std::size_t index_;       // sequential number of this data point (0 <= index_ < max_values_)
        double_type timestep_;    // current time step
        array1d<nodedata> value_;    // current value

    private:
        // customized serialization support
        friend struct actions::manage_object_action<
            stencil_data, server::stencil_config_data>;

        template<class Archive>
        void save(Archive & ar, const unsigned int version,
            server::stencil_config_data const* config) const
        {
            ar & max_index_ & index_ & timestep_;
            if (config) {
                if ( config->face_ == 0 ) {
                  // right face -- coming from the left
                  value_.do_save(ar, value_.size() - config->count_, value_.size());
                } else if ( config->face_ == 1 ) {
                  // left face -- coming from the right
                  value_.do_save(ar, 0, config->count_);
                } else {
                  BOOST_ASSERT(false);
                }
                //value_.do_save(ar, config->start_, config->start_+config->count_);
            } else
                value_.do_save(ar, 0, value_.size());
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version,
            server::stencil_config_data const* config)
        {
            ar & max_index_ & index_ & timestep_;
            value_.do_load(ar);
        }

        // 'normal' serialization
        friend class boost::serialization::access;

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            BOOST_ASSERT(false);    // shouldn't ever be called
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            BOOST_ASSERT(false);    // shouldn't ever be called
        }
    };
}}}

#endif

