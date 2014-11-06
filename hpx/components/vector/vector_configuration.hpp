//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_VECTOR_CONFIGURATION_OCT_20_0948PM)
#define HPX_VECTOR_CONFIGURATION_OCT_20_0948PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <hpx/components/vector/distribution_policy.hpp>
#include <hpx/components/vector/partition_vector_component.hpp>

#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>

namespace hpx { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT vector_configuration
      : public hpx::components::simple_component_base<vector_configuration>
    {
    public:
        // Each partition is described by it's corresponding client object, its
        // size, and locality id.
        struct partition_data
        {
            partition_data()
              : size_(0), locality_id_(naming::invalid_locality_id)
            {}

            partition_data(future<id_type> && part, std::size_t size,
                    boost::uint32_t locality_id)
              : partition_(part.share()),
                size_(size), locality_id_(locality_id)
            {}

            partition_data(id_type const& part, std::size_t size,
                    boost::uint32_t locality_id)
              : partition_(make_ready_future(part).share()),
                size_(size), locality_id_(locality_id)
            {}

            id_type get_id() const
            {
                return partition_.get();
            }

            hpx::shared_future<id_type> partition_;
            std::size_t size_;
            boost::uint32_t locality_id_;
        };

        struct config_data
        {
            config_data()
              : size_(0), block_size_(0), policy_(0)
            {}

            config_data(std::size_t size, std::size_t block_size,
                    std::vector<partition_data> && partitions, int policy)
              : size_(size),
                block_size_(block_size),
                partitions_(std::move(partitions)),
                policy_(policy)
            {}

            std::size_t size_;
            std::size_t block_size_;
            std::vector<partition_data> partitions_;
            int policy_;
        };

        ///////////////////////////////////////////////////////////////////////
        vector_configuration() { HPX_ASSERT(false); }

        vector_configuration(config_data const& data)
          : data_(data)
        {}

        /// Retrieve the configuration data.
        config_data get() const { return data_; }

        ///////////////////////////////////////////////////////////////////////
        HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION(vector_configuration, get);

    private:
        config_data data_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
// Non-intrusive serialization.
namespace boost { namespace serialization
{
    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_iarchive& ar,
        hpx::server::vector_configuration::partition_data& cfg, unsigned int const);
    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_oarchive& ar,
        hpx::server::vector_configuration::partition_data& cfg, unsigned int const);

    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_iarchive& ar,
        hpx::server::vector_configuration::config_data& cfg, unsigned int const);
    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_oarchive& ar,
        hpx::server::vector_configuration::config_data& cfg, unsigned int const);
}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::server::vector_configuration::get_action,
    vector_configuration_get_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<
            hpx::server::vector_configuration::config_data
        >::set_value_action,
    set_value_action_vector_config_data);

#endif

