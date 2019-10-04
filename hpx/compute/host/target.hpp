///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HOST_TARGET_HPP
#define HPX_COMPUTE_HOST_TARGET_HPP

#include <hpx/config.hpp>

#include <hpx/compute/host/get_targets.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace compute { namespace host
{
    struct target
    {
    public:
        struct HPX_EXPORT native_handle_type
        {
            native_handle_type()
              : mask_(hpx::threads::get_topology().get_machine_affinity_mask())
            {}

            explicit native_handle_type(hpx::threads::mask_type mask)
              : mask_(mask)
            {}

            hpx::threads::mask_type& get_device() noexcept
            {
                return mask_;
            }
            hpx::threads::mask_type const& get_device() const noexcept
            {
                return mask_;
            }

        private:
            friend struct target;

            hpx::threads::mask_type mask_;
        };

    public:
        // Constructs default target
        target()
          : handle_(), locality_(hpx::find_here())
        {
        }

        // Constructs target from a given mask of processing units
        explicit target(hpx::threads::mask_type mask)
          : handle_(mask), locality_(hpx::find_here())
        {
        }

        explicit target(hpx::id_type const& locality)
          : handle_(), locality_(locality)
        {
        }

        target(hpx::id_type const& locality, hpx::threads::mask_type mask)
          : handle_(mask), locality_(locality)
        {
        }

        native_handle_type & native_handle() noexcept
        {
            return handle_;
        }
        native_handle_type const& native_handle() const noexcept
        {
            return handle_;
        }

        hpx::id_type const& get_locality() const noexcept
        {
            return locality_;
        }

        std::pair<std::size_t, std::size_t> num_pus() const;

        void synchronize() const
        {
            // nothing to do here...
        }

        hpx::future<void> get_future() const
        {
            return hpx::make_ready_future();
        }

        static std::vector<target> get_local_targets()
        {
            return host::get_local_targets();
        }
        static hpx::future<std::vector<target> >
            get_targets(hpx::id_type const& locality)
        {
            return host::get_targets(locality);
        }

        friend bool operator==(target const& lhs, target const& rhs)
        {
            return lhs.handle_.get_device() == rhs.handle_.get_device() &&
                lhs.locality_ == rhs.locality_;
        }

    private:
        friend class hpx::serialization::access;

        void serialize(serialization::input_archive& ar, const unsigned int);
        void serialize(serialization::output_archive& ar, const unsigned int);

        native_handle_type handle_;
        hpx::id_type locality_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
