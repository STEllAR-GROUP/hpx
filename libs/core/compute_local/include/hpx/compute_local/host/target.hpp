///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/compute_local/host/get_targets.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::compute::host {

    struct HPX_CORE_EXPORT target
    {
    public:
        struct native_handle_type
        {
            native_handle_type()
              : mask_(
                    hpx::threads::create_topology().get_machine_affinity_mask())
            {
            }

            explicit native_handle_type(hpx::threads::mask_type mask)
              : mask_(mask)
            {
            }

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
        target() = default;

        // Constructs target from a given mask of processing units
        explicit target(hpx::threads::mask_type mask)
          : handle_(mask)
        {
        }

        native_handle_type& native_handle() noexcept
        {
            return handle_;
        }
        native_handle_type const& native_handle() const noexcept
        {
            return handle_;
        }

        std::pair<std::size_t, std::size_t> num_pus() const;

        constexpr void synchronize() const noexcept
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

        friend bool operator==(target const& lhs, target const& rhs)
        {
            return lhs.handle_.get_device() == rhs.handle_.get_device();
        }

    private:
        friend class hpx::serialization::access;

        void serialize(serialization::input_archive& ar, unsigned int);
        void serialize(serialization::output_archive& ar, unsigned int);

        native_handle_type handle_;
    };
}    // namespace hpx::compute::host

#include <hpx/config/warnings_suffix.hpp>
