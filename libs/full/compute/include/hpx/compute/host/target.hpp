///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <hpx/compute/host/get_targets.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/topology/topology.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/runtime/find_here.hpp>
#endif

#include <cstddef>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace compute { namespace host {
    struct HPX_EXPORT target
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
        target()
          : handle_()
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
          , locality_(hpx::find_here())
#endif
        {
        }

        // Constructs target from a given mask of processing units
        explicit target(hpx::threads::mask_type mask)
          : handle_(mask)
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
          , locality_(hpx::find_here())
#endif
        {
        }

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        explicit target(hpx::id_type const& locality)
          : handle_()
          , locality_(locality)
        {
        }

        target(hpx::id_type const& locality, hpx::threads::mask_type mask)
          : handle_(mask)
          , locality_(locality)
        {
        }
#endif

        native_handle_type& native_handle() noexcept
        {
            return handle_;
        }
        native_handle_type const& native_handle() const noexcept
        {
            return handle_;
        }

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        hpx::id_type const& get_locality() const noexcept
        {
            return locality_;
        }
#endif

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

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        static hpx::future<std::vector<target>> get_targets(
            hpx::id_type const& locality)
        {
            return host::get_targets(locality);
        }
#endif

        friend bool operator==(target const& lhs, target const& rhs)
        {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            return lhs.handle_.get_device() == rhs.handle_.get_device() &&
                lhs.locality_ == rhs.locality_;
#else
            return lhs.handle_.get_device() == rhs.handle_.get_device();
#endif
        }

    private:
        friend class hpx::serialization::access;

        void serialize(serialization::input_archive& ar, const unsigned int);
        void serialize(serialization::output_archive& ar, const unsigned int);

        native_handle_type handle_;
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        hpx::id_type locality_;
#endif
    };
}}}    // namespace hpx::compute::host

#include <hpx/config/warnings_suffix.hpp>
