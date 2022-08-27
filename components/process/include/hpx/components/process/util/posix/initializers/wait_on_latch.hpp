// Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/serialization/string.hpp>

#include <string>

namespace hpx { namespace components { namespace process { namespace posix {
    namespace initializers {
        class wait_on_latch : public initializer_base
        {
        public:
            wait_on_latch() = default;

            explicit wait_on_latch(std::string const& connect_to)
              : connect_to_(connect_to)
            {
            }

            template <typename PosixExecutor>
            void on_fork_success(PosixExecutor&) const
            {
                // wait for the newly launched HPX locality to connect back here
                hpx::distributed::latch l(2);
                l.register_as(connect_to_);
                l.arrive_and_wait();
            }

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                // clang-format off
                ar & connect_to_;
                // clang-format on
            }

            std::string connect_to_;
        };
}}}}}    // namespace hpx::components::process::posix::initializers

#endif
