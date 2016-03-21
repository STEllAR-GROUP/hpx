// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_WINDOWS_INITIALIZERS_WAIT_ON_LATCH_HPP
#define HPX_PROCESS_WINDOWS_INITIALIZERS_WAIT_ON_LATCH_HPP

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/lcos/latch.hpp>
#include <hpx/runtime/serialization/string.hpp>
#include <hpx/components/process/util/windows/initializers/initializer_base.hpp>

#include <string>

namespace hpx { namespace components { namespace process { namespace windows
{
    namespace initializers
    {
        class wait_on_latch : public initializer_base
        {
        public:
            wait_on_latch() {}

            explicit wait_on_latch(std::string const& connect_to)
              : connect_to_(connect_to)
            {}

            template <typename WindowsExecutor>
            void on_CreateProcess_success(WindowsExecutor &) const
            {
                // wait for the newly launched HPX locality to connect back here
                hpx::lcos::latch l(2);
                l.register_as(connect_to_);
                l.count_down_and_wait();
            }

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & connect_to_;
            }

            std::string connect_to_;
        };
    }
}}}}

#endif
#endif
