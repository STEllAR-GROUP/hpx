// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>

#include <unistd.h>

namespace hpx { namespace components { namespace process { namespace posix {

    namespace initializers {

        template <class Range>
        class close_fds_ : public initializer_base
        {
        public:
            explicit close_fds_(const Range& fds)
              : fds_(fds)
            {
            }

            template <class PosixExecutor>
            void on_exec_setup(PosixExecutor&) const
            {
                for (auto& fd : fds_)
                {
                    ::close(fd);
                }
            }

        private:
            Range fds_;
        };

        template <class Range>
        close_fds_<Range> close_fds(const Range& fds)
        {
            return close_fds_<Range>(fds);
        }

}}}}}    // namespace hpx::components::process::posix::initializers

#endif
