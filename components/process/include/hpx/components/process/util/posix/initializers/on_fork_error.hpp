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
#include <hpx/serialization/serialization_fwd.hpp>

#include <utility>

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

template <class Handler>
class on_fork_error_ : public initializer_base
{
public:
    on_fork_error_() {}

    explicit on_fork_error_(Handler handler)
      : handler_(std::move(handler))
    {}

    template <class PosixExecutor>
    void on_fork_error(PosixExecutor &e) const
    {
        handler_(e);
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned const)
    {
        ar & handler_;
    }

    Handler handler_;
};

template <class Handler>
on_fork_error_<Handler> on_fork_error(Handler && handler)
{
    return on_fork_error_<Handler>(std::forward<Handler>(handler));
}

}

}}}}

#endif
