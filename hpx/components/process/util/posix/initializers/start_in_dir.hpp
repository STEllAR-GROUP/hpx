// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_INITIALIZERS_START_IN_DIR_HPP
#define HPX_PROCESS_POSIX_INITIALIZERS_START_IN_DIR_HPP

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>

#include <string>
#include <unistd.h>

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

class start_in_dir : public initializer_base
{
public:
    start_in_dir() {}

    explicit start_in_dir(const std::string &s) : s_(s) {}

    template <class PosixExecutor>
    void on_exec_setup(PosixExecutor&) const
    {
        ::chdir(s_.c_str());
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned const)
    {
        ar & s_;
    }

    std::string s_;
};

}

}}}}

#endif
#endif
