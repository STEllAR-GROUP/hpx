// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_INITIALIZERS_SET_ENV_HPP
#define HPX_PROCESS_POSIX_INITIALIZERS_SET_ENV_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/string.hpp>
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>

#include <string>
#include <vector>

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

template <class Range>
class set_env_ : public initializer_base
{
public:
    set_env_()
    {
        env_.resize(1);
        env_[0] = 0;
    }

    explicit set_env_(const Range &envs)
    {
        string_env_.resize(envs.size());
        env_.resize(envs.size() + 1);
        for (std::size_t i = 0; i != envs.size(); ++i)
        {
            string_env_[i] = envs[i];
            env_[i] = const_cast<char*>(string_env_[i].c_str());
        }
        env_[envs.size()] = 0;
    }

    template <class PosixExecutor>
    void on_fork_setup(PosixExecutor &e) const
    {
        e.env = env_.data();
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void save(Archive& ar, unsigned const) const
    {
        ar & string_env_;
    }

    template <typename Archive>
    void load(Archive& ar, unsigned const)
    {
        ar & string_env_;

        env_.resize(string_env_.size() + 1);
        for (std::size_t i = 0; i != string_env_.size(); ++i)
        {
            env_[i] = const_cast<char*>(string_env_[i].c_str());
        }
        env_[string_env_.size()] = 0;
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()

    std::vector<std::string> string_env_;
    std::vector<char*> env_;
};

template <class Range>
set_env_<Range> set_env(const Range &envs)
{
    return set_env_<Range>(envs);
}

}

}}}}

#endif
