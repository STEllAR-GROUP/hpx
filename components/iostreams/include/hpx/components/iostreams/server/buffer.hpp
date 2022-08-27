//  Copyright (c) 2011-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/synchronization/recursive_mutex.hpp>

#include <hpx/components/iostreams/export_definitions.hpp>
#include <hpx/components/iostreams/write_functions.hpp>

#include <iosfwd>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace hpx { namespace iostreams { namespace detail {
    struct buffer
    {
    protected:
        typedef lcos::local::recursive_mutex mutex_type;

    public:
        buffer()
          : data_(new std::vector<char>)
          , mtx_(new mutex_type)
        {
        }

        buffer(buffer const& rhs)
          : data_(rhs.data_)
          , mtx_(rhs.mtx_)
        {
        }

        buffer(buffer&& rhs)
          : data_(HPX_MOVE(rhs.data_))
          , mtx_(HPX_MOVE(rhs.mtx_))
        {
        }

        buffer& operator=(buffer const& rhs)
        {
            if (this != &rhs)
            {
                data_ = rhs.data_;
                mtx_ = rhs.mtx_;
            }
            return *this;
        }

        buffer& operator=(buffer&& rhs)
        {
            if (this != &rhs)
            {
                data_ = HPX_MOVE(rhs.data_);
                mtx_ = HPX_MOVE(rhs.mtx_);
            }
            return *this;
        }

        bool empty() const
        {
            std::lock_guard<mutex_type> l(*mtx_);
            return empty_locked();
        }

        bool empty_locked() const
        {
            return !data_.get() || data_->empty();
        }

        buffer init()
        {
            std::lock_guard<mutex_type> l(*mtx_);
            return init_locked();
        }

        buffer init_locked()
        {
            buffer b;
            std::swap(b.data_, data_);
            return b;
        }

        template <typename Char>
        std::streamsize write(Char const* s, std::streamsize n)
        {
            std::lock_guard<mutex_type> l(*mtx_);
            std::copy(s, s + n, std::back_inserter(*data_));
            return n;
        }

        template <typename Mutex>
        void write(write_function_type const& f, Mutex& mtx)
        {
            std::unique_lock<mutex_type> l(*mtx_);
            if (data_.get())
            {
                // execute even for empty buffers as this will flush the output
                std::shared_ptr<std::vector<char>> data(data_);
                data_.reset();
                l.unlock();

                std::lock_guard<Mutex> ll(mtx);
                f(*data);
            }
        }

    private:
        std::shared_ptr<std::vector<char>> data_;

    protected:
        std::shared_ptr<mutex_type> mtx_;

    private:
        friend class hpx::serialization::access;

        HPX_IOSTREAMS_EXPORT void save(
            serialization::output_archive& ar, unsigned) const;
        HPX_IOSTREAMS_EXPORT void load(
            serialization::input_archive& ar, unsigned);

        HPX_SERIALIZATION_SPLIT_MEMBER();
    };
}}}    // namespace hpx::iostreams::detail
