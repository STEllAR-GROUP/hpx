//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// See http://www.boost.org/libs/iostreams for documentation.

// File:        hpx/iostream/detail/functional.hpp
// Date:        Sun Dec 09 05:38:03 MST 2007
// Copyright:   2007-2008 CodeRage, LLC
// Author:      Jonathan Turkanis
// Contact:     turkanis at coderage dot com
//
// Defines several function objects and object generators for use with
// execute_all()

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostream/close.hpp>

#include <iosfwd>

namespace hpx::iostream::detail {

    // Function objects and object generators for invoking hpx::iostream::close
    HPX_CXX_CORE_EXPORT template <typename T>
    class device_close_operation
    {
    public:
        constexpr device_close_operation(
            T& t, std::ios_base::openmode const which) noexcept
          : t_(t)
          , which_(which)
        {
        }

        ~device_close_operation() = default;

        void operator()() const
        {
            iostream::close(t_, which_);
        }

        device_close_operation(device_close_operation const&) = delete;
        device_close_operation(device_close_operation&&) = delete;
        device_close_operation& operator=(
            device_close_operation const&) = delete;
        device_close_operation& operator=(device_close_operation&&) = delete;

    private:
        T& t_;
        std::ios_base::openmode which_;
    };

    HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
    class filter_close_operation
    {
    public:
        constexpr filter_close_operation(
            T& t, Sink& snk, std::ios_base::openmode const which) noexcept
          : t_(t)
          , snk_(snk)
          , which_(which)
        {
        }

        ~filter_close_operation() = default;

        filter_close_operation(filter_close_operation const&) = delete;
        filter_close_operation(filter_close_operation&&) = delete;
        filter_close_operation& operator=(
            filter_close_operation const&) = delete;
        filter_close_operation& operator=(filter_close_operation&&) = delete;

        void operator()() const
        {
            iostream::close(t_, snk_, which_);
        }

    private:
        T& t_;
        Sink& snk_;
        std::ios_base::openmode which_;
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    constexpr device_close_operation<T> call_close(
        T& t, std::ios_base::openmode const which) noexcept
    {
        return device_close_operation<T>(t, which);
    }

    HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
    constexpr filter_close_operation<T, Sink> call_close(
        T& t, Sink& snk, std::ios_base::openmode const which) noexcept
    {
        return filter_close_operation<T, Sink>(t, snk, which);
    }

    // Function objects and object generators for invoking
    // hpx::iostream::detail::close_all
    HPX_CXX_CORE_EXPORT template <typename T>
    class device_close_all_operation
    {
    public:
        explicit constexpr device_close_all_operation(T& t) noexcept
          : t_(t)
        {
        }

        void operator()() const
        {
            detail::close_all(t_);
        }

        ~device_close_all_operation() = default;

        device_close_all_operation(device_close_all_operation const&) = delete;
        device_close_all_operation(device_close_all_operation&&) = delete;
        device_close_all_operation& operator=(
            device_close_all_operation const&) = delete;
        device_close_all_operation& operator=(
            device_close_all_operation&&) = delete;

    private:
        T& t_;
    };

    HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
    class filter_close_all_operation
    {
    public:
        constexpr filter_close_all_operation(T& t, Sink& snk) noexcept
          : t_(t)
          , snk_(snk)
        {
        }

        ~filter_close_all_operation() = default;

        filter_close_all_operation(filter_close_all_operation const&) = delete;
        filter_close_all_operation(filter_close_all_operation&&) = delete;
        filter_close_all_operation& operator=(
            filter_close_all_operation&&) = delete;
        filter_close_all_operation& operator=(
            filter_close_all_operation const&) = delete;

        void operator()() const
        {
            detail::close_all(t_, snk_);
        }

    private:
        T& t_;
        Sink& snk_;
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    constexpr device_close_all_operation<T> call_close_all(T& t) noexcept
    {
        return device_close_all_operation<T>(t);
    }

    HPX_CXX_CORE_EXPORT template <typename T, typename Sink>
    constexpr filter_close_all_operation<T, Sink> call_close_all(
        T& t, Sink& snk) noexcept
    {
        return filter_close_all_operation<T, Sink>(t, snk);
    }

    // Function object and object generator for invoking a member function void
    // close(std::ios_base::openmode)
    HPX_CXX_CORE_EXPORT template <typename T>
    class member_close_operation
    {
    public:
        constexpr member_close_operation(
            T& t, std::ios_base::openmode const which) noexcept
          : t_(t)
          , which_(which)
        {
        }

        ~member_close_operation() = default;

        member_close_operation(member_close_operation const&) = delete;
        member_close_operation(member_close_operation&&) = delete;
        member_close_operation& operator=(
            member_close_operation const&) = delete;
        member_close_operation& operator=(member_close_operation&&) = delete;

        void operator()() const
        {
            t_.close(which_);
        }

    private:
        T& t_;
        std::ios_base::openmode which_;
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    constexpr member_close_operation<T> call_member_close(
        T& t, std::ios_base::openmode which) noexcept
    {
        return member_close_operation<T>(t, which);
    }

    // Function object and object generator for invoking a member function void
    // reset()
    HPX_CXX_CORE_EXPORT template <typename T>
    class reset_operation
    {
    public:
        explicit constexpr reset_operation(T& t) noexcept
          : t_(t)
        {
        }

        ~reset_operation() = default;

        reset_operation(reset_operation const&) = delete;
        reset_operation(reset_operation&&) = delete;
        reset_operation& operator=(reset_operation const&) = delete;
        reset_operation& operator=(reset_operation&&) = delete;

        void operator()() const
        {
            t_.reset();
        }

    private:
        T& t_;
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    constexpr reset_operation<T> call_reset(T& t) noexcept
    {
        return reset_operation<T>(t);
    }

    // Function object and object generator for clearing a flag
    HPX_CXX_CORE_EXPORT template <typename T>
    class clear_flags_operation
    {
    public:
        explicit constexpr clear_flags_operation(T& t) noexcept
          : t_(t)
        {
        }

        ~clear_flags_operation() = default;

        clear_flags_operation(clear_flags_operation const&) = delete;
        clear_flags_operation(clear_flags_operation&&) = delete;
        clear_flags_operation& operator=(clear_flags_operation const&) = delete;
        clear_flags_operation& operator=(clear_flags_operation&&) = delete;

        void operator()() const
        {
            t_ = 0;
        }

    private:
        T& t_;
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    constexpr clear_flags_operation<T> clear_flags(T& t) noexcept
    {
        return clear_flags_operation<T>(t);
    }

    // Function object and generator for flushing a buffer Function object for
    // use with execute_all()
    HPX_CXX_CORE_EXPORT template <typename Buffer, typename Device>
    class flush_buffer_operation
    {
    public:
        constexpr flush_buffer_operation(
            Buffer& buf, Device& dev, bool const flush) noexcept
          : buf_(buf)
          , dev_(dev)
          , flush_(flush)
        {
        }

        ~flush_buffer_operation() = default;

        flush_buffer_operation(flush_buffer_operation const&) = delete;
        flush_buffer_operation(flush_buffer_operation&&) = delete;
        flush_buffer_operation& operator=(
            flush_buffer_operation const&) = delete;
        flush_buffer_operation& operator=(flush_buffer_operation&&) = delete;

        void operator()() const
        {
            if (flush_)
                buf_.flush(dev_);
        }

    private:
        Buffer& buf_;
        Device& dev_;
        bool flush_;
    };

    HPX_CXX_CORE_EXPORT template <typename Buffer, typename Device>
    constexpr flush_buffer_operation<Buffer, Device> flush_buffer(
        Buffer& buf, Device& dev, bool flush) noexcept
    {
        return flush_buffer_operation<Buffer, Device>(buf, dev, flush);
    }
}    // namespace hpx::iostream::detail
