//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostreams/detail/wrap_unwrap.hpp>
#include <hpx/iostreams/stream_buffer.hpp>
#include <hpx/iostreams/traits.hpp>
#include <hpx/modules/type_support.hpp>

#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>

namespace hpx::iostreams {
    namespace detail {

        template <typename Device, typename Tr>
        struct stream_traits
        {
            using char_type = char_type_of_t<Device>;
            using traits_type = Tr;
            using mode = category_of_t<Device>;

            // clang-format off
            using stream_type =
                util::select_t<
                    std::conjunction<std::is_convertible<mode, input>,
                        std::is_convertible<mode, output>>,
                        std::basic_iostream<char_type, traits_type>,
                    std::is_convertible<mode, input>,
                        std::basic_istream<char_type, traits_type>, 
                    util::else_t,
                        std::basic_ostream<char_type, traits_type>>;
            using stream_tag =
                util::select_t<
                    std::conjunction<std::is_convertible<mode, input>,
                        std::is_convertible<mode, output>>,
                        iostream_tag, 
                    std::is_convertible<mode, input>, 
                        istream_tag,
                    util::else_t, 
                        ostream_tag>;
            // clang-format on
        };

        template <typename Device,
            typename Tr = std::char_traits<char_type_of_t<Device>>,
            typename Alloc = std::allocator<char_type_of_t<Device>>,
            typename Base = typename stream_traits<Device, Tr>::stream_type>
        class stream_base
          : protected util::base_from_member<stream_buffer<Device, Tr, Alloc>>
          , public Base
        {
            using pbase_type =
                util::base_from_member<stream_buffer<Device, Tr, Alloc>>;
            using stream_type = stream_traits<Device, Tr>::stream_type;

        protected:
            // Avoid warning about 'this' in initializer list.
            using pbase_type::member;

        public:
            stream_base()
              : pbase_type()
              , stream_type(&member)
            {
            }
        };
    }    // namespace detail

    //
    // Template name: stream.
    //
    // Description: A iostream which reads from and writes to an instance of a
    //      designated device type.
    // Template parameters:
    //      Device - A device type.
    //      Alloc - The allocator type.
    //
    template <typename Device, typename Tr, typename Alloc>
    struct stream : detail::stream_base<Device, Tr, Alloc>
    {
        using char_type = char_type_of_t<Device>;

        struct category
          : mode_of_t<Device>
          , closable_tag
          , detail::stream_traits<Device, Tr>::stream_tag
        {
        };

        using traits_type = Tr;
        using int_type = traits_type::int_type;
        using off_type = traits_type::off_type;
        using pos_type = traits_type::pos_type;

    private:
        using stream_type = detail::stream_traits<Device, Tr>::stream_type;

    public:
        stream() = default;

        explicit stream(Device const& t, std::streamsize const buffer_size = -1,
            std::streamsize const pback_size = -1)
        {
            this->open_impl(detail::wrap(t), buffer_size, pback_size);
        }

        explicit stream(Device& t, std::streamsize const buffer_size = -1,
            std::streamsize const pback_size = -1)
        {
            this->open_impl(detail::wrap(t), buffer_size, pback_size);
        }

        explicit stream(std::reference_wrapper<Device> const& ref,
            std::streamsize const buffer_size = -1,
            std::streamsize const pback_size = -1)
        {
            this->open_impl(ref, buffer_size, pback_size);
        }

        void open(Device const& t, std::streamsize const buffer_size = -1,
            std::streamsize const pback_size = -1)
        {
            this->open_impl(detail::wrap(t), buffer_size, pback_size);
        }

        void open(Device& t, std::streamsize const buffer_size = -1,
            std::streamsize const pback_size = -1)
        {
            this->open_impl(detail::wrap(t), buffer_size, pback_size);
        }

        void open(std::reference_wrapper<Device> const& ref,
            std::streamsize const buffer_size = -1,
            std::streamsize const pback_size = -1)
        {
            this->open_impl(ref, buffer_size, pback_size);
        }

        template <typename U, typename... Us>
            requires(!std::same_as<U, Device>)
        explicit stream(U&& u, Us&&... us)
        {
            this->open_impl(Device(HPX_FORWARD(U, u), HPX_FORWARD(Us, us)...));
        }

        template <typename U, typename... Us>
            requires(!std::same_as<U, Device>)
        void open(U&& u, Us&&... us)
        {
            this->open_impl(Device(HPX_FORWARD(U, u), HPX_FORWARD(Us, us)...));
        }

        [[nodiscard]] bool is_open() const
        {
            return this->member.is_open();
        }
        void close()
        {
            this->member.close();
        }
        [[nodiscard]] bool auto_close() const
        {
            return this->member.auto_close();
        }
        void set_auto_close(bool close)
        {
            this->member.set_auto_close(close);
        }
        bool strict_sync()
        {
            return this->member.strict_sync();
        }

        Device& operator*()
        {
            return *this->member;
        }
        Device* operator->()
        {
            return std::addressof(*this->member);
        }
        Device* component()
        {
            return this->member.component();
        }

    private:
        template <typename Dev>
        void open_impl(Dev&& dev, std::streamsize const buffer_size = -1,
            std::streamsize const pback_size = -1)
        {
            this->clear();
            this->member.open(HPX_FORWARD(Dev, dev), buffer_size, pback_size);
        }
    };
}    // namespace hpx::iostreams
