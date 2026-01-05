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
#include <hpx/iostreams/chain.hpp>
#include <hpx/iostreams/char_traits.hpp>
#include <hpx/iostreams/detail/access_control.hpp>
#include <hpx/iostreams/detail/streambuf/linked_streambuf.hpp>
#include <hpx/iostreams/traits.hpp>

//#include <hpx/iostreams/detail/config/wide_streams.hpp>

#include <streambuf>

namespace hpx::iostreams::detail {

    //--------------Definition of chainbuf----------------------------------------//

    //
    // Template name: chainbuf.
    // Description: Stream buffer which operates by delegating to the first
    //      linked_streambuf in a chain.
    // Template parameters:
    //      Chain - The chain type.
    //
    template <typename Chain, typename Mode, typename Access>
    class chainbuf
      : public std::basic_streambuf<typename Chain::char_type,
            typename Chain::traits_type>
      , public access_control<typename Chain::client_type, Access>
    {
        using client_type = access_control<chain_client<Chain>, Access>;

    public:
        using char_type = Chain::char_type;
        using traits_type = Chain::traits_type;
        using int_type = traits_type::int_type;
        using off_type = traits_type::off_type;
        using pos_type = traits_type::pos_type;

    protected:
        using delegate_type = linked_streambuf<char_type, traits_type>;

        chainbuf()
        {
            client_type::set_chain(&chain_);
        }

        ~chainbuf() override = default;

    public:
        chainbuf(chainbuf const&) = delete;
        chainbuf(chainbuf&&) = default;
        chainbuf& operator=(chainbuf const&) = delete;
        chainbuf& operator=(chainbuf&&) = default;

    protected:
        int_type underflow() override
        {
            sentry t(this);
            return translate(delegate().underflow());
        }

        int_type pbackfail(int_type c) override
        {
            sentry t(this);
            return translate(delegate().pbackfail(c));
        }

        std::streamsize xsgetn(char_type* s, std::streamsize n) override
        {
            sentry t(this);
            return delegate().xsgetn(s, n);
        }

        int_type overflow(int_type c) override
        {
            sentry t(this);
            return translate(delegate().overflow(c));
        }

        std::streamsize xsputn(char_type const* s, std::streamsize n) override
        {
            sentry t(this);
            return delegate().xsputn(s, n);
        }

        int sync() override
        {
            sentry t(this);
            return delegate().sync();
        }

        pos_type seekoff(off_type off, std::ios_base::seekdir way,
            std::ios_base::openmode which = std::ios_base::in |
                std::ios_base::out) override
        {
            sentry t(this);
            return delegate().seekoff(off, way, which);
        }

        pos_type seekpos(pos_type sp,
            std::ios_base::openmode which = std::ios_base::in |
                std::ios_base::out) override
        {
            sentry t(this);
            return delegate().seekpos(sp, which);
        }

        using base_type = std::basic_streambuf<char_type, traits_type>;

    private:
        // Translate from std int_type to chain's int_type.
        using std_traits = std::char_traits<char_type>;

        static traits_type::int_type translate(std_traits::int_type c)
        {
            return translate_int_type<std_traits, traits_type>(c);
        }

        delegate_type& delegate()
        {
            return static_cast<delegate_type&>(chain_.front());
        }

        void get_pointers()
        {
            this->setg(
                delegate().eback(), delegate().gptr(), delegate().egptr());
            this->setp(delegate().pbase(), delegate().epptr());
            this->pbump(
                static_cast<int>(delegate().pptr() - delegate().pbase()));
        }

        void set_pointers()
        {
            delegate().setg(this->eback(), this->gptr(), this->egptr());
            delegate().setp(this->pbase(), this->epptr());
            delegate().pbump(static_cast<int>(this->pptr() - this->pbase()));
        }

        struct sentry
        {
            explicit sentry(chainbuf<Chain, Mode, Access>* buf)
              : buf_(buf)
            {
                buf_->set_pointers();
            }

            ~sentry()
            {
                buf_->get_pointers();
            }

            sentry(sentry const&) = delete;
            sentry(sentry&&) = delete;
            sentry& operator=(sentry const&) = delete;
            sentry& operator=(sentry&&) = delete;

            chainbuf<Chain, Mode, Access>* buf_;
        };

        friend struct sentry;
        Chain chain_;
    };
}    // namespace hpx::iostreams::detail
