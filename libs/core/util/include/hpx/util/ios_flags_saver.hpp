//  Copyright (c) 2020-2022 Hartmut Kaiser
//  Copyright 2002, 2005 Daryle Walker
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <ios>

namespace hpx::util {

    // this is taken from the Boost.Io library
    class ios_flags_saver
    {
    public:
        using state_type = ::std::ios_base;
        using aspect_type = ::std::ios_base::fmtflags;

        explicit ios_flags_saver(state_type& s)
          : s_save_(s)
          , a_save_(s.flags())
        {
        }
        ios_flags_saver(state_type& s, aspect_type a)
          : s_save_(s)
          , a_save_(s.flags(a))
        {
        }

        ~ios_flags_saver()
        {
            restore();
        }

        ios_flags_saver(ios_flags_saver const&) = delete;
        ios_flags_saver& operator=(ios_flags_saver const&) = delete;

        void restore() const
        {
            s_save_.flags(a_save_);
        }

    private:
        state_type& s_save_;
        aspect_type const a_save_;
    };
}    // namespace hpx::util
