//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/system/error_code.hpp
// hpxinspect:nodeprecatedname:boost::system::error_code

#pragma once

#include <hpx/local/config.hpp>

#include <boost/system/error_code.hpp>

#include <system_error>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
/// \cond NODETAIL
namespace hpx {

    class compat_error_code
    {
    public:
        explicit compat_error_code(std::error_code& ec)
          : _ec(ec)
          , _compat()
        {
        }

        ~compat_error_code()
        {
            _ec = _compat;
        }

        operator boost::system::error_code&() noexcept
        {
            return _compat;
        }

        template <typename E,
            typename Enable = typename std::enable_if<
                boost::system::is_error_code_enum<E>::value>::type>
        static bool equal(std::error_code const& lhs, E rhs) noexcept
        {
            return lhs == std::error_code(make_error_code(rhs));
        }

    private:
        std::error_code& _ec;
        boost::system::error_code _compat;
    };
}    // namespace hpx
