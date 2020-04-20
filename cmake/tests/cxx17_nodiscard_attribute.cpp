////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(__has_cpp_attribute)
#  error "__has_cpp_attribute not supported, assume [[nodiscard]] is not supported"
#else
#  if !__has_cpp_attribute(nodiscard)
#    error "__has_cpp_attribute(nodiscard) not supported"
#  endif
#endif

int main()
{
    return 0;
}
