//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if _MSC_VER >= 1929
// VS2019 v16.10 and later (_MSC_FULL_VER >= 192829913 for VS 2019 v16.9) supports
// [[msvc::no_unique_address]] with all C++ version starting /std:c++14
// see: https://devblogs.microsoft.com/cppblog/msvc-cpp20-and-the-std-cpp20-switch/
//
// clang-cl does not support neither of the attributes, however
#  if defined(__clang__)
#    error "clang does not support the no_unique_address attribute on Windows"
#  endif
#else
#  if !defined(__has_cpp_attribute)
#    error "__has_cpp_attribute not supported, assume [[no_unique_address]] is not supported"
#  else
#    if !__has_cpp_attribute(no_unique_address)
#      error "__has_cpp_attribute(no_unique_address) not supported"
#    endif
#  endif
#endif

int main()
{
    return 0;
}
