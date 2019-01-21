//  Copyright (c) 2018 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// sphinx:undocumented

#if !defined(HPX_CONFIG_WEAK_SYMBOL_HPP)
#define HPX_CONFIG_WEAK_SYMBOL_HPP

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define HPX_WEAK_SYMBOL __attribute__((weak))
#else
#define HPX_WEAK_SYMBOL
#endif

#endif
