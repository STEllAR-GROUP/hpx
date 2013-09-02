//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifndef HPX_FUNCTION_DETAIL_VTABLE_PTR_BASE_FWD_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_PTR_BASE_FWD_HPP

namespace hpx { namespace util { namespace detail {

    template <
        typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_virtbase;

    template <
        typename Sig
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base;

}}}

#endif
