//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_VTABLE_HPP

#include <hpx/config/forceinline.hpp>
#include <hpx/util/move.hpp>

#include <memory>
#include <typeinfo>

namespace hpx { namespace util { namespace detail
{
    struct vtable
    {
        template <typename T>
        BOOST_FORCEINLINE static std::type_info const& get_type()
        {
            return typeid(T);
        }
        typedef std::type_info const& (*get_type_t)();

        template <typename T>
        BOOST_FORCEINLINE static T& get(void** v)
        {
            if (sizeof(T) <= sizeof(void*))
            {
                return *reinterpret_cast<T*>(v);
            } else {
                return **reinterpret_cast<T**>(v);
            }
        }

        template <typename T>
        BOOST_FORCEINLINE static T const& get(void* const* v)
        {
            if (sizeof(T) <= sizeof(void*))
            {
                return *reinterpret_cast<T const*>(v);
            } else {
                return **reinterpret_cast<T* const*>(v);
            }
        }

        template <typename T>
        BOOST_FORCEINLINE static void default_construct(void** v)
        {
            if (sizeof(T) <= sizeof(void*))
            {
                new (v) T;
            } else {
                *v = new T;
            }
        }

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void construct(void** v, Arg&& arg)
        {
            if (sizeof(T) <= sizeof(void*))
            {
                new (v) T(std::forward<Arg>(arg));
            } else {
                *v = new T(std::forward<Arg>(arg));
            }
        }

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void reconstruct(void** v, Arg&& arg)
        {
            destruct<T>(v);
            construct<T, Arg>(v, std::forward<Arg>(arg));
        }

        template <typename T>
        BOOST_FORCEINLINE static void destruct(void** v)
        {
            get<T>(v).~T();
        }
        typedef void (*destruct_t)(void**);

        template <typename T>
        BOOST_FORCEINLINE static void delete_(void** v)
        {
            if (sizeof(T) <= sizeof(void*))
            {
                destruct<T>(v);
            } else {
                delete &get<T>(v);
            }
        }
        typedef void (*delete_t)(void**);
    };
}}}

#endif
