//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_RAW_PTR_HPP
#define HPX_SERIALIZATION_RAW_PTR_HPP

#include <hpx/runtime/serialization/detail/pointer.hpp>

namespace hpx { namespace serialization
{
    namespace detail
    {
        template <class T>
        struct raw_ptr_type
        {
            typedef T element_type;

            raw_ptr_type(T* t)
                : t(t)
            {}

            T* get() const
            {
                return t;
            }

            T& operator*() const
            {
                return *t;
            }

            operator bool() const
            {
                return t != 0;
            }

        private:
            T* t;
        };

        template <class T>
        struct raw_ptr_proxy
        {
            raw_ptr_proxy(T*& t):
                t(t)
            {}

            raw_ptr_proxy(T* const & t):
                t(const_cast<T*&>(t))
            {}

            void serialize(output_archive& ar) const
            {
                serialize_pointer_untracked(ar, raw_ptr_type<T>(t));
            }

            void serialize(input_archive& ar)
            {
                raw_ptr_type<T> ptr(t);
                serialize_pointer_untracked(ar, ptr);
                t = ptr.get();
            }

            T*& t;
        };

        template <class T> HPX_FORCEINLINE
        raw_ptr_proxy<T> raw_ptr(T*& t)
        {
            return raw_ptr_proxy<T>(t);
        }

        template <class T> HPX_FORCEINLINE
        raw_ptr_proxy<T> raw_ptr(T* const & t)
        {
            return raw_ptr_proxy<T>(t);
        }

        // allow raw_ptr_type to be serialized as prvalue
        template <class T> HPX_FORCEINLINE
        output_archive & operator<<(output_archive & ar, raw_ptr_proxy<T> t)
        {
            t.serialize(ar);
            return ar;
        }

        template <class T> HPX_FORCEINLINE
        input_archive & operator>>(input_archive & ar, raw_ptr_proxy<T> t)
        {
            t.serialize(ar);
            return ar;
        }

        template <class T> HPX_FORCEINLINE
        output_archive & operator&(output_archive & ar, raw_ptr_proxy<T> t) //-V524
        {
            t.serialize(ar);
            return ar;
        }

        template <class T> HPX_FORCEINLINE
        input_archive & operator&(input_archive & ar, raw_ptr_proxy<T> t) //-V524
        {
            t.serialize(ar);
            return ar;
        }
}}}

#endif
