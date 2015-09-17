//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_ACCESS_HPP
#define HPX_SERIALIZATION_ACCESS_HPP

#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/polymorphic_traits.hpp>
#include <hpx/traits/has_serialize.hpp>

#include <boost/type_traits/is_empty.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/identity.hpp>

#include <string>

namespace hpx { namespace serialization
{
    namespace detail
    {
        template <class Archive, class T>
        BOOST_FORCEINLINE void serialize_force_adl(Archive& ar, T& t, unsigned)
        {
            serialize(ar, t, 0);
        }
    }

    class access
    {
        template <class T>
        class serialize_dispatcher
        {
            struct intrusive_polymorphic
            {
                // both following template functions are viable
                // to call right overloaded function according to T constness
                // and to prevent calling templated version of serialize function
                static void call(hpx::serialization::input_archive& ar, T& t, unsigned)
                {
                    t.serialize(ar, 0);
                }

                static void call(hpx::serialization::output_archive& ar,
                    const T& t, unsigned)
                {
                    t.serialize(ar, 0);
                }
            };

            struct non_intrusive
            {
                // this additional indirection level is needed to
                // force ADL on the second phase of template lookup.
                // call of serialize function directly from base_object
                // finds only serialize-member function and doesn't
                // perform ADL
                template <class Archive>
                static void call(Archive& ar, T& t, unsigned)
                {
                    detail::serialize_force_adl(ar, t, 0);
                }
            };

            struct empty
            {
                template <class Archive>
                static void call(Archive& ar, T& t, unsigned)
                {
                }
            };

            struct intrusive_usual
            {
                template <class Archive>
                static void call(Archive& ar, T& t, unsigned)
                {
                    t.serialize(ar, 0);
                }
            };

        public:
            typedef typename boost::mpl::eval_if<
                hpx::traits::is_intrusive_polymorphic<T>,
                    boost::mpl::identity<intrusive_polymorphic>,
                        boost::mpl::eval_if<
                            hpx::traits::has_serialize<T>,
                                boost::mpl::identity<intrusive_usual>,
                                boost::mpl::eval_if<
                                    boost::is_empty<T>,
                                        boost::mpl::identity<empty>,
                                        boost::mpl::identity<non_intrusive>
                                >
                        >
            >::type type;
        };

    public:
        template <class Archive, class T>
        static void serialize(Archive& ar, T& t, unsigned)
        {
            serialize_dispatcher<T>::type::call(ar, t, 0);
        }

        template <typename Archive, typename T> BOOST_FORCEINLINE
        static void save_base_object(Archive & ar, const T & t, unsigned)
        {
            // explicitly specify virtual function
            // of base class to avoid infinite recursion
            t.T::save(ar, 0);
        }

        template <typename Archive, typename T> BOOST_FORCEINLINE
        static void load_base_object(Archive & ar, T & t, unsigned)
        {
            // explicitly specify virtual function
            // of base class to avoid infinite recursion
            t.T::load(ar, 0);
        }

        template <typename T> BOOST_FORCEINLINE
        static std::string get_name(const T* t)
        {
            return t->hpx_serialization_get_name();
        }
    };

}}

#endif
