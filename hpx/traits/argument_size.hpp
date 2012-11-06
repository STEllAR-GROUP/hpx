//  Copyright (c) 2012 Vinay C Amatya
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_giMcvwgb6F6ljtAzZHfS99KSE6gn3KnvWt1TgTfV)
#define HPX_giMcvwgb6F6ljtAzZHfS99KSE6gn3KnvWt1TgTfV

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>

#include <boost/mpl/has_xxx.hpp>

#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

namespace hpx { namespace traits
{
    //////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        BOOST_MPL_HAS_XXX_TRAIT_DEF(uses_sizeof);
    }

    template <typename T, typename Enable>
    struct argument_size
    {
        typedef void uses_sizeof;

        static std::size_t call(T const& t)
        {
            return sizeof(T);
        }
    };

    // handle references
    template <typename T>
    struct argument_size<T&>
      : argument_size<T>
    {};

#if !defined(BOOST_NO_RVALUE_REFERENCES)
    template <typename T>
    struct argument_size<T&&>
      : argument_size<T>
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    // handle containers
    namespace detail
    {
        BOOST_MPL_HAS_XXX_TRAIT_DEF(value_type)
        BOOST_MPL_HAS_XXX_TRAIT_DEF(iterator)
        BOOST_MPL_HAS_XXX_TRAIT_DEF(size_type)
        BOOST_MPL_HAS_XXX_TRAIT_DEF(reference)

        template <typename T>
        struct is_container
          : boost::mpl::bool_<
                has_value_type<T>::value && has_iterator<T>::value &&
                has_size_type<T>::value && has_reference<T>::value>
        {};

        template <typename T>
        struct is_container<T&>
          : is_container<T>
        {};
    }

    template <typename T>
    struct argument_size<T,
        typename boost::enable_if<traits::detail::is_container<T> >::type>
    {
        template <typename T_>
        static std::size_t call(T_ const& v, boost::mpl::false_)
        {
            std::size_t sum = sizeof(T_);
            typename T_::const_iterator end = v.end();
            for (typename T_::const_iterator it = v.begin(); it != end; ++it)
                sum += argument_size<typename T_::value_type>::call(*it);
            return sum;
        }

        template <typename T_>
        static std::size_t call(T_ const& v, boost::mpl::true_)
        {
            return sizeof(T_) + v.size() * sizeof(typename T_::value_type);
        }

        static std::size_t call(T const& v)
        {
            typedef boost::mpl::bool_<
                traits::detail::has_uses_sizeof<
                    traits::argument_size<typename T::value_type>
                >::value> predicate;
            return call(v, predicate());
        }
    };

    //////////////////////////////////////////////////////////////////////////
    // handle Fusion sequences
    namespace detail
    {
        struct get_size
        {
            typedef std::size_t result_type;

            template <typename T>
            std::size_t operator()(std::size_t size, T const& t) const
            {
                return size + argument_size<T>::call(t);
            }
        };
    }

    template <typename T>
    struct argument_size<T,
        typename boost::enable_if<boost::fusion::traits::is_sequence<T> >::type>
    {
        static std::size_t call(T const& v)
        {
            std::size_t sum = sizeof(T);
            return boost::fusion::accumulate(v, sum, traits::detail::get_size());
        }
    };

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct argument_size<boost::shared_ptr<T> >
    {
        static std::size_t call(boost::shared_ptr<T> const& p)
        {
            return argument_size<T>::call(*p);
        }
    };

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct argument_size<boost::intrusive_ptr<T> >
    {
        static std::size_t call(boost::intrusive_ptr<T> const& p)
        {
            return argument_size<T>::call(*p);
        }
    };

    //////////////////////////////////////////////////////////////////////////
    /// Handle parcel 
    template <>
    struct argument_size<hpx::parcelset::parcel>
    {
        static std::size_t call(hpx::parcelset::parcel const& parcel)
        {
            return sizeof(parcel) + parcel.get_action()->get_argument_size();
        }
    };
}}

#endif
