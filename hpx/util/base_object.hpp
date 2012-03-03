////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2002 Robert Ramey 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_DFDB7D1E_F238_4803_BA38_16F1CFD77622)
#define HPX_DFDB7D1E_F238_4803_BA38_16F1CFD77622

#include <hpx/util/void_cast.hpp>

#include <boost/serialization/base_object.hpp>

namespace hpx { namespace util
{

namespace detail
{

// Get the base type for a given derived type preserving the constness.
template <
    typename Base
  , typename Derived
>
struct base_cast
{
    typedef typename boost::mpl::if_<
        boost::is_const<Derived>,
        const Base,
        Base
    >::type type;

    BOOST_STATIC_ASSERT
        (boost::is_const<type>::value == boost::is_const<Derived>::value);
};

// Only register void casts if the types are polymorphic.
template <
    typename Base
  , typename Derived
>
struct base_register
{
    struct polymorphic
    {
        static void const* invoke()
        {
            Base const* const b = 0;
            Derived const* const d = 0;
            return &hpx::util::void_cast_register_nonvirt(d, b);
        }
    };

    struct non_polymorphic
    {
        static void const* invoke()
        {
            return 0;
        }
    };

    static void const* invoke()
    {
        typedef typename boost::mpl::eval_if<
            boost::is_polymorphic<Base>
          , boost::mpl::identity<polymorphic>
          , boost::mpl::identity<non_polymorphic>
        >::type type;
        return type::invoke();
    }
};

} 

template <
    typename Base
  , typename Derived
>
typename detail::base_cast<Base, Derived>::type& base_object_nonvirt(
    Derived &d
    )
{
    typedef typename detail::base_cast<Base, Derived>::type type;
    detail::base_register<type, Derived>::invoke();
    return boost::serialization::access::cast_reference<type, Derived>(d);
}

}}

#endif // HPX_DFDB7D1E_F238_4803_BA38_16F1CFD77622

