//  copyright (c) 2005
//  troy d. straszheim <troy@resophonic.com>
//  http://www.resophonic.com
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_VARIANT_HPP
#define HPX_SERIALIZATION_VARIANT_HPP

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <boost/mpl/front.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/empty.hpp>

#include <boost/variant.hpp>

namespace hpx { namespace serialization
{
    struct variant_save_visitor :
        boost::static_visitor<>
    {
        variant_save_visitor(output_archive& ar) :
            m_ar(ar)
        {}

        template<class T>
        void operator()(T const & value) const
        {
            m_ar << value;
        }
    private:
        output_archive & m_ar;
    };

    template<class S>
    struct variant_impl {

        struct load_null {
            template<class V>
            static void invoke(
                input_archive& /*ar*/,
                int /*which*/,
                V & /*v*/
            ){}
        };

        struct load_impl {
            template<class V>
            static void invoke(
                input_archive& ar,
                int which,
                V & v
            ){
                if(which == 0){
                    // note: A non-intrusive implementation (such as this one)
                    // necessary has to copy the value.  This wouldn't be necessary
                    // with an implementation that de-serialized to the address of the
                    // aligned storage included in the variant.
                    typedef typename boost::mpl::front<S>::type head_type;
                    head_type value;
                    ar >> value;
                    v = value;
                    return;
                }
                typedef typename boost::mpl::pop_front<S>::type type;
                variant_impl<type>::load(ar, which - 1, v);
            }
        };

        template<class V>
        static void load(
            input_archive& ar,
            int which,
            V & v
        ){
            typedef typename boost::mpl::eval_if<boost::mpl::empty<S>,
                boost::mpl::identity<load_null>,
                boost::mpl::identity<load_impl>
            >::type typex;
            typex::invoke(ar, which, v);
        }

    };

    template<BOOST_VARIANT_ENUM_PARAMS(class T)>
    void save(output_archive& ar,
            boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)> const & v, unsigned)
    {
        int which = v.which();
        ar << which;
        variant_save_visitor visitor(ar);
        v.apply_visitor(visitor);
    }

    template<BOOST_VARIANT_ENUM_PARAMS(class T)>
    void load(input_archive& ar,
            boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)>& v, unsigned)
    {
        int which;
        typedef typename boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)>::types types;
        ar >> which;
        if(which >=  boost::mpl::size<types>::value)
            // this might happen if a type was removed from the list of variant types
            HPX_THROW_EXCEPTION(serialization_error
              , "load<Archive, Variant, version>"
              , "type was removed from the list of variant types");
        variant_impl<types>::load(ar, which, v);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE((template
        <BOOST_VARIANT_ENUM_PARAMS(class T)>),
            (boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)>));

} // namespace serialization
} // namespace boost

#endif //HPX_SERIALIZATION_VARIANT_HPP
