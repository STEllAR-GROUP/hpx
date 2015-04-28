//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DETAIL_POINTER_HPP
#define HPX_SERIALIZATION_DETAIL_POINTER_HPP

#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/runtime/serialization/string.hpp>
#include <hpx/traits/polymorphic_traits.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/mpl/eval_if.hpp>

namespace hpx { namespace serialization
{
    namespace detail
    {
        template <class Pointer>
        struct erase_ptr_helper : ptr_helper
        {
            typedef typename Pointer::element_type referred_type;

            erase_ptr_helper(Pointer&& t, Pointer& ptr)
                : t_(std::move(t))
            {
                ptr = t_;
            }

            Pointer t_;
        };

        template <class Pointer>
        class pointer_input_dispatcher
        {
            typedef typename Pointer::element_type referred_type;

            struct intrusive_polymorphic
            {
                static void call(input_archive& ar,
                    Pointer& ptr, boost::uint64_t pos)
                {
                    std::string name;
                    ar >> name;

                    Pointer t(
                        polymorphic_intrusive_factory::instance().
                            create<referred_type>(name)
                    );
                    ar >> *t;
                    register_pointer(
                          ar
                        , pos
                        , ptr_helper_ptr(
                              new detail::erase_ptr_helper<Pointer>(std::move(t), ptr)
                          )
                    );
                }
            };

            struct nonintrusive_polymorphic
            {
                static void call(input_archive& ar,
                    Pointer& ptr, boost::uint64_t pos)
                {
                    Pointer t (
                        polymorphic_nonintrusive_factory::instance().load<referred_type>(ar)
                    );
                    register_pointer(
                        ar
                      , pos
                      , ptr_helper_ptr(
                            new detail::erase_ptr_helper<Pointer>(std::move(t), ptr)
                        )
                    );
                }
            };

            struct usual
            {
                static void call(input_archive& ar,
                    Pointer& ptr, boost::uint64_t pos)
                {
                    //referred_type t;
                    Pointer t(new referred_type);
                    ar >> *t;
                    register_pointer(
                        ar
                      , pos
                      , ptr_helper_ptr(
                            new detail::erase_ptr_helper<Pointer>(std::move(t), ptr)
                        )
                    );
                }
            };

        public:
            typedef typename boost::mpl::eval_if<
                hpx::traits::is_intrusive_polymorphic<referred_type>,
                    boost::mpl::identity<intrusive_polymorphic>,
                    boost::mpl::eval_if<
                        hpx::traits::is_nonintrusive_polymorphic<referred_type>,
                            boost::mpl::identity<nonintrusive_polymorphic>,
                            boost::mpl::identity<usual>
                  >
            >::type type;
        };

        template <class Pointer>
        class pointer_output_dispatcher
        {
            typedef typename Pointer::element_type referred_type;

            struct intrusive_polymorphic
            {
                static void call(output_archive& ar,
                    Pointer& ptr)
                {
                    const std::string name = access::get_name(ptr.get());
                    ar << name;
                    ar << *ptr;
                }
            };

            struct usual
            {
                static void call(output_archive& ar,
                    Pointer& ptr)
                {
                    ar << *ptr;
                }
            };

        public:
            typedef typename boost::mpl::if_<
                hpx::traits::is_intrusive_polymorphic<referred_type>,
                    intrusive_polymorphic,
                    usual
                >::type type;
        };

        // forwarded serialize pointer functions
        template <typename Pointer> BOOST_FORCEINLINE
        void serialize_pointer(output_archive & ar, Pointer ptr, unsigned)
        {
            bool valid = static_cast<bool>(ptr);
            ar << valid;
            if(valid)
            {
                boost::uint64_t cur_pos = current_pos(ar);
                boost::uint64_t pos = track_pointer(ar, ptr.get());
                ar << pos;
                if(pos == boost::uint64_t(-1))
                {
                    ar << cur_pos;
                    detail::pointer_output_dispatcher<Pointer>::type::call(ar, ptr);
                }
            }
        }

        template <class Pointer> BOOST_FORCEINLINE
        void serialize_pointer(input_archive& ar, Pointer& ptr, unsigned)
        {
            bool valid = false;
            ar >> valid;

            if(valid)
            {
                boost::uint64_t pos = 0;
                ar >> pos;
                if(pos == boost::uint64_t(-1))
                {
                    pos = 0;
                    ar >> pos;
                    detail::pointer_input_dispatcher<Pointer>::type::call(ar, ptr, pos);
                }
                else
                {
                    detail::erase_ptr_helper<Pointer> & helper =
                        tracked_pointer<detail::erase_ptr_helper<Pointer> >(ar, pos);
                    ptr = helper.t_;
                }
            }
        }

    } // detail
}}

#endif // HPX_SERIALIZATION_DETAIL_POINTER_HPP
