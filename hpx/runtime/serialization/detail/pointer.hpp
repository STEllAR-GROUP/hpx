//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DETAIL_POINTER_HPP
#define HPX_SERIALIZATION_DETAIL_POINTER_HPP

#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/basic_archive.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_id_factory.hpp>
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
                static Pointer call(input_archive& ar)
                {
                    std::string name;
                    ar >> name;

                    Pointer t(polymorphic_intrusive_factory::instance().
                        create<referred_type>(name));
                    ar >> *t;
                    return t;
                }
            };

            struct polymorphic_with_id
            {
                static Pointer call(input_archive& ar)
                {
                    boost::uint32_t id;
                    ar >> id;

                    Pointer t(polymorphic_id_factory::create<referred_type>(id));
                    ar >> *t;
                    return t;
                }
            };

            struct nonintrusive_polymorphic
            {
                static Pointer call(input_archive& ar)
                {
                    return Pointer(polymorphic_nonintrusive_factory::
                        instance().load<referred_type>(ar));
                }
            };

            struct usual
            {
                static Pointer call(input_archive& ar)
                {
                    Pointer t(constructor_selector<referred_type>::create(ar));
                    return t;
                }
            };

        public:
            typedef typename boost::mpl::eval_if<
                hpx::traits::is_serialized_with_id<referred_type>,
                    boost::mpl::identity<polymorphic_with_id>,
                    boost::mpl::eval_if<
                        hpx::traits::is_intrusive_polymorphic<referred_type>,
                            boost::mpl::identity<intrusive_polymorphic>,
                            boost::mpl::eval_if<
                                hpx::traits::is_nonintrusive_polymorphic<referred_type>,
                                    boost::mpl::identity<nonintrusive_polymorphic>,
                                    boost::mpl::identity<usual>
                        >
                    >
                >::type type;
        };

        template <class Pointer>
        class pointer_output_dispatcher
        {
            typedef typename Pointer::element_type referred_type;

            struct intrusive_polymorphic
            {
                static void call(output_archive& ar, const Pointer& ptr)
                {
                    const std::string name = access::get_name(ptr.get());
                    ar << name;
                    ar << *ptr;
                }
            };

            struct polymorphic_with_id
            {
                static void call(output_archive& ar, const Pointer& ptr)
                {
                    const boost::uint32_t id =
                        polymorphic_id_factory::get_id(
                            access::get_name(ptr.get()));
                    ar << id;
                    ar << *ptr;
                }
            };

            struct usual
            {
                static void call(output_archive& ar, const Pointer& ptr)
                {
                    ar << *ptr;
                }
            };

        public:
            typedef typename boost::mpl::if_<
                hpx::traits::is_serialized_with_id<referred_type>,
                    polymorphic_with_id,
                    typename boost::mpl::if_<
                        hpx::traits::is_intrusive_polymorphic<referred_type>,
                            intrusive_polymorphic,
                            usual
                    >::type
                >::type type;
        };

        // forwarded serialize pointer functions
        template <typename Pointer> BOOST_FORCEINLINE
        void serialize_pointer_tracked(output_archive & ar, const Pointer& ptr)
        {
            bool valid = static_cast<bool>(ptr);
            ar << valid;
            if(valid)
            {
                boost::uint64_t cur_pos =
                    static_cast<basic_archive<output_archive>&>(ar).current_pos();
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
        void serialize_pointer_tracked(input_archive& ar, Pointer& ptr)
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
                    Pointer temp = detail::pointer_input_dispatcher<
                        Pointer>::type::call(ar);
                    register_pointer(ar, pos, ptr_helper_ptr(
                            new detail::erase_ptr_helper<Pointer>
                                (std::move(temp), ptr)));
                }
                else
                {
                    detail::erase_ptr_helper<Pointer> & helper =
                        tracked_pointer<detail::erase_ptr_helper<Pointer> >(ar, pos);
                    ptr = helper.t_;
                }
            }
        }

        template <typename Pointer> BOOST_FORCEINLINE
        void serialize_pointer_untracked(output_archive & ar, const Pointer& ptr)
        {
            bool valid = static_cast<bool>(ptr);
            ar << valid;
            if(valid)
            {
                detail::pointer_output_dispatcher<Pointer>::type::call(ar, ptr);
            }
        }

        template <class Pointer> BOOST_FORCEINLINE
        void serialize_pointer_untracked(input_archive& ar, Pointer& ptr)
        {
            bool valid = false;
            ar >> valid;
            if(valid)
            {
                ptr = detail::pointer_input_dispatcher<Pointer>::type::call(ar);
            }
        }

    } // detail
}}

#endif // HPX_SERIALIZATION_DETAIL_POINTER_HPP
