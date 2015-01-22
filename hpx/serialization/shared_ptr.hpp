//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_SHARED_PTR_HPP
#define HPX_SERIALIZATION_SHARED_PTR_HPP

#include <hpx/config.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/polymorphic_intrusive_factory.hpp>
#include <hpx/serialization/polymorphic_nonintrusive_factory.hpp>
#include <hpx/traits/polymorphic_traits.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/mpl/eval_if.hpp>

namespace hpx { namespace serialization {

    namespace detail
    {
        template <typename T>
        struct shared_ptr_helper : ptr_helper
        {
            shared_ptr_helper(T && t, boost::shared_ptr<T> & ptr)
              : t_(new T(std::move(t)))
            {
                ptr = t_;
            }

            // for polymorphic version
            shared_ptr_helper(boost::shared_ptr<T>&& t, boost::shared_ptr<T>& ptr)
              : t_(std::move(t))
            {
                ptr = t_;
            }

            boost::shared_ptr<T> t_;
        };

        template <class T>
        class shared_ptr_input_dispatcher
        {
          struct intrusive_polymorphic
          {
            static void serialize(input_archive& ar,
                boost::shared_ptr<T>& ptr, boost::uint64_t pos)
            {
              std::string name;
              ar >> name;

              boost::shared_ptr<T> t(
                  polymorphic_intrusive_factory::instance().
                    create<T>(name)
              );
              ar >> *t;
              register_pointer(
                  ar
                , pos
                , HPX_STD_UNIQUE_PTR<detail::ptr_helper>(
                      new detail::shared_ptr_helper<T>(std::move(t), ptr)
                  )
              );
            }
          };

          struct nonintrusive_polymorphic
          {
            static void serialize(input_archive& ar,
                boost::shared_ptr<T>& ptr, boost::uint64_t pos)
            {
              boost::shared_ptr<T> t (
                polymorphic_nonintrusive_factory::instance().load<T>(ar)
              );
              register_pointer(
                  ar
                , pos
                , HPX_STD_UNIQUE_PTR<detail::ptr_helper>(
                      new detail::shared_ptr_helper<T>(std::move(t), ptr)
                  )
              );
            }
          };

          struct usual
          {
            static void serialize(input_archive& ar,
                boost::shared_ptr<T>& ptr, boost::uint64_t pos)
            {
              T t;
              ar >> t;
              register_pointer(
                  ar
                , pos
                , HPX_STD_UNIQUE_PTR<detail::ptr_helper>(
                      new detail::shared_ptr_helper<T>(std::move(t), ptr)
                  )
              );
            }
          };

        public:
          typedef typename boost::mpl::eval_if<
            hpx::traits::is_intrusive_polymorphic<T>,
              boost::mpl::identity<intrusive_polymorphic>,
              boost::mpl::eval_if<
                hpx::traits::is_nonintrusive_polymorphic<T>,
                  boost::mpl::identity<nonintrusive_polymorphic>,
                  boost::mpl::identity<usual>
              >
          >::type type;
        };

        template <class T>
        class shared_ptr_output_dispatcher
        {
          struct intrusive_polymorphic
          {
            static void serialize(output_archive& ar,
                boost::shared_ptr<T>& ptr) {
              const std::string name = access::get_name(ptr.get());
              ar << name;
              ar << *ptr;
            }
          };

          struct usual
          {
            static void serialize(output_archive& ar,
                boost::shared_ptr<T>& ptr) {
              ar << *ptr;
            }
          };

        public:
          typedef typename boost::mpl::if_<
            hpx::traits::is_intrusive_polymorphic<T>,
              intrusive_polymorphic,
              usual
            >::type type;
        };
    }

    // load shared_ptr ...
    template <typename T>
    void serialize(input_archive & ar, boost::shared_ptr<T> & ptr, unsigned)
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
                detail::shared_ptr_input_dispatcher<T>::type::serialize(ar, ptr, pos);
            }
            else
            {
                detail::shared_ptr_helper<T> & helper =
                  tracked_pointer<detail::shared_ptr_helper<T> >(ar, pos);
                ptr = helper.t_;
            }
        }
    }

    // save shared_ptr ...
    template <typename T>
    void serialize(output_archive & ar, boost::shared_ptr<T> ptr, unsigned)
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
                detail::shared_ptr_output_dispatcher<T>::type::serialize(ar, ptr);
            }
        }
    }
}}

#endif
