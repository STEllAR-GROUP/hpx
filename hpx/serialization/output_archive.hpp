//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_OUTPUT_ARCHIVE_HPP
#define HPX_SERIALIZATION_OUTPUT_ARCHIVE_HPP

#include <hpx/config.hpp>

#include <hpx/serialization/archive.hpp>
#include <hpx/serialization/output_container.hpp>
#include <hpx/serialization/polymorphic_nonintrusive_factory.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace serialization {
    struct HPX_ALWAYS_EXPORT output_archive
      : archive<output_archive>
    {
        typedef archive<output_archive> base_type;
        typedef std::map<void *, std::size_t> pointer_tracker;

        template <typename Container>
        static HPX_STD_UNIQUE_PTR<container> make_container(Container & buffer)
        {
            return HPX_STD_UNIQUE_PTR<container>(new output_container<Container>(buffer));
        }

        template <typename Container>
        output_archive(Container & buffer)
          : base_type(0, make_container(buffer))
        {}

        template <typename T>
        void invoke_impl(T const & t)
        {
            save(t);
        }

        template <typename T>
        typename boost::disable_if<
            boost::is_integral<T>
        >::type
        save(T const & t)
        {
            save_bitwise(t,
                typename hpx::traits::is_bitwise_serializable<T>::type());
        }

        template <typename T>
        void save_bitwise(T const & t, boost::mpl::false_)
        {
            save_nonintrusively_polymorphic(t,
                hpx::traits::is_nonintrusive_polymorphic<T>());
        }

        template <typename T>
        void save_bitwise(T const & t, boost::mpl::true_)
        {
            BOOST_STATIC_ASSERT_MSG(!boost::is_abstract<T>::value,
                "Can not bitwise serialize a class that is abstract");
            if(disable_array_optimization())
            {
                serialize(*this, const_cast<T &>(t), 0);
            }
            else
            {
                save_binary(&t, sizeof(t));
            }
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const & t, boost::mpl::false_)
        {
            serialize(*this, const_cast<T &>(t), 0);
        }

        template <typename T>
        void save_nonintrusively_polymorphic(T const & t, boost::mpl::true_)
        {
            polymorphic_nonintrusive_factory::instance().save(*this, t);
        }

        template <typename T>
        typename boost::enable_if<
            boost::is_integral<T>
        >::type
        save(T t)
        {
            save_integral(t,
                typename boost::is_unsigned<T>::type());
        }

        template <typename T>
        void save_integral(T val, boost::mpl::false_)
        {
            save_impl(static_cast<boost::int64_t>(val));
        }

        template <typename T>
        void save_integral(T val, boost::mpl::true_)
        {
            save_impl(static_cast<boost::uint64_t>(val));
        }

        void save(float f)
        {
            save_binary(&f, sizeof(float));
        }

        void save(double d)
        {
            save_binary(&d, sizeof(double));
        }

        void save(char c)
        {
            save_binary(&c, sizeof(char));
        }

        void save(bool b)
        {
            HPX_ASSERT(0 == static_cast<int>(b) || 1 == static_cast<int>(b));
            save_binary(&b, sizeof(bool));
        }

        void save_impl(boost::int64_t t);
        void save_impl(boost::uint64_t t);
        void save_binary(void const * address, std::size_t count);

        std::size_t track_pointer(void * p)
        {
            pointer_tracker::iterator it = pointer_tracker_.find(p);
            if(it == pointer_tracker_.end())
            {
                pointer_tracker_.insert(std::make_pair(p, size_));
                return npos;
            }
            return it->second;
        }

        pointer_tracker pointer_tracker_;
    };

    std::size_t track_pointer(output_archive & ar, void * pos);
}}

#endif
