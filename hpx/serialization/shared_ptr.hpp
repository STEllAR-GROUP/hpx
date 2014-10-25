//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_SHARED_PTR_HPP
#define HPX_SERIALIZATION_SHARED_PTR_HPP

#include <hpx/config.hpp>
#include <hpx/serialization/serialize.hpp>

#include <boost/shared_ptr.hpp>

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

            boost::shared_ptr<T> t_;
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
        bool valid = ptr;
        ar << valid;
        if(valid)
        {
            boost::uint64_t cur_pos = current_pos(ar);
            boost::uint64_t pos = track_pointer(ar, ptr.get());
            ar << pos;
            if(pos == boost::uint64_t(-1))
            {
                ar << cur_pos;
                ar << *ptr;
            }
        }
    }
}}

#endif
