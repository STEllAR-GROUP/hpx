// Copyright (c) 2012 Bryce Adelstein-Lelbach
// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ZERO_COPY_OUTPUT_OCT_15_2012_0136PM)
#define HPX_UTIL_ZERO_COPY_OUTPUT_OCT_15_2012_0136PM

#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/foreach.hpp>
#include <boost/integer/endian.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>

namespace hpx { namespace util
{
    // Helper wrapper allowing to abstract the concrete serialization strategy.
    struct zero_copy_output
    {
        zero_copy_output()
          : out_priority_(0),
            out_chunks_(0)
        {}

        // Serialize a vector of parcels (this is a template to decouple from
        // the parcel layer.
        template <typename T>
        void save(std::vector<T> const& pv)
        {
            // we choose the highest priority of all parcels for this message
            threads::thread_priority priority = threads::thread_priority_default;

            // push the first two values on to the list of buffers
            buffers_.push_back(boost::asio::buffer(&out_priority_, sizeof(out_priority_)));
            buffers_.push_back(boost::asio::buffer(&out_chunks_, sizeof(out_chunks_)));

            {
                typedef container_device<std::vector<char> > io_device_type;
                boost::iostreams::stream<io_device_type> io(out_buffer_);

                portable_binary_oarchive archive(io);
                archive << pv;
            }

            // set the priority and the overall size 
            BOOST_FOREACH(T const& p, pv)
            {
                priority = (std::max)(p.get_thread_priority(), priority);
            }

            out_priority_ = boost::integer::ulittle8_t(priority);
            chunk_sizes_.push_back(out_buffer_.size());
            out_chunks_ = chunk_sizes_.size();

            buffers_.push_back(boost::asio::buffer(chunk_sizes_));
            buffers_.push_back(boost::asio::buffer(out_buffer_));
        }

        // reset all internal data structures for next data preparation
        void clear()
        {
            out_priority_ = 0;
            out_chunks_ = 0;
            chunk_sizes_.clear();
            out_buffer_.clear();
            buffers_.clear();
        }

        // access the vector of buffers used for serialization
        std::vector<boost::asio::const_buffer> const& get_buffers() const
        {
            return buffers_;
        }

        std::size_t bytes() const
        {
            return out_buffer_.size();
        }

    private:
        boost::integer::ulittle8_t out_priority_;
        boost::integer::ulittle64_t out_chunks_;
        std::vector<boost::integer::ulittle64_t> chunk_sizes_;
        std::vector<char> out_buffer_;
        std::vector<boost::asio::const_buffer> buffers_;
    };
}}

#endif
