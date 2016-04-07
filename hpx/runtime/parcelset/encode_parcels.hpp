//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011-2015 Thomas Heller
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_ENCODE_PARCEL_HPP
#define HPX_PARCELSET_ENCODE_PARCEL_HPP

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/integer/endian.hpp>
#include <hpx/traits/is_chunk_allocator.hpp>

#include <boost/cstdint.hpp>

#include <memory>
#include <vector>

namespace hpx
{
    namespace parcelset
    {
        namespace detail
        {
            ///////////////////////////////////////////////////////////////////
            inline char
            to_digit(int number)
            {
                char number_tmp = static_cast<char>(number);
                if (number >= 0 && number <= 9)
                {
                    return static_cast<char>(number_tmp + '0');
                }
                return static_cast<char>(number_tmp - 10 + 'A');
            }

            inline void
            convert_byte(boost::uint8_t b, char* buffer, char const* end)
            {
                *buffer++ = to_digit((b & 0xF0) >> 4);
                *buffer++ = to_digit(b & 0x0F);
            }

            template <typename Buffer>
            std::string binary_archive_content(Buffer const& buffer)
            {
                std::string result;
                if (LPT_ENABLED(debug))
                {
                    result.reserve(buffer.data_.size() * 2 + 1);
                    for (boost::uint8_t byte: buffer.data_)
                    {
                        char b[3] = { 0 };
                        convert_byte(byte, &b[0], &b[3]);
                        result += b;
                    }
                }
                return result;
            }

            template <typename Buffer>
            void encode_finalize(Buffer & buffer, std::size_t arg_size)
            {
                buffer.size_ = buffer.data_.size();
                buffer.data_size_ = arg_size;

                LPT_(debug) << binary_archive_content(buffer);

                performance_counters::parcels::data_point& data = buffer.data_point_;
                data.bytes_ = arg_size;
                data.raw_bytes_ = buffer.data_.size();

                // prepare chunk data for transmission, the transmission_chunks data
                // first holds all zero-copy, then all non-zero-copy chunk infos
                typedef typename Buffer::transmission_chunk_type
                    transmission_chunk_type;
                typedef typename Buffer::count_chunks_type
                    count_chunks_type;

                std::vector<transmission_chunk_type>& chunks =
                    buffer.transmission_chunks_;

                chunks.clear();
                chunks.reserve(buffer.chunks_.size());

                std::size_t index = 0;
                for (serialization::serialization_chunk& c : buffer.chunks_)
                {
                    if (c.type_ == serialization::chunk_type_pointer)
                        chunks.push_back(transmission_chunk_type(index, c.size_));
                    ++index;
                }

                buffer.num_chunks_ = count_chunks_type(
                    static_cast<boost::uint32_t>(chunks.size()),
                    static_cast<boost::uint32_t>(buffer.chunks_.size() - chunks.size())
                );

                if (!chunks.empty()) {
                    // the remaining number of chunks are non-zero-copy
                    for (serialization::serialization_chunk& c : buffer.chunks_)
                    {
                        if (c.type_ == serialization::chunk_type_index) {
                            chunks.push_back(
                                transmission_chunk_type(c.data_.index_, c.size_));
                        }
                    }
                }
            }

            inline std::size_t
            get_archive_size(parcel const& p, boost::uint32_t flags,
                boost::uint32_t dest_locality_id,
                std::vector<serialization::serialization_chunk>* chunks)
            {
                // gather the required size for the archive
                hpx::serialization::detail::size_gatherer_container gather_size;
                hpx::serialization::output_archive archive(
                    gather_size, flags, dest_locality_id, chunks);
                archive << p;
                return gather_size.size();
            }
        }

        template <typename Buffer, typename NewGids>
        std::size_t
        encode_parcels(parcel const * ps, std::size_t num_parcels, Buffer & buffer,
            int archive_flags_, boost::uint64_t max_outbound_size, NewGids new_gids)
        {
            HPX_ASSERT(buffer.data_.empty());
            // collect argument sizes from parcels
            std::size_t arg_size = 0;
            boost::uint32_t dest_locality_id =
                ps[0].destination_locality_id();

            std::size_t parcels_sent = 0;

            std::size_t parcels_size = 1;
            if(num_parcels != std::size_t(-1))
                parcels_size = num_parcels;

            // guard against serialization errors
            try {
                try {
                    std::unique_ptr<serialization::binary_filter> filter(
                        ps[0].get_serialization_filter());

                    int archive_flags = archive_flags_;
                    if (filter.get() != 0)
                        archive_flags |= serialization::enable_compression;

                    // Get the chunk size from the allocator if it supports it
                    size_t chunk_default = hpx::traits::default_chunk_size<
                            typename Buffer::allocator_type
                        >::call(buffer.data_.get_allocator());

                    // preallocate data
                    for (/**/; parcels_sent != parcels_size; ++parcels_sent)
                    {
                        if (arg_size >= max_outbound_size)
                            break;
                        arg_size += detail::get_archive_size(ps[parcels_sent],
                            archive_flags, dest_locality_id, &buffer.chunks_);
                    }

                    buffer.data_.reserve((std::max)(chunk_default, arg_size));

                    // mark start of serialization
                    util::high_resolution_timer timer;

                    {
                        // Serialize the data
                        if (filter.get() != 0)
                            filter->set_max_length(buffer.data_.capacity());

                        serialization::output_archive archive(
                            buffer.data_
                          , archive_flags
                          , dest_locality_id
                          , &buffer.chunks_
                          , filter.get()
                          , new_gids);

                        if(num_parcels != std::size_t(-1))
                            archive << parcels_sent; //-V128

                        for(std::size_t i = 0; i != parcels_sent; ++i)
                        {
                            LPT_(debug) << ps[i];
                            archive << ps[i];
                        }

                        arg_size = archive.bytes_written();
                    }

                    // store the time required for serialization
                    buffer.data_point_.serialization_time_ =
                        timer.elapsed_nanoseconds();
                }
                catch (hpx::exception const& e) {
                    LPT_(fatal)
                        << "encode_parcels: "
                           "caught hpx::exception: "
                        << e.what();
                    hpx::report_error(boost::current_exception());
                    return 0;
                }
                catch (boost::system::system_error const& e) {
                    LPT_(fatal)
                        << "encode_parcels: "
                           "caught boost::system::error: "
                        << e.what();
                    hpx::report_error(boost::current_exception());
                    return 0;
                }
                catch (boost::exception const&) {
                    LPT_(fatal)
                        << "encode_parcels: "
                           "caught boost::exception";
                    hpx::report_error(boost::current_exception());
                    return 0;
                }
                catch (std::exception const& e) {
                    // We have to repackage all exceptions thrown by the
                    // serialization library as otherwise we will loose the
                    // e.what() description of the problem, due to slicing.
                    boost::throw_exception(boost::enable_error_info(
                        hpx::exception(serialization_error, e.what())));
                    return 0;
                }
            }
            catch (...) {
                LPT_(fatal)
                        << "encode_parcels: "
                       "caught unknown exception";
                hpx::report_error(boost::current_exception());
                    return 0;
            }

            buffer.data_point_.num_parcels_ = parcels_sent;
            detail::encode_finalize(buffer, arg_size);

            return parcels_sent;
        }
    }
}

#endif
