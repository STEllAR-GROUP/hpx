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

#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/integer/endian.hpp>

namespace hpx
{
    namespace parcelset
    {
        template <typename Buffer>
        void encode_finalize(Buffer & buffer, std::size_t arg_size)
        {
            buffer.size_ = buffer.data_.size();
            buffer.data_size_ = arg_size;

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
            BOOST_FOREACH(util::serialization_chunk& c, buffer.chunks_)
            {
                if (c.type_ == util::chunk_type_pointer)
                    chunks.push_back(transmission_chunk_type(index, c.size_));
                ++index;
            }

            buffer.num_chunks_ = count_chunks_type(
                static_cast<boost::uint32_t>(chunks.size()),
                static_cast<boost::uint32_t>(buffer.chunks_.size() - chunks.size())
            );

            if (!chunks.empty()) {
                // the remaining number of chunks are non-zero-copy
                BOOST_FOREACH(util::serialization_chunk& c, buffer.chunks_)
                {
                    if (c.type_ == util::chunk_type_index) {
                        chunks.push_back(
                            transmission_chunk_type(c.data_.index_, c.size_));
                    }
                }
            }
        }

        template <typename Buffer>
        void
        encode_parcel(parcel const & p, Buffer & buffer,
            int archive_flags_, std::size_t max_outbound_size)
        {
            // collect argument sizes from parcels
            std::size_t arg_size = traits::get_type_size(p);
            boost::uint32_t dest_locality_id = p.get_destination_locality_id();

            // guard against serialization errors
            try {
                try {
                    // preallocate data
                    buffer.data_.reserve(arg_size);

                    // mark start of serialization
                    util::high_resolution_timer timer;

                    {
                        // Serialize the data
                        HPX_STD_UNIQUE_PTR<util::binary_filter> filter(
                            p.get_serialization_filter());

                        int archive_flags = archive_flags_;
                        if (filter.get() != 0) {
                            filter->set_max_length(buffer.data_.capacity());
                            archive_flags |= util::enable_compression;
                        }

                        util::portable_binary_oarchive archive(
                            buffer.data_
                          , &buffer.chunks_
                          , dest_locality_id
                          , filter.get()
                          , archive_flags);

                        archive << p;

                        arg_size = archive.bytes_written();
                    }

                    // store the time required for serialization
                    buffer.data_point_.serialization_time_ = timer.elapsed_nanoseconds();
                }
                catch (hpx::exception const& e) {
                    LPT_(fatal)
                        << "encode_parcels: "
                           "caught hpx::exception: "
                        << e.what();
                    hpx::report_error(boost::current_exception());
                }
                catch (boost::system::system_error const& e) {
                    LPT_(fatal)
                        << "encode_parcels: "
                           "caught boost::system::error: "
                        << e.what();
                    hpx::report_error(boost::current_exception());
                }
                catch (boost::exception const&) {
                    LPT_(fatal)
                        << "encode_parcels: "
                           "caught boost::exception";
                    hpx::report_error(boost::current_exception());
                }
                catch (std::exception const& e) {
                    // We have to repackage all exceptions thrown by the
                    // serialization library as otherwise we will loose the
                    // e.what() description of the problem, due to slicing.
                    boost::throw_exception(boost::enable_error_info(
                        hpx::exception(serialization_error, e.what())));
                }
            }
            catch (...) {
                LPT_(fatal)
                        << "encode_parcels: "
                       "caught unknown exception";
                hpx::report_error(boost::current_exception());
            }

            buffer.data_point_.num_parcels_ = 1;
            encode_finalize(buffer, arg_size);
        }
    }
}

#endif
