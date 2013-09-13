//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_MPI_PARCELCACHE_HPP)
#define HPX_PARCELSET_MPI_PARCELCACHE_HPP

#include <hpx/hpx_fwd.hpp>
//#include <hpx/runtime/parcelset/mpi/allocator.hpp>
#include <hpx/runtime/parcelset/mpi/sender.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/scoped_unlock.hpp>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <map>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace mpi { namespace detail
{
    struct parcel_holder
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        parcel_holder(std::vector<parcel> const & parcels, std::vector<write_handler_type> const & handlers)
          : parcels_(parcels), handlers_(handlers)
        {}

        std::size_t size() const
        {
            BOOST_ASSERT(parcels_.size() == handlers_.size());
            return parcels_.size();
        }

        void clear()
        {
            parcels_.clear();
            handlers_.clear();
        }

        std::vector<parcel> parcels_;
        std::vector<write_handler_type> handlers_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct parcel_cache
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        //typedef std::vector<boost::shared_ptr<parcel_buffer> > parcel_buffers;
        typedef std::map<int, parcel_holder> parcel_holders;
        typedef hpx::lcos::local::spinlock mutex_type;

        parcel_cache(std::size_t num_threads, boost::uint64_t max_outbound_size)
          : archive_flags_(boost::archive::no_header)
          , max_outbound_size_(max_outbound_size)
        {
#ifdef BOOST_BIG_ENDIAN
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
            if (endian_out == "little")
                archive_flags_ |= util::endian_little;
            else if (endian_out == "big")
                archive_flags_ |= util::endian_big;
            else {
                BOOST_ASSERT(endian_out =="little" || endian_out == "big");
            }

            std::string array_optimization =
                get_config_entry("hpx.parcel.array_optimization", "1");
            if (boost::lexical_cast<int>(array_optimization) == 0)
                archive_flags_ |= util::disable_array_optimization;

            archive_flags_ |= util::disable_data_chunking;
        }

        ~parcel_cache()
        {
        }

        void set_parcel(parcel const& p, write_handler_type const & f, std::size_t idx = -1)
        {
            set_parcel(std::vector<parcel>(1, p), std::vector<write_handler_type>(1, f), idx);
        }

        void set_parcel(std::vector<parcel> const& pv,
            std::vector<write_handler_type> const& handlers, std::size_t idx = -1)
        {
            int dest(pv[0].get_destination_locality().get_rank());

            typedef parcel_holders::iterator iterator;
            {
                mutex_type::scoped_lock lk(parcel_holders_mtx_);

                iterator it = parcel_holders_.find(dest);
                if(it == parcel_holders_.end())
                {
                    parcel_holders_.insert(
                        std::make_pair(dest, parcel_holder(pv, handlers)));
                }
                else
                {
                    parcel_holder& ph = it->second;
                    ph.parcels_.insert(ph.parcels_.end(), pv.begin(), pv.end());
                    ph.handlers_.insert(ph.handlers_.end(), handlers.begin(), handlers.end());
                }
            }
        }

        template <typename Parcelport>
        std::size_t encode_parcels(boost::shared_ptr<parcel_buffer> & buffer,
            std::vector<parcel> const & pv, Parcelport& pp)
        {
            // collect argument sizes from parcels
            std::size_t arg_size = 0;

            buffer->rank_ = pv[0].get_destination_locality().get_rank();
            if (buffer->rank_ == -1)
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    "can't send over MPI without a known destination rank");
                return 0;
            }

            BOOST_FOREACH(parcel const & p, pv)
            {
                arg_size += traits::get_type_size(p);
            }

            util::high_resolution_timer timer;

            boost::shared_ptr<std::vector<char/*, allocator<char>*/ > >& b =
                 buffer->buffer_;

            b.reset(new std::vector<char/*, allocator<char>*/ >());
            b->reserve(arg_size + HPX_PARCEL_SERIALIZATION_OVERHEAD);

            performance_counters::parcels::data_point& datapoint = buffer->send_data_;
            datapoint.buffer_allocate_time_ = timer.elapsed_nanoseconds();

            // mark start of serialization
            boost::int64_t serialization_start = timer.elapsed_nanoseconds();

            // guard against serialization errors
            try {
                // Serialize the data
                HPX_STD_UNIQUE_PTR<util::binary_filter> filter(
                    pv[0].get_serialization_filter());

                int archive_flags = archive_flags_;
                if (filter.get() != 0) {
                    filter->set_max_length(buffer->buffer_->capacity());
                    archive_flags |= util::enable_compression;
                }

                util::portable_binary_oarchive archive(
                    *buffer->buffer_, filter.get(), archive_flags);

                std::size_t count = pv.size();
                archive << count; //-V128

                BOOST_FOREACH(parcel const& p, pv)
                {
                    archive << p;
                }

                arg_size = archive.bytes_written();

                // store the time required for serialization
                datapoint.serialization_time_ =
                    timer.elapsed_nanoseconds() - serialization_start;
            }
            catch (boost::archive::archive_exception const& e) {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem.
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization failed, caught "
                        "boost::archive::archive_exception: %s") % e.what()));
                return 0;
            }
            catch (boost::system::system_error const& e) {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization failed, caught "
                        "boost::system::system_error: %d (%s)") %
                            e.code().value() % e.code().message()));
                return 0;
            }
            catch (std::exception const& e) {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization failed, caught "
                        "std::exception: %s") % e.what()));
                return 0;
            }

            // make sure outgoing message is not larger than allowed
            if (buffer->buffer_->size() > max_outbound_size_)
            {
                HPX_THROW_EXCEPTION(serialization_error,
                    "mpi::detail::parcel_cache::set_parcel",
                    boost::str(boost::format(
                        "parcelport: parcel serialization created message larger "
                        "than allowed (created: %ld, allowed: %ld), consider"
                        "configuring larger hpx.parcel.max_message_size") %
                            buffer->buffer_->size() % max_outbound_size_));
                return 0;
            }

            datapoint.raw_bytes_ = buffer->buffer_->size();
            datapoint.bytes_ = arg_size;
            datapoint.num_parcels_ = pv.size();

            return arg_size;
        }

        template <typename Parcelport>
        std::list<boost::shared_ptr<sender> > get_senders(
            HPX_STD_FUNCTION<bool(int&)> const& tag_generator, MPI_Comm communicator,
            Parcelport& pp, std::size_t & num_requests, std::size_t max_requests)
        {
            std::list<boost::shared_ptr<sender> > res;
            parcel_holders phs;
            {
                mutex_type::scoped_try_lock lk0(parcel_holders_mtx_);
                if(!lk0)
                {
                    return res;
                }
                std::swap(parcel_holders_, phs);
            }
            bool early_exit = false;
            BOOST_FOREACH(parcel_holders::value_type & ph, phs)
            {
                if(ph.second.size() == 0) continue;
                if(num_requests >= max_requests)
                {
                    early_exit=true;
                    break;
                }
                int tag;
                if(!tag_generator(tag))
                {
                    // If no new tag could be generated, we need to put the
                    // remaining parcels back in the cache
                    early_exit=true;
                    break;
                }

                // collect outgoing data
                boost::shared_ptr<parcel_buffer> buffer(boost::make_shared<parcel_buffer>());
                std::vector<parcel> parcels;

                buffer->rank_ = ph.first;
                std::swap(buffer->handlers_, ph.second.handlers_);
                std::swap(parcels, ph.second.parcels_);

                std::size_t numbytes = encode_parcels(buffer, parcels, pp);
                res.push_back(
                    boost::make_shared<sender>(
                        header(
                            buffer->rank_
                          , tag
                          , static_cast<int>(buffer->buffer_->size())
                          , static_cast<int>(numbytes)
                        )
                      , buffer
                      , communicator
                    )
                );
                ++num_requests;
            }
            if(early_exit)
            {
                BOOST_FOREACH(parcel_holders::value_type & ph, phs)
                {
                    if(ph.second.size() == 0) continue;
                    set_parcel(ph.second.parcels_, ph.second.handlers_);
                    ph.second.clear();
                }
            }
            return boost::move(res);
        }

        mutex_type parcel_holders_mtx_;
        parcel_holders parcel_holders_;

        // archive flags
        int archive_flags_;

        boost::uint64_t max_outbound_size_;
    };
}}}}

#endif
