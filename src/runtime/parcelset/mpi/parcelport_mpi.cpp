//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/parcelset/mpi/parcelport.hpp>
#include <hpx/runtime/parcelset/detail/call_for_each.hpp>
#include <hpx/util/mpi_environment.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/hash.hpp>
#include <hpx/components/security/parcel_suffix.hpp>
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/signed_type.hpp>
#endif

#include <boost/version.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

#if defined(HPX_HAVE_SECURITY)
namespace hpx
{
    /// \brief Verify the certificate in the given byte sequence
    ///
    /// \param data      The full received message buffer, assuming that it
    ///                  has a parcel_suffix appended.
    /// \param parcel_id The parcel id of the first parcel in side the message
    ///
    HPX_API_EXPORT bool verify_parcel_suffix(std::vector<char> const& data,
        naming::gid_type& parcel_id, error_code& ec = throws);
}
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace mpi
{
    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : parcelset::parcelport(ini),
        io_service_pool_(1, on_start_thread, on_stop_thread, "parcel_pool_mpi", "-mpi"),
        max_requests_(ini.get_max_mpi_requests()),
        parcel_cache_(ini.get_os_thread_count(), this->get_max_message_size()),
        stopped_(false),
        handling_messages_(false),
        next_tag_(1)
    {
        if (here_.get_type() != connection_mpi) {
            HPX_THROW_EXCEPTION(network_error, "mpi::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
    }

    parcelport::~parcelport()
    {
    }

    util::io_service_pool* parcelport::get_thread_pool(char const* name)
    {
        if (0 == std::strcmp(name, io_service_pool_.get_name()))
            return &io_service_pool_;
        return 0;
    }

    bool parcelport::run(bool blocking)
    {
        io_service_pool_.run(false);    // start pool

        acceptor_.run(util::mpi_environment::communicator());
        do_background_work();      // schedule message handler

        if (blocking)
        {
            io_service_pool_.join();
        }

        return true;
    }

    void parcelport::stop(bool blocking)
    {
        // make sure no more work is pending, wait for service pool to get empty
        io_service_pool_.stop();
        stopped_ = true;
        if (blocking) {
            io_service_pool_.join();
            io_service_pool_.clear();
        }
    }

    // Make sure all pending requests are handled
    void parcelport::do_background_work()
    {
        if (stopped_)
            return;

        // Atomically set handling_messages_ to true, if another work item hasn't
        // started executing before us.
        bool false_ = false;
        if (!handling_messages_.compare_exchange_strong(false_, true))
            return;

        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        io_service.post(HPX_STD_BIND(&parcelport::handle_messages, this->shared_from_this()));
    }

    namespace detail
    {
        struct handling_messages
        {
            handling_messages(boost::atomic<bool>& handling_messages_flag)
              : handling_messages_(handling_messages_flag)
            {}

            ~handling_messages()
            {
                handling_messages_.store(false);
            }

            boost::atomic<bool>& handling_messages_;
        };
    }

    void parcelport::handle_messages()
    {
        detail::handling_messages hm(handling_messages_);       // reset on exit

        MPI_Comm communicator = util::mpi_environment::communicator();

        bool bootstrapping = hpx::is_starting();
        bool has_work = true;

        std::size_t num_requests = 0;

        hpx::util::high_resolution_timer t;

        // We let the message handling loop spin for another 2 seconds to avoid the
        // costs involved with posting it to asio
        while(bootstrapping || (!stopped_ && has_work) || (t.elapsed() < 2.0))
        {
            // add new receive requests
            if(num_requests < max_requests_)
            {
                std::pair<bool, header> next(acceptor_.next_header());
                if(next.first)
                {
                    receivers_.push_back(boost::make_shared<receiver>(next.second, communicator, *this));
                    ++num_requests;
                }
            }

            // handle all receive requests
            for(receivers_type::iterator it = receivers_.begin(); it != receivers_.end(); /**/)
            {
                if((*it)->done(*this))
                {
                    it = receivers_.erase(it);
                    --num_requests;
                }
                else
                {
                    ++it;
                }
            }
            has_work = !receivers_.empty();

            // add new send requests
            using HPX_STD_PLACEHOLDERS::_1;
            senders_.splice(senders_.end(), parcel_cache_.get_senders(
                HPX_STD_BIND(&parcelport::get_next_tag, this->shared_from_this(), _1),
                communicator, *this, num_requests, max_requests_));

            // handle all send requests
            for(senders_type::iterator it = senders_.begin(); it != senders_.end(); /**/)
            {
                if((*it)->done(*this))
                {
                    free_tags_.push_back((*it)->tag());
                    it = senders_.erase(it);
                    --num_requests;
                }
                else
                {
                    ++it;
                }
            }

            if (!has_work)
                has_work = !senders_.empty();

            if (bootstrapping)
                bootstrapping = hpx::is_starting();

            if(has_work)
            {
                t.restart();
            }
            else
            {
#if defined( WIN32 ) || defined( _WIN32 ) || defined( __WIN32__ ) || defined( __CYGWIN__ )
                Sleep( 1 );
#elif defined( BOOST_HAS_PTHREADS )
                // g++ -Wextra warns on {} or {0}
                struct timespec rqtp = { 0, 0 };

                // POSIX says that timespec has tv_sec and tv_nsec
                // But it doesn't guarantee order or placement

                rqtp.tv_sec = 0;
                rqtp.tv_nsec = 1000;

                nanosleep( &rqtp, 0 );
#endif
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::put_parcels(std::vector<parcel> const & parcels,
        std::vector<write_handler_type> const& handlers)
    {
        do_background_work();      // schedule message handler
        if (parcels.size() != handlers.size())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::put_parcels",
                "mismatched number of parcels and handlers");
            return;
        }

        naming::locality locality_id = parcels[0].get_destination_locality();

#if defined(HPX_DEBUG)
        // make sure all parcels go to the same locality
        for (std::size_t i = 1; i != parcels.size(); ++i)
        {
            BOOST_ASSERT(locality_id.get_rank() == parcels[i].get_destination_locality().get_rank());
        }
#endif

        parcel_cache_.set_parcel(parcels, handlers);
    }

    void parcelport::put_parcel(parcel const& p, write_handler_type const& f)
    {
        do_background_work();      // schedule message handler
        parcel_cache_.set_parcel(p, f);
    }

    void parcelport::send_early_parcel(parcel& p)
    {
        do_background_work();      // schedule message handler
        parcel_cache_.set_parcel(p, write_handler_type(), 0);
    }

    void decode_message(
        std::vector<char/*, allocator<char>*/ > const & parcel_data,
        boost::uint64_t inbound_data_size,
        parcelport& pp,
        performance_counters::parcels::data_point& receive_data
    )
    {
        unsigned archive_flags = boost::archive::no_header;
        if (!pp.allow_array_optimizations())
            archive_flags |= util::disable_array_optimization;
                
        archive_flags |= util::disable_data_chunking;

        // protect from un-handled exceptions bubbling up
        try {
            try {
                // mark start of serialization
                util::high_resolution_timer timer;
                boost::int64_t overall_add_parcel_time = 0;

                // De-serialize the parcel data
                util::portable_binary_iarchive archive(parcel_data,
                    inbound_data_size, archive_flags);

                std::size_t parcel_count = 0;
                archive >> parcel_count; //-V128

                BOOST_ASSERT(parcel_count > 0);
                for(std::size_t i = 0; i != parcel_count; ++i)
                {
                    // de-serialize parcel and add it to incoming parcel queue
                    parcel p;
                    archive >> p;

                    // make sure this parcel ended up on the right locality
                    BOOST_ASSERT(p.get_destination_locality().get_rank() == pp.here().get_rank());

                    // be sure not to measure add_parcel as serialization time
                    boost::int64_t add_parcel_time = timer.elapsed_nanoseconds();

                    pp.add_received_parcel(p);
                    overall_add_parcel_time += timer.elapsed_nanoseconds() -
                        add_parcel_time;
                }

                // complete received data with parcel count
                receive_data.num_parcels_ = parcel_count;
                receive_data.raw_bytes_ = archive.bytes_read();

                // store the time required for serialization
                receive_data.serialization_time_ = timer.elapsed_nanoseconds() -
                    overall_add_parcel_time;

                pp.add_received_data(receive_data);
            }
            catch (hpx::exception const& e) {
                LPT_(error)
                    << "decode_message(mpi): caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::system::system_error const& e) {
                LPT_(error)
                    << "decode_message(mpi): caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::exception const&) {
                LPT_(error)
                    << "decode_message(mpi): caught boost::exception.";
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
            LPT_(error)
                << "decode_message(mpi): caught unknown exception.";
            hpx::report_error(boost::current_exception());
        }
    }
}}}

#endif
