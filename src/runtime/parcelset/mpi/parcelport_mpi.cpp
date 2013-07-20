//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/parcelset/mpi/parcelport.hpp>
#include <hpx/runtime/parcelset/mpi/acceptor.hpp>
#include <hpx/runtime/parcelset/mpi/sender.hpp>
#include <hpx/runtime/parcelset/mpi/receiver.hpp>
#include <hpx/runtime/parcelset/detail/call_for_each.hpp>
#include <hpx/util/mpi_environment.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/logging.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/hash.hpp>
#include <hpx/components/security/parcel_suffix.hpp>
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/signed_type.hpp>
#endif

#include <boost/version.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace mpi
{
    void decode_message(
        std::vector<char> const & parcel_data,
        parcelport& pp);

    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : parcelset::parcelport(ini),
        io_service_pool_(1, on_start_thread, on_stop_thread, "parcel_pool_mpi", "-mpi"),
        parcel_cache_(ini.get_os_thread_count()),
        stopped(false),
        next_tag(2)
        /*
        acceptor_(NULL),
        connection_cache_(ini.get_max_connections(), ini.get_max_connections_per_loc())
        */
    {
        if (here_.get_type() != connection_mpi) {
            HPX_THROW_EXCEPTION(network_error, "mpi::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
    }

    parcelport::~parcelport()
    {
        std::cout << util::mpi_environment::rank() << " parcelport::~parcelport()\n";
    }

    util::io_service_pool* parcelport::get_thread_pool(char const* name)
    {
        if (0 == std::strcmp(name, io_service_pool_.get_name()))
            return &io_service_pool_;
        return 0;
    }

    bool parcelport::run(bool blocking)
    {
        //std::cout << util::mpi_environment::rank() <<  " running MPI parcelport\n";
        io_service_pool_.run(false);    // start pool

        boost::asio::io_service& io_service = io_service_pool_.get_io_service();

        io_service.post(HPX_STD_BIND(&parcelport::handle_messages, this->shared_from_this()));

        if (blocking)
        {
            std::cout << util::mpi_environment::rank() <<  " running MPI parcelport blocked\n";
            io_service_pool_.join();
        }

        return true;
    }

    void parcelport::stop(bool blocking)
    {
        // make sure no more work is pending, wait for service pool to get empty
        std::cout << util::mpi_environment::rank() << " parcelport::stop() start\n";
        io_service_pool_.stop();
        stopped = true;
        if (blocking) {
            io_service_pool_.join();

            io_service_pool_.clear();
        }
        //MPI_Barrier(util::mpi_environment::communicator());
        std::cout << util::mpi_environment::rank() << " parcelport::stop() finished\n";
    }

    void parcelport::handle_messages()
    {
        MPI_Comm communicator = util::mpi_environment::communicator();
        acceptor ack(communicator);
        typedef std::list<boost::shared_ptr<receiver> > receivers_type;
        typedef std::list<boost::shared_ptr<sender> > senders_type;
        receivers_type receivers;
        senders_type senders;

        while(!stopped)
        {
            std::pair<bool, header> next(ack.next_header());
            if(next.first)
            {
                receivers.push_back(boost::make_shared<receiver>(next.second, communicator));
            }
            for(receivers_type::iterator it = receivers.begin(); it != receivers.end();)
            {
                if((*it)->done(*this))
                {
                    it = receivers.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            for(senders_type::iterator it = senders.begin(); it != senders.end();)
            {
                if((*it)->done())
                {
                    free_tags.push_back((*it)->tag());
                    it = senders.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            senders_type const & tmp = parcel_cache_.get_senders(HPX_STD_BIND(&parcelport::get_next_tag, this->shared_from_this()), communicator);
            senders.insert(senders.end(), tmp.begin(), tmp.end());
        }
        // cancel all remaining requests
        std::cout << util::mpi_environment::rank() << " handle_messages stopped\n";
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::put_parcels(std::vector<parcel> const & parcels,
        std::vector<write_handler_type> const& handlers)
    {
        /*
        std::cout << util::mpi_environment::rank() <<  " MPI parcelport: put parcels\n";
        std::cout << util::mpi_environment::rank() << " destination: " << parcels[0].get_destination_locality() << "\n";
        */

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
            BOOST_ASSERT(locality_id == parcels[i].get_destination_locality());
        }
#endif
        parcel_cache_.set_parcel(parcels, handlers);

    }

    void parcelport::put_parcel(parcel const& p, write_handler_type f)
    {
        /*
        std::cout << util::mpi_environment::rank() <<  " MPI parcelport: put parcel\n";
        std::cout << util::mpi_environment::rank() << " destination: " << p.get_destination_locality() << "\n";
        */
        parcel_cache_.set_parcel(p, f);
    }

    void parcelport::send_early_parcel(parcel& p)
    {
        /*
        std::cout << util::mpi_environment::rank() << " MPI parcelport: send early parcel\n";
        std::cout << util::mpi_environment::rank() << " destination: " << p.get_destination_locality() << "\n";
        */
        naming::locality const& l = p.get_destination_locality();
        error_code ec;

        parcel_cache_.set_parcel(p, write_handler_type(), 0);
    }

    void decode_message(
        std::vector<char> const & parcel_data,
        parcelport& pp
        //performance_counters::parcels::data_point receive_data,
    )
    {
        // protect from un-handled exceptions bubbling up
        try {
            try {
                // mark start of serialization
                util::high_resolution_timer timer;
                boost::int64_t overall_add_parcel_time = 0;

                // De-serialize the parcel data
                util::portable_binary_iarchive archive(parcel_data,
                    parcel_data.size(), boost::archive::no_header);

                std::size_t parcel_count = 0;
                archive >> parcel_count;
                BOOST_ASSERT(parcel_count > 0);
                for(std::size_t i = 0; i != parcel_count; ++i)
                {
                    // de-serialize parcel and add it to incoming parcel queue
                    parcel p;
                    archive >> p;
                    // make sure this parcel ended up on the right locality
                    BOOST_ASSERT(p.get_destination_locality() == pp.here());

                    // be sure not to measure add_parcel as serialization time
                    boost::int64_t add_parcel_time = timer.elapsed_nanoseconds();

                    pp.add_received_parcel(p);
                    overall_add_parcel_time += timer.elapsed_nanoseconds() -
                        add_parcel_time;
                }

                // complete received data with parcel count
                /*
                receive_data.num_parcels_ = parcel_count;
                receive_data.raw_bytes_ = archive.bytes_read();
                */

                // store the time required for serialization
                /*
                receive_data.serialization_time_ = timer.elapsed_nanoseconds() -
                    overall_add_parcel_time;
                */

                //pp.add_received_data(receive_data);
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
