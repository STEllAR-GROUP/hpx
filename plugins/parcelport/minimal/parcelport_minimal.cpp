//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#include <hpx/hpx_fwd.hpp>

#include <hpx/plugins/parcelport_factory.hpp>
#include <hpx/util/command_line_handling.hpp>

// parcelport
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>
// Local parcelport plugin

#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>


namespace hpx { namespace parcelset { namespace policies { namespace minimal
{
    // this example uses an integer rank for locality representation internally
    struct locality
    {
        static const char *type()
        {
            return "minimal";
        }

        explicit locality(boost::int32_t rank) : rank_(rank) {}

        locality() : rank_(-1) {}

        boost::int32_t rank() const
        {
            return rank_;
        }

        // some condition marking this locality as valid
        operator util::safe_bool<locality>::result_type() const
        {
            return util::safe_bool<locality>()(rank_ != -1);
        }

        void save(util::portable_binary_oarchive & ar) const
        {
          // save the state
          ar.save(rank_);
        }

        void load(util::portable_binary_iarchive & ar)
        {
          // load the state
          ar.load(rank_);
        }

    private:
        friend bool operator==(locality const & lhs, locality const & rhs)
        {
            return lhs.rank_ == rhs.rank_;
        }

        friend bool operator<(locality const & lhs, locality const & rhs)
        {
            return lhs.rank_ < rhs.rank_;
        }

        friend std::ostream & operator<<(std::ostream & os, locality const & loc)
        {
            boost::io::ios_flags_saver ifs(os);
            os << loc.rank_;

            return os;
        }

        boost::int32_t rank_;
    };

    class parcelport
      : public parcelset::parcelport
    {
    private:
        static parcelset::locality here()
        {
            return
                parcelset::locality(
                    locality(
                       0 // whatever "here" means
                    )
                );
        }

    public:
        parcelport(util::runtime_configuration const& ini,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread)
          : parcelset::parcelport(ini, here(), "minimal")
          , archive_flags_(boost::archive::no_header)
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
                HPX_ASSERT(endian_out =="little" || endian_out == "big");
            }

            if (!this->allow_array_optimizations()) {
                archive_flags_ |= util::disable_array_optimization;
                archive_flags_ |= util::disable_data_chunking;
            }
            else {
                if (!this->allow_zero_copy_optimizations())
                    archive_flags_ |= util::disable_data_chunking;
            }
        }

        ~parcelport()
        {
        }

        bool can_bootstrap() const
        {
            return false/* return true if this pp can be used at bootstrapping, otherwise omit */;
        }

        /// Return the name of this locality
        std::string get_locality_name() const
        {
            return "minimal"/* whatever this means for your pp */;
        }

        parcelset::locality
        agas_locality(util::runtime_configuration const & ini) const
        {
            return
                parcelset::locality(
                    locality(
                        0/* whatever it takes to address AGAS */
                    )
                );
        }

        parcelset::locality create_locality() const
        {
            return parcelset::locality(locality());
        }

        void put_parcels(std::vector<parcelset::locality> dests,
            std::vector<parcel> parcels,
            std::vector<write_handler_type> handlers)
        {
            // Put a vector of parcels
        }

        void send_early_parcel(parcelset::locality const & dest, parcel& p)
        {
            // Only necessary if your PP an be used at bootstrapping
            put_parcel(dest, p
              , boost::bind(
                    &parcelport::early_write_handler
                  , this
                  , ::_1
                  , p
                )
            );
        }

        util::io_service_pool* get_thread_pool(char const* name)
        {
            return 0;
        }

        // This parcelport doesn't maintain a connection cache
        boost::int64_t get_connection_cache_statistics(
            connection_cache_statistics_type, bool reset)
        {
            return 0;
        }

        void remove_from_connection_cache(parcelset::locality const& loc)
        {}

        bool run(bool blocking = true)
        {
            // This should start the receiving side of your PP
            return true;
        }

        void stop(bool blocking = true)
        {
            // Stop receiving and sending of parcels
        }

        void enable(bool new_state)
        {
             // enable/disable sending and receiving of parcels
        }

        void put_parcel(parcelset::locality const & dest, parcel p,
            write_handler_type f)
        {
            // Send a single parcel, after succesful sending, f should be called.
        }

        bool do_background_work(std::size_t num_thread)
        {
            // This is called whenever a HPX OS thread is idling, can be used to poll for incoming messages
          return true;
        }

    private:
        int archive_flags_;

        // Only needed for bootstrapping
        void early_write_handler(
            boost::system::error_code const& ec, parcel const & p)
        {
            if (ec) {
                // all errors during early parcel handling are fatal
                boost::exception_ptr exception =
                    hpx::detail::get_exception(hpx::exception(ec),
                        "mpi::early_write_handler", __FILE__, __LINE__,
                        "error while handling early parcel: " +
                            ec.message() + "(" +
                            boost::lexical_cast<std::string>(ec.value()) +
                            ")" + parcelset::dump_parcel(p));

                hpx::report_error(exception);
            }
        }
    };
}}}}

namespace hpx { namespace traits
{
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.minimal]
    //      ...
    //      priority = 100
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::minimal::parcelport>
    {
        static char const* priority()
        {
            return "100";
        }
        static void init(int *argc, char ***argv, util::command_line_handling &cfg)
        {
            // This is used to initialize your parcelport, for example check for availability of devices etc.
        }

        static char const* call()
        {
            return
                "key = value\n"
                "key2 = value2\n"
                ;
        }
    };
}}

HPX_REGISTER_PARCELPORT(
    hpx::parcelset::policies::minimal::parcelport,
    minimal);

