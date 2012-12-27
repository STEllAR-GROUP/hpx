//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SHMEM_INTERPROCESS_ERRORS_NOV_25_2012_0703PM)
#define HPX_PARCELSET_SHMEM_INTERPROCESS_ERRORS_NOV_25_2012_0703PM

#include <boost/interprocess/errors.hpp>
#include <boost/system/system_error.hpp>
#include <boost/static_assert.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        char const* const error_names[] =
        {
             "no_error",
             "system_error",     // system generated error; if possible, is translated
                                 // to one of the more specific errors below.
             "other_error",      // library generated error
             "security_error",   // includes access rights, permissions failures
             "read_only_error",
             "io_error",
             "path_error",
             "not_found_error",
             "busy_error",       // implies trying again might succeed
             "already_exists_error",
             "not_empty_error",
             "is_directory_error",
             "out_of_space_error",
             "out_of_memory_error",
             "out_of_resource_error",
             "lock_error",
             "sem_error",
             "mode_error",
             "size_error",
             "corrupted_error",
             "not_such_file_or_directory",
             "invalid_argument"
#if BOOST_VERSION >= 104800
           , "timeout_when_locking_error"
#if BOOST_VERSION >= 104900
           , "timeout_when_waiting_error"
#endif
#endif
        };

        class interprocess_category : public boost::system::error_category
        {
        public:
            const char* name() const
            {
                return "Shmem";
            }

            std::string message(int value) const
            {
                using namespace boost::interprocess;

                // make sure our assumption about error codes is reasonably correct
#if BOOST_VERSION < 104800
                BOOST_STATIC_ASSERT(sizeof(error_names)/sizeof(error_names[0]) ==
                    invalid_argument+1);
#elif BOOST_VERSION < 104900
                BOOST_STATIC_ASSERT(sizeof(error_names)/sizeof(error_names[0]) ==
                    timeout_when_locking_error+1);
#else
                BOOST_STATIC_ASSERT(sizeof(error_names)/sizeof(error_names[0]) ==
                    timeout_when_waiting_error+1);
#endif

                if (value >= no_error && value <= timeout_when_waiting_error)
                    return std::string("Shmem(") + error_names[value] + ")";

                return "Shmem(unknown_error)";
            }
        };
    }

    inline boost::system::error_category const& get_interprocess_category()
    {
        static detail::interprocess_category instance;
        return instance;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::system::error_code
    make_error_code(boost::interprocess::error_code_t e)
    {
        return boost::system::error_code(
            static_cast<int>(e), get_interprocess_category());
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace system
{
    // make sure our errors get recognized by the Boost.System library
    template<> struct is_error_code_enum<boost::interprocess::error_code_t>
    {
        static const bool value = true;
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_SHMEM_THROWS_IF(ec, code)                                         \
        if (&ec != &boost::system::throws) ec = code;                         \
        else boost::asio::detail::throw_error(code);                          \
    /**/
#define HPX_SHMEM_RESET_EC(ec)                                                \
        if (&ec != &boost::system::throws) ec = boost::system::error_code();  \
    /**/

#endif
