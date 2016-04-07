//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file exception.hpp

#if !defined(HPX_ERROR_CODE_MAR_24_2008_0929AM)
#define HPX_ERROR_CODE_MAR_24_2008_0929AM

#include <hpx/config.hpp>
#include <hpx/error.hpp>
#include <hpx/exception_fwd.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/system/error_code.hpp>

#include <stdexcept>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

#if !defined(BOOST_SYSTEM_NOEXCEPT)
#define BOOST_SYSTEM_NOEXCEPT HPX_NOEXCEPT
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    /// \cond NODETAIL
    namespace detail
    {
        class hpx_category : public boost::system::error_category
        {
        public:
            const char* name() const BOOST_SYSTEM_NOEXCEPT
            {
                return "HPX";
            }

            std::string message(int value) const
            {
                if (value >= success && value < last_error)
                    return std::string("HPX(") + error_names[value] + ")"; //-V108
                if (value & system_error_flag)
                    return std::string("HPX(system_error)");
                return "HPX(unknown_error)";
            }
        };

        struct lightweight_hpx_category : hpx_category {};

        // this doesn't add any text to the exception what() message
        class hpx_category_rethrow : public boost::system::error_category
        {
        public:
            const char* name() const BOOST_SYSTEM_NOEXCEPT
            {
                return "";
            }

            std::string message(int) const HPX_NOEXCEPT
            {
                return "";
            }
        };

        struct lightweight_hpx_category_rethrow : hpx_category_rethrow {};

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT boost::exception_ptr access_exception(error_code const&);

        ///////////////////////////////////////////////////////////////////////
        struct command_line_error : std::logic_error
        {
            explicit command_line_error(char const* msg)
              : std::logic_error(msg)
            {}

            explicit command_line_error(std::string const& msg)
              : std::logic_error(msg)
            {}
        };
    } // namespace detail
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns generic HPX error category used for new errors.
    inline boost::system::error_category const& get_hpx_category()
    {
        static detail::hpx_category instance;
        return instance;
    }

    /// \brief Returns generic HPX error category used for errors re-thrown
    ///        after the exception has been de-serialized.
    inline boost::system::error_category const& get_hpx_rethrow_category()
    {
        static detail::hpx_category_rethrow instance;
        return instance;
    }

    /// \cond NOINTERNAL
    inline boost::system::error_category const&
    get_lightweight_hpx_category()
    {
        static detail::lightweight_hpx_category instance;
        return instance;
    }

    inline boost::system::error_category const& get_hpx_category(throwmode mode)
    {
        switch(mode) {
        case rethrow:
            return get_hpx_rethrow_category();

        case lightweight:
        case lightweight_rethrow:
            return get_lightweight_hpx_category();

        case plain:
        default:
            break;
        }
        return get_hpx_category();
    }

    inline boost::system::error_code
    make_system_error_code(error e, throwmode mode = plain)
    {
        return boost::system::error_code(
            static_cast<int>(e), get_hpx_category(mode));
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::system::error_condition
    make_error_condition(error e, throwmode mode)
    {
        return boost::system::error_condition(
            static_cast<int>(e), get_hpx_category(mode));
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief A hpx::error_code represents an arbitrary error condition.
    ///
    /// The class hpx::error_code describes an object used to hold error code
    /// values, such as those originating from the operating system or other
    /// low-level application program interfaces.
    ///
    /// \note Class hpx::error_code is an adjunct to error reporting by
    /// exception
    ///
    class error_code : public boost::system::error_code //-V690
    {
    public:
        /// Construct an object of type error_code.
        ///
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        explicit error_code(throwmode mode = plain)
          : boost::system::error_code(make_system_error_code(success, mode))
        {}

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        inline explicit error_code(error e, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        inline error_code(error e, char const* func, char const* file,
            long line, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, char const* msg, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, char const* msg, char const* func,
                char const* file, long line, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, std::string const& msg, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the hpx::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               hpx::error_code belongs to the error category
        ///               \a hpx_category (if mode is \a plain, this is the
        ///               default) or to the category \a hpx_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        inline error_code(error e, std::string const& msg, char const* func,
                char const* file, long line, throwmode mode = plain);

        /// Return a reference to the error message stored in the hpx::error_code.
        ///
        /// \throws nothing
        inline std::string get_message() const;

        /// \brief Clear this error_code object.
        /// The postconditions of invoking this method are
        /// * value() == hpx::success and category() == hpx::get_hpx_category()
        void clear()
        {
            this->boost::system::error_code::assign(success, get_hpx_category());
            exception_ = boost::exception_ptr();
        }

        /// Assignment operator for error_code
        ///
        /// \note This function maintains the error category of the left hand
        ///       side if the right hand side is a success code.
        inline error_code& operator=(error_code const& rhs);

    private:
        friend boost::exception_ptr detail::access_exception(error_code const&);
        friend class exception;
        friend error_code make_error_code(boost::exception_ptr const&);

        inline error_code(int err, hpx::exception const& e);
        inline explicit error_code(boost::exception_ptr const& e);

        boost::exception_ptr exception_;
    };

    /// @{
    /// \brief Returns a new error_code constructed from the given parameters.
    inline error_code
    make_error_code(error e, throwmode mode = plain)
    {
        return error_code(e, mode);
    }
    inline error_code
    make_error_code(error e, char const* func, char const* file, long line,
        throwmode mode = plain)
    {
        return error_code(e, func, file, line, mode);
    }

    /// \brief Returns error_code(e, msg, mode).
    inline error_code
    make_error_code(error e, char const* msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }
    inline error_code
    make_error_code(error e, char const* msg, char const* func,
        char const* file, long line, throwmode mode = plain)
    {
        return error_code(e, msg, func, file, line, mode);
    }

    /// \brief Returns error_code(e, msg, mode).
    inline error_code
    make_error_code(error e, std::string const& msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }
    inline error_code
    make_error_code(error e, std::string const& msg, char const* func,
        char const* file, long line, throwmode mode = plain)
    {
        return error_code(e, msg, func, file, line, mode);
    }
    inline error_code
    make_error_code(boost::exception_ptr const& e)
    {
        return error_code(e);
    }
    ///@}

    /// \brief Returns error_code(hpx::success, "success", mode).
    inline error_code
    make_success_code(throwmode mode = plain)
    {
        return error_code(mode);
    }

    ///////////////////////////////////////////////////////////////////////////
    // \cond NOINTERNAL
    inline error_code::error_code(error e, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(e, "", mode);
    }

    inline error_code::error_code(error e, char const* func,
            char const* file, long line, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight)) {
            exception_ = detail::get_exception(e, "", mode, func, file, line);
        }
    }

    inline error_code::error_code(error e, char const* msg, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(e, msg, mode);
    }

    inline error_code::error_code(error e, char const* msg,
            char const* func, char const* file, long line, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight)) {
            exception_ = detail::get_exception(e, msg, mode, func, file, line);
        }
    }

    inline error_code::error_code(error e, std::string const& msg,
            throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(e, msg, mode);
    }

    inline error_code::error_code(error e, std::string const& msg,
            char const* func, char const* file, long line, throwmode mode)
      : boost::system::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight)) {
            exception_ = detail::get_exception(e, msg, mode, func, file, line);
        }
    }

    inline error_code::error_code(int err, hpx::exception const& e)
    {
        this->boost::system::error_code::assign(err, get_hpx_category());
        exception_ = get_exception_ptr(e);
    }

    inline error_code::error_code(boost::exception_ptr const& e)
      : boost::system::error_code(make_system_error_code(get_error(e), rethrow)),
        exception_(e)
    {}

    ///////////////////////////////////////////////////////////////////////////
    inline std::string error_code::get_message() const
    {
        if (exception_) {
            try {
                boost::rethrow_exception(exception_);
            }
            catch (boost::exception const& be) {
                return dynamic_cast<std::exception const*>(&be)->what();
            }
        }
        return get_error_what(*this);   // provide at least minimal error text
    }

    ///////////////////////////////////////////////////////////////////////////
    inline error_code& error_code::operator=(error_code const& rhs)
    {
        if (this != &rhs) {
            if (rhs.value() == success) {
                // if the rhs is a success code, we maintain our throw mode
                this->boost::system::error_code::operator=(
                    make_success_code(
                        (category() == get_lightweight_hpx_category()) ?
                            hpx::lightweight : hpx::plain));
            }
            else {
                this->boost::system::error_code::operator=(rhs);
            }
            exception_ = rhs.exception_;
        }
        return *this;
    }
    /// \endcond
}

#include <hpx/throw_exception.hpp>
#include <hpx/config/warnings_suffix.hpp>

#endif
