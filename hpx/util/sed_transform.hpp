////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_EC1602ED_CCC2_471C_BC28_1DBB98902F40)
#define HPX_EC1602ED_CCC2_471C_BC28_1DBB98902F40

#include <hpx/config.hpp>

#include <memory>
#include <string>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

namespace hpx { namespace util
{

/// Parse a sed command.
///
/// \param input    [in] The content to parse.
/// \param search   [out] If the parsing is successful, this string is set to
///                 the search expression.
/// \param search   [out] If the parsing is successful, this string is set to
///                 the replace expression.
///
/// \returns \a true if the parsing was successful, false otherwise.
///
/// \note Currently, only supports search and replace syntax (s/search/replace/)
HPX_EXPORT bool parse_sed_expression(
    std::string const& input
  , std::string& search
  , std::string& replace
    );

/// An unary function object which applies a sed command to its subject and
/// returns the resulting string.
///
/// \note Currently, only supports search and replace syntax (s/search/replace/)
struct HPX_EXPORT sed_transform
{
  private:
    struct command;

    std::shared_ptr<command> command_;

  public:
    sed_transform(
        std::string const& search
      , std::string const& replace
        );

    sed_transform(
        std::string const& expression
        );

    std::string operator()(
        std::string const& input
        ) const;

    explicit operator bool() const HPX_NOEXCEPT
    {
        // avoid compiler warning about conversion to bool
        return command_.get() ? true : false;
    }

    bool operator!() const
    {
        return !command_.get();
    }
};

}}

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

#endif // HPX_EC1602ED_CCC2_471C_BC28_1DBB98902F40

