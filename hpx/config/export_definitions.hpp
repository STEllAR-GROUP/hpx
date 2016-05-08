//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXPORT_DEFINITIONS_SEPTEMBER_25_2008_0214PM)
#define HPX_EXPORT_DEFINITIONS_SEPTEMBER_25_2008_0214PM

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
# define HPX_SYMBOL_EXPORT      __declspec(dllexport)
# define HPX_SYMBOL_IMPORT      __declspec(dllimport)
# define HPX_SYMBOL_INTERNAL    /* empty */
# define HPX_APISYMBOL_EXPORT   __declspec(dllexport)
# define HPX_APISYMBOL_IMPORT   __declspec(dllimport)
#elif defined(__CUDACC__)
# define HPX_SYMBOL_EXPORT      /* empty */
# define HPX_SYMBOL_IMPORT      /* empty */
# define HPX_SYMBOL_INTERNAL    /* empty */
# define HPX_APISYMBOL_EXPORT   /* empty */
# define HPX_APISYMBOL_IMPORT   /* empty */
#elif defined(HPX_HAVE_ELF_HIDDEN_VISIBILITY)
# define HPX_SYMBOL_EXPORT      __attribute__((visibility("default")))
# define HPX_SYMBOL_IMPORT      __attribute__((visibility("default")))
# define HPX_SYMBOL_INTERNAL    __attribute__((visibility("hidden")))
# define HPX_APISYMBOL_EXPORT   __attribute__((visibility("default")))
# define HPX_APISYMBOL_IMPORT   __attribute__((visibility("default")))
#endif

// make sure we have reasonable defaults
#if !defined(HPX_SYMBOL_EXPORT)
# define HPX_SYMBOL_EXPORT      /* empty */
#endif
#if !defined(HPX_SYMBOL_IMPORT)
# define HPX_SYMBOL_IMPORT      /* empty */
#endif
#if !defined(HPX_SYMBOL_INTERNAL)
# define HPX_SYMBOL_INTERNAL    /* empty */
#endif
#if !defined(HPX_APISYMBOL_EXPORT)
# define HPX_APISYMBOL_EXPORT   /* empty */
#endif
#if !defined(HPX_APISYMBOL_IMPORT)
# define HPX_APISYMBOL_IMPORT   /* empty */
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros used by the runtime module
#if defined(HPX_EXPORTS)
# define  HPX_EXPORT             HPX_SYMBOL_EXPORT
# define  HPX_EXCEPTION_EXPORT   HPX_SYMBOL_EXPORT
# define  HPX_API_EXPORT         HPX_APISYMBOL_EXPORT
#else
# define  HPX_EXPORT             HPX_SYMBOL_IMPORT
# define  HPX_EXCEPTION_EXPORT   HPX_SYMBOL_IMPORT
# define  HPX_API_EXPORT         HPX_APISYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros to be used for component modules
#if defined(HPX_COMPONENT_EXPORTS)
# define  HPX_COMPONENT_EXPORT   HPX_SYMBOL_EXPORT
#else
# define  HPX_COMPONENT_EXPORT   HPX_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// define the export/import helper macros to be used for component modules
#if defined(HPX_LIBRARY_EXPORTS)
# define  HPX_LIBRARY_EXPORT     HPX_SYMBOL_EXPORT
#else
# define  HPX_LIBRARY_EXPORT     HPX_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
#if defined(HPX_EXPORTS) || defined(HPX_COMPONENT_EXPORTS) || \
    defined(HPX_APPLICATION_EXPORTS) || defined(HPX_SERIALIZATION_EXPORTS) || \
    defined(HPX_LIBRARY_EXPORTS)
# define HPX_ALWAYS_EXPORT       HPX_SYMBOL_EXPORT
#else
# define HPX_ALWAYS_EXPORT       HPX_SYMBOL_IMPORT
#endif

#endif
