# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
# Copyright (C) 2007 Douglas Gregor <doug.gregor@gmail.com>                   #
# Copyright (C) 2007 Troy Straszheim                                          #
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# This utility macro determines whether a particular string value
# occurs within a list of strings:
#
#  list_contains(result string_to_find arg1 arg2 arg3 ... argn)
#
# This macro sets the variable named by result equal to TRUE if
# string_to_find is found anywhere in the following arguments.
macro(list_contains var value)
    set(${var})
    foreach(value2 ${ARGN})
        if(${value} STREQUAL ${value2})
            set(${var} TRUE)
        endif(${value} STREQUAL ${value2})
    endforeach(value2)
endmacro(list_contains)

###############################################################################
# Copyright (C) 2007 Douglas Gregor <doug.gregor@gmail.com>                   #
# Copyright (C) 2007 Troy Straszheim                                          #
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# The PARSE_ARGUMENTS macro will take the arguments of another macro and
# define several variables. The first argument to PARSE_ARGUMENTS is a
# prefix to put on all variables it creates. The second argument is a
# list of names, and the third argument is a list of options. Both of
# these lists should be quoted. The rest of PARSE_ARGUMENTS are
# arguments from another macro to be parsed.
#
#     PARSE_ARGUMENTS(prefix arg_names options arg1 arg2...)
#
# For each item in options, PARSE_ARGUMENTS will create a variable with
# that name, prefixed with prefix_. So, for example, if prefix is
# MY_MACRO and options is OPTION1;OPTION2, then PARSE_ARGUMENTS will
# create the variables MY_MACRO_OPTION1 and MY_MACRO_OPTION2. These
# variables will be set to true if the option exists in the command line
# or false otherwise.
#
# For each item in arg_names, PARSE_ARGUMENTS will create a variable
# with that name, prefixed with prefix_. Each variable will be filled
# with the arguments that occur after the given arg_name is encountered
# up to the next arg_name or the end of the arguments. All options are
# removed from these lists. PARSE_ARGUMENTS also creates a
# prefix_DEFAULT_ARGS variable containing the list of all arguments up
# to the first arg_name encountered.
macro(PARSE_ARGUMENTS prefix arg_names option_names)
    set(DEFAULT_ARGS)
    foreach(arg_name ${arg_names})
        set(${prefix}_${arg_name})
    endforeach(arg_name)
    foreach(option ${option_names})
        set(${prefix}_${option} FALSE)
    endforeach(option)

    set(current_arg_name DEFAULT_ARGS)
    set(current_arg_list)
    foreach(arg ${ARGN})
        list_contains(is_arg_name ${arg} ${arg_names})
        if(is_arg_name)
            set(${prefix}_${current_arg_name} ${current_arg_list})
            set(current_arg_name ${arg})
            set(current_arg_list)
        else(is_arg_name)
            list_contains(is_option ${arg} ${option_names})
            if(is_option)
                set(${prefix}_${arg} TRUE)
            else(is_option)
                set(current_arg_list ${current_arg_list} ${arg})
            endif(is_option)
        endif(is_arg_name)
    endforeach(arg)
    set(${prefix}_${current_arg_name} ${current_arg_list})
endmacro(PARSE_ARGUMENTS)

###############################################################################
# This macro allows to build a HPX component
macro(ADD_HPX_COMPONENT name)
    # retrieve arguments
    parse_arguments(${name}
        "SOURCES;HEADERS;DEPENDENCIES;INI"
        "DEBUG"
        ${ARGN})

    if(${name}_DEBUG)
        message(STATUS ${name}_SOURCES ": " ${${name}_SOURCES})
        message(STATUS ${name}_HEADERS ": " ${${name}_HEADERS})
        message(STATUS ${name}_DEPENDENCIES ": " ${${name}_DEPENDENCIES})
        message(STATUS ${name}_INI ": " ${${name}_INI})
    endif()

    add_definitions(-DHPX_COMPONENT_NAME=${name})
    add_definitions(-DHPX_COMPONENT_EXPORTS)

    add_library (${name}_component SHARED 
        ${${name}_SOURCES} 
        ${${name}_HEADERS})

    # set properties of generated shared library
    set_target_properties(${name}_component PROPERTIES
        VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
        SOVERSION ${HPX_SOVERSION}
        CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
        OUTPUT_NAME ${component_LIBRARY_PREFIX}${name})

    target_link_libraries(${name}_component 
        ${${name}_DEPENDENCIES}
        ${hpx_LIBRARIES} ${Boost_LIBRARIES})

    # install binary
    install(TARGETS ${name}_component
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

    # install hpx configuration (ini) file
    if(${name}_INI)
        install(FILES ${${name}_INI} DESTINATION share/hpx/ini)
    endif()

endmacro(ADD_HPX_COMPONENT)

###############################################################################
# This macro allows to build a HPX executable
macro(ADD_HPX_EXECUTABLE name)
    # retrieve arguments
    parse_arguments(${name}
        "SOURCES;HEADERS;DEPENDENCIES"
        "DEBUG"
        ${ARGN})

    if(${name}_DEBUG)
        message(STATUS ${name}_SOURCES ": " ${${name}_SOURCES})
        message(STATUS ${name}_HEADERS ": " ${${name}_HEADERS})
        message(STATUS ${name}_DEPENDENCIES ": " ${${name}_DEPENDENCIES})
    endif()

    add_definitions(-DHPX_APPLICATION_EXPORTS)

    # add the executable build target
    add_executable(${name}_exe 
        ${${name}_SOURCES} 
        ${${name}_HEADERS})

    # avoid conflicts between source and binary target names
    set_target_properties(${name}_exe PROPERTIES
        DEBUG_OUTPUT_NAME ${name}${CMAKE_DEBUG_POSTFIX})
    set_target_properties(${name}_exe PROPERTIES
        RELEASE_OUTPUT_NAME ${name})
    set_target_properties(${name}_exe PROPERTIES
        RELWITHDEBINFO_OUTPUT_NAME ${name})
    set_target_properties(${name}_exe PROPERTIES
        MINSIZEREL_OUTPUT_NAME ${name})

    # linker instructions
    target_link_libraries(
        ${name}_exe                           # executable
        ${${name}_DEPENDENCIES}               # components it depends on
        ${hpx_LIBRARIES} ${Boost_LIBRARIES})  # libraries it depends on

    # installation instructions
    install(TARGETS ${name}_exe
        RUNTIME DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

endmacro(ADD_HPX_EXECUTABLE)
