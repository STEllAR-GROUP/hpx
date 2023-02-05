# Remove during final commit
set(CMAKE_CXX_COMPILER "/home/hhn/makes/gcc-modules-install/usr/local/bin/g++")
message(STATUS "${CMAKE_CXX_COMPILER}")
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")

if(HPX_CXX_MODULES)
    MESSAGE(WARNING "C++ modules is experimental, make sure the requirements are met")
    MESSAGE(WARNING "Patches for GCC to implementing module dependency scanning 
                    haven't landed yet. Make sure you have the right build of GCC.")
    MESSAGE(WARNING "GCC Build : https://github.com/mathstuf/gcc/tree/p1689r5")
    MESSAGE(WARNING "GNU-Make does not support fully support modules yet,
                    use Ninja(>1.12) as your generator")
    MESSAGE(WARNING "Set the CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API to the correct value")
    MESSAGE(WARNING "set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API \"2182bf5c-ef0d-489a-91da-49dbc3090d2a\")")
endif()



if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Search for Clang implementation of Modules   

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
    string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
      "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E -x c++ <SOURCE>"
      " -MT <DYNDEP_FILE> -MD -MF <DEP_FILE>"
      " -fmodules-ts -fdep-file=<DYNDEP_FILE> -fdep-output=<OBJECT> -fdep-format=trtbd"
      " -o <PREPROCESSED_SOURCE>")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "gcc")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "-fmodules-ts -fmodule-mapper=<MODULE_MAP_FILE> -fdep-format=trtbd -x c++")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # Search for Intel C++ implementation of Modules   

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
    string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
      "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> <SOURCE> -nologo -TP"
      " -showIncludes"
      " -scanDependencies <DYNDEP_FILE>"
      " -Fo<OBJECT>")
    set(CMAKE_EXPERIMENTAL_CXX_SCANDEP_DEPFILE_FORMAT "msvc")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "msvc")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "@<MODULE_MAP_FILE>")
endif()

function (generate_module module_name module_componenets)
# all module components are public
    message(STATUS ${module_componenets})
    add_library(${module_name})
    target_link_libraries(${module_name})
    set_target_properties(${module_name} PROPERTIES LINKER_LANGUAGE CXX)
    target_sources(${module_name}
        PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES
            FILES ${module_componenets}
    )       
endfunction()
