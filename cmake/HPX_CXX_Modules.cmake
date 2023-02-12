function (generate_module module_name module_componenets)
# all module components are public
    # message(STATUS ${module_componenets})
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
    string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
      "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E -x c++ <SOURCE>"
      " -MT <DYNDEP_FILE> -MD -MF <DEP_FILE>"
      " -fmodules-ts -fdep-file=<DYNDEP_FILE> -fdep-output=<OBJECT> -fdep-format=trtbd"
      " -o <PREPROCESSED_SOURCE>")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "gcc")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "-fmodules-ts -fmodule-mapper=<MODULE_MAP_FILE> -fdep-format=trtbd -x c++")

    add_library(${module_name})
    target_link_libraries(${module_name})
    set_target_properties(${module_name} PROPERTIES LINKER_LANGUAGE CXX)
    target_sources(${module_name}
        PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES
            FILES ${module_componenets}
    )       
endfunction()
