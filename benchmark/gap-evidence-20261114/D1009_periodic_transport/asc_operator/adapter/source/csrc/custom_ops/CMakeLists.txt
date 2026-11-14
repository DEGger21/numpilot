add_library(custom_ops OBJECT roll.cpp)
target_include_directories(custom_ops PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ASNUMPY_CUSTOM_OPP_VENDOR}/op_api/include)
target_compile_definitions(custom_ops PUBLIC
    ASNUMPY_CUSTOM_OPP_LIBRARY=\"${ASNUMPY_CUSTOM_OPP_VENDOR}/op_api/lib/libcust_opapi.so\")
target_link_libraries(custom_ops PUBLIC fmt::fmt ascend_sdk pybind11::pybind11)
