# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

add_library(asnumpy INTERFACE)
add_library(ascend_sdk INTERFACE)

set(ASNUMPY_CUSTOM_OPP_VENDOR "$ENV{ASNUMPY_CUSTOM_OPP_VENDOR}" CACHE PATH
    "Installed custom OPP vendor directory for generated operators.")
if(ASNUMPY_CUSTOM_OPP_VENDOR STREQUAL "")
    message(FATAL_ERROR "ASNUMPY_CUSTOM_OPP_VENDOR must point at the installed custom OPP vendor")
endif()

target_include_directories(asnumpy INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(asnumpy INTERFACE SPDLOG_FMT_EXTERNAL)

target_include_directories(ascend_sdk INTERFACE ${ASCEND_CANN_PATH}/include)
target_link_directories(ascend_sdk INTERFACE ${ASCEND_CANN_PATH}/lib64)
target_link_libraries(ascend_sdk INTERFACE ascendcl nnopbase opapi)

add_subdirectory(array)
add_subdirectory(cann)
add_subdirectory(dtypes)
add_subdirectory(linalg)
add_subdirectory(random)
add_subdirectory(math)
add_subdirectory(logic)
add_subdirectory(sorting)
add_subdirectory(statistics)
add_subdirectory(nn)
add_subdirectory(utils)
add_subdirectory(custom_ops)

target_link_libraries(asnumpy INTERFACE array cann dtypes linalg random math logic sorting statistics nn utils custom_ops)
