/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <algorithm>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

void bind_array(pybind11::module_& array);
void bind_cann(pybind11::module_& cann);
void bind_fft(pybind11::module_& fft);
void bind_linalg(pybind11::module_& linalg);
void bind_linalg_no_submodule(pybind11::module_& m);
void bind_math(pybind11::module_& math);
void bind_logic(pybind11::module_& logic);
void bind_random(pybind11::module_& random);
void bind_sorting(pybind11::module_& sorting);
void bind_testing(pybind11::module_& testing);
void bind_utils(pybind11::module_& utils);
void bind_custom_ops(pybind11::module_& module);
namespace asnumpy {
void bind_statistics(pybind11::module_& statistics);
void bind_nn(pybind11::module_& nn);
} // namespace asnumpy

// Global storage for CannError exception class (used by translator)
static pybind11::object g_cann_error_cls;

PYBIND11_MODULE(_core, module) {
    module.doc() = "*** AsNumpy Core ***";

    // Register custom CANN error exception (subclass of RuntimeError)
    g_cann_error_cls = pybind11::register_exception<std::runtime_error>(module, "CannError");

    // Register exception translators
    pybind11::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const std::bad_alloc& e) {
            PyErr_SetString(PyExc_MemoryError, e.what());
        } catch (const std::runtime_error& e) {
            // ACL error messages contain "error ="; classify as CannError
            std::string msg = e.what();
            if (msg.find("error =") != std::string::npos) {
                PyErr_SetObject(g_cann_error_cls.ptr(), pybind11::str(msg).ptr());
            } else {
                PyErr_SetString(PyExc_RuntimeError, msg.c_str());
            }
        }
    });

    auto array = module.def_submodule("array");
    auto cann = module.def_submodule("cann");
    auto fft = module.def_submodule("fft");
    auto linalg = module.def_submodule("linalg");
    auto math = module.def_submodule("math");
    auto logic = module.def_submodule("logic");
    auto random = module.def_submodule("random");
    auto sorting = module.def_submodule("sorting");
    auto statistics = module.def_submodule("statistics");
    auto nn = module.def_submodule("nn");
    auto testing = module.def_submodule("testing");
    // auto utils = module.def_submodule("utils");

    bind_array(array);
    bind_cann(cann);
    bind_fft(fft);
    bind_linalg(linalg);
    bind_linalg_no_submodule(module);
    bind_math(math);
    bind_logic(logic);
    bind_random(random);
    bind_sorting(sorting);
    asnumpy::bind_statistics(statistics);
    asnumpy::bind_nn(nn);
    bind_testing(testing);
    bind_utils(module);
    bind_custom_ops(module);
}
