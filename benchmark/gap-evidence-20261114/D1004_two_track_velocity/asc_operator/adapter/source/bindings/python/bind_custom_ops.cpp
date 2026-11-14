#include <asnumpy/custom_ops/diff.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {
void bind_custom_ops(py::module_& module) {
    module.def("diff", &NumpyDiff, py::arg("a").noconvert(), py::arg("axis") = 1,
               "NumPy-compatible asnumpy.diff on the NPU.");
}
} // namespace asnumpy

void bind_custom_ops(py::module_& module) {
    asnumpy::bind_custom_ops(module);
}
