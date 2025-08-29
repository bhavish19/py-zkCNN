#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <mcl/bls12_381.hpp>
#include <vector>
#include <string>

namespace py = pybind11;
using namespace mcl::bn;

// Wrapper class for BLS12-381 operations
class BLS12_381Wrapper {
public:
    BLS12_381Wrapper() {
        // Initialize BLS12-381
        initPairing(mcl::BLS12_381);
    }
    
    // Field element operations
    Fr field_add(const Fr& a, const Fr& b) const {
        return a + b;
    }
    
    Fr field_mul(const Fr& a, const Fr& b) const {
        return a * b;
    }
    
    Fr field_sub(const Fr& a, const Fr& b) const {
        return a - b;
    }
    
    Fr field_inv(const Fr& a) const {
        return a.inv();
    }
    
    Fr field_random() const {
        Fr r;
        r.setRand();
        return r;
    }
    
    // Group element operations
    G1 group_random() const {
        G1 g;
        g.setRand();
        return g;
    }
    
    G1 group_scalar_mul(const Fr& scalar, const G1& point) const {
        return scalar * point;
    }
    
    G1 group_add(const G1& a, const G1& b) const {
        return a + b;
    }
    
    // Polynomial commitment operations
    std::vector<G1> poly_commit(const std::vector<Fr>& coeffs, const std::vector<G1>& gens) const {
        if (coeffs.size() != gens.size()) {
            throw std::runtime_error("Coefficients and generators must have same size");
        }
        
        std::vector<G1> commitment;
        G1 comm = G1::zero();
        
        for (size_t i = 0; i < coeffs.size(); ++i) {
            comm += coeffs[i] * gens[i];
        }
        
        commitment.push_back(comm);
        return commitment;
    }
    
    // Convert between types
    Fr int_to_field(int64_t value) const {
        Fr f;
        f.set(value);
        return f;
    }
    
    std::string field_to_string(const Fr& f) const {
        return f.getStr();
    }
    
    std::string group_to_string(const G1& g) const {
        return g.getStr();
    }
    
    // Vector operations
    std::vector<Fr> create_field_vector(const std::vector<int64_t>& values) const {
        std::vector<Fr> result;
        for (auto val : values) {
            result.push_back(int_to_field(val));
        }
        return result;
    }
    
    std::vector<G1> create_group_vector(size_t size) const {
        std::vector<G1> result;
        for (size_t i = 0; i < size; ++i) {
            result.push_back(group_random());
        }
        return result;
    }
    
    // Get field order
    std::string get_field_order() const {
        Fr order;
        order.setOrder();
        return order.getStr();
    }
    
    // Get group order
    std::string get_group_order() const {
        G1 g;
        g.setOrder();
        return g.getStr();
    }
};

PYBIND11_MODULE(bls12_381_python, m) {
    m.doc() = "Python bindings for BLS12-381 operations"; // optional module docstring
    
    py::class_<BLS12_381Wrapper>(m, "BLS12_381Wrapper")
        .def(py::init<>())
        .def("field_add", &BLS12_381Wrapper::field_add)
        .def("field_mul", &BLS12_381Wrapper::field_mul)
        .def("field_sub", &BLS12_381Wrapper::field_sub)
        .def("field_inv", &BLS12_381Wrapper::field_inv)
        .def("field_random", &BLS12_381Wrapper::field_random)
        .def("group_random", &BLS12_381Wrapper::group_random)
        .def("group_scalar_mul", &BLS12_381Wrapper::group_scalar_mul)
        .def("group_add", &BLS12_381Wrapper::group_add)
        .def("poly_commit", &BLS12_381Wrapper::poly_commit)
        .def("int_to_field", &BLS12_381Wrapper::int_to_field)
        .def("field_to_string", &BLS12_381Wrapper::field_to_string)
        .def("group_to_string", &BLS12_381Wrapper::group_to_string)
        .def("create_field_vector", &BLS12_381Wrapper::create_field_vector)
        .def("create_group_vector", &BLS12_381Wrapper::create_group_vector)
        .def("get_field_order", &BLS12_381Wrapper::get_field_order)
        .def("get_group_order", &BLS12_381Wrapper::get_group_order);
}








