#include <mcl/bls12_381.hpp>
#include <vector>
#include <cstring>

using namespace mcl::bn;

// Global variables to store field and group elements
std::vector<Fr> field_elements;
std::vector<G1> group_elements;

// Initialize BLS12-381
extern "C" void init_bls12_381() {
    initPairing(mcl::BLS12_381);
}

// Field element operations
extern "C" int64_t create_field_element(int64_t value) {
    Fr f;
    f.setByCSPRNG(); // Create random element for now
    field_elements.push_back(f);
    return field_elements.size() - 1;
}

extern "C" int64_t field_add(int64_t a_idx, int64_t b_idx) {
    if (a_idx >= field_elements.size() || b_idx >= field_elements.size()) {
        return -1; // Error
    }
    Fr result;
    Fr::add(result, field_elements[a_idx], field_elements[b_idx]);
    field_elements.push_back(result);
    return field_elements.size() - 1;
}

extern "C" int64_t field_mul(int64_t a_idx, int64_t b_idx) {
    if (a_idx >= field_elements.size() || b_idx >= field_elements.size()) {
        return -1; // Error
    }
    Fr result;
    Fr::mul(result, field_elements[a_idx], field_elements[b_idx]);
    field_elements.push_back(result);
    return field_elements.size() - 1;
}

extern "C" int64_t field_sub(int64_t a_idx, int64_t b_idx) {
    if (a_idx >= field_elements.size() || b_idx >= field_elements.size()) {
        return -1; // Error
    }
    Fr result;
    Fr::sub(result, field_elements[a_idx], field_elements[b_idx]);
    field_elements.push_back(result);
    return field_elements.size() - 1;
}

extern "C" int64_t field_inv(int64_t a_idx) {
    if (a_idx >= field_elements.size()) {
        return -1; // Error
    }
    Fr result;
    Fr::inv(result, field_elements[a_idx]);
    field_elements.push_back(result);
    return field_elements.size() - 1;
}

extern "C" int64_t field_random() {
    Fr r;
    r.setByCSPRNG();
    field_elements.push_back(r);
    return field_elements.size() - 1;
}

// Group element operations
extern "C" int64_t create_group_element() {
    // Create a random group element by multiplying generator by random scalar
    Fr scalar;
    scalar.setByCSPRNG();
    G1 g = mcl::bn::getG1basePoint() * scalar;
    group_elements.push_back(g);
    return group_elements.size() - 1;
}

extern "C" int64_t group_scalar_mul(int64_t scalar_idx, int64_t point_idx) {
    if (scalar_idx >= field_elements.size() || point_idx >= group_elements.size()) {
        return -1; // Error
    }
    G1 result;
    G1::mul(result, group_elements[point_idx], field_elements[scalar_idx]);
    group_elements.push_back(result);
    return group_elements.size() - 1;
}

extern "C" int64_t group_add(int64_t a_idx, int64_t b_idx) {
    if (a_idx >= group_elements.size() || b_idx >= group_elements.size()) {
        return -1; // Error
    }
    G1 result;
    G1::add(result, group_elements[a_idx], group_elements[b_idx]);
    group_elements.push_back(result);
    return group_elements.size() - 1;
}

// String conversion
extern "C" void field_to_string(int64_t idx, char* buffer, int buffer_size) {
    if (idx >= field_elements.size()) {
        strcpy(buffer, "ERROR");
        return;
    }
    std::string str = field_elements[idx].getStr(10);
    strncpy(buffer, str.c_str(), buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
}

extern "C" void group_to_string(int64_t idx, char* buffer, int buffer_size) {
    if (idx >= group_elements.size()) {
        strcpy(buffer, "ERROR");
        return;
    }
    std::string str = group_elements[idx].getStr(16);
    strncpy(buffer, str.c_str(), buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
}

// Get field order as string
extern "C" void get_field_order(char* buffer, int buffer_size) {
    // BLS12-381 field order
    const char* order_str = "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
    strncpy(buffer, order_str, buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
}

// Clear storage
extern "C" void clear_storage() {
    field_elements.clear();
    group_elements.clear();
}

// Get storage sizes
extern "C" int64_t get_field_elements_count() {
    return field_elements.size();
}

extern "C" int64_t get_group_elements_count() {
    return group_elements.size();
}








