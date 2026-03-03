#ifndef GPU_BLAS_HPP
#define GPU_BLAS_HPP
#include "Random.hpp"
#include <baseliner/Case.hpp>
#include <cstddef>
#include <memory>
#include <vector>
namespace GpuBlas {

  /*
  Sgemm //float
  DGemm // double
  CGemm // complex
  Zgemm // double complex
  64 // using 64 bit integer for matrix sizes -> HUGE matrices
  */
  template <typename T>
  struct TypeOperations {
    static constexpr size_t factor = 1;
  };

  constexpr size_t DEFAULT_M = 200;
  constexpr size_t DEFAULT_N = 200;
  constexpr size_t DEFAULT_K = 400;

  template <typename BackendT, typename TypeT>
  class IGemm : public Baseliner::ICase<BackendT> {
  public:
    auto name() -> std::string override {
      return "Gemm" + std::string(typeid(TypeT).name());
    };
    auto validate_case() -> bool override {
      return true;
    }

    void alloc_host() {
      m_k = DEFAULT_K * this->get_work_size();
      m_m = DEFAULT_M;
      m_n = DEFAULT_N;
      m_A.resize(m_m * m_k);
      m_B.resize(m_k * m_n);
      m_C.resize(m_m * m_n);
      Random::random_fill_vector(m_A, this->get_seed());
      Random::random_fill_vector(m_B, this->get_seed());
    }
    void free_host() {
      m_A.clear();
      m_B.clear();
      m_C.clear();
    }
    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return 2 * m_n * m_k * m_m * TypeOperations<TypeT>::factor;
    }
    auto number_of_bytes() -> std::optional<size_t> override {
      return (2ULL * m_m * m_n + m_k * m_m + m_k * m_n) * sizeof(TypeT);
    }
    void register_options() override {
      Baseliner::ICase<BackendT>::register_options();
      this->add_option("Gemm", "m", "The number of rows of matrix A and C", m_m);
      this->add_option("Gemm", "k", "The number of columns of matrix A and rows of matric B", m_k);
      this->add_option("Gemm", "n", "The number of columns of matrix B and C", m_n);
      this->add_option("Gemm", "alpha", "The alpha", m_alpha);
      this->add_option("Gemm", "beta", "The Beta", m_beta);
    };

  protected:
    // HOST
    std::vector<TypeT> m_A; // has dimension m*k
    size_t m_m = DEFAULT_M;
    size_t m_k = DEFAULT_K;
    size_t m_n = DEFAULT_N;
    std::vector<TypeT> m_B; // has dimension k*n
    std::vector<TypeT> m_C; // has dimension m*n

    // DEVICE
    TypeT *m_d_A; // has dimension m*k
    TypeT *m_d_B; // has dimension k*n
    TypeT *m_d_C; // has dimension m*n

    // Options
    TypeT m_alpha{1};
    TypeT m_beta{0.5};
  };

} // namespace GpuBlas
#endif // GPU_BLAS_HPP