/*!
 * Copyright 2014-2021 by Contributors
 */
#ifndef XGBOOST_TREE_PARAM_ONEAPI_H_
#define XGBOOST_TREE_PARAM_ONEAPI_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "xgboost/parameter.h"
#include "xgboost/data.h"

namespace xgboost {
namespace tree {

/*! \brief Wrapper for necessary training parameters for regression tree to access on device */
struct TrainParamOneAPI {
  float min_child_weight;
  float reg_lambda;
  float reg_alpha;
  float max_delta_step;

  TrainParamOneAPI() {}

  TrainParamOneAPI(const TrainParam& param) {
    reg_lambda = param.reg_lambda;
    reg_alpha = param.reg_alpha;
    min_child_weight = param.min_child_weight;
    max_delta_step = param.max_delta_step;
  }
};

/*! \brief core statistics used for tree construction */
template<typename GradType>
struct GradStatsOneAPI {
  /*! \brief sum gradient statistics */
  GradType sum_grad { 0 };
  /*! \brief sum hessian statistics */
  GradType sum_hess { 0 };

 public:
  GradType GetGrad() const { return sum_grad; }
  GradType GetHess() const { return sum_hess; }

  friend std::ostream& operator<<(std::ostream& os, GradStatsOneAPI s) {
    os << s.GetGrad() << "/" << s.GetHess();
    return os;
  }

  GradStatsOneAPI() {
  }

  template <typename GpairT>
  explicit GradStatsOneAPI(const GpairT &sum)
      : sum_grad(sum.GetGrad()), sum_hess(sum.GetHess()) {}
  explicit GradStatsOneAPI(const GradType grad, const GradType hess)
      : sum_grad(grad), sum_hess(hess) {}
  /*!
   * \brief accumulate statistics
   * \param p the gradient pair
   */
  inline void Add(GradientPair p) { this->Add(p.GetGrad(), p.GetHess()); }

  /*! \brief add statistics to the data */
  inline void Add(const GradStatsOneAPI& b) {
    sum_grad += b.sum_grad;
    sum_hess += b.sum_hess;
  }
  /*! \brief same as add, reduce is used in All Reduce */
  inline static void Reduce(GradStatsOneAPI& a, const GradStatsOneAPI& b) { // NOLINT(*)
    a.Add(b);
  }
  /*! \brief set current value to a - b */
  inline void SetSubstract(const GradStatsOneAPI& a, const GradStatsOneAPI& b) {
    sum_grad = a.sum_grad - b.sum_grad;
    sum_hess = a.sum_hess - b.sum_hess;
  }
  /*! \return whether the statistics is not used yet */
  inline bool Empty() const { return sum_hess == 0.0; }
  /*! \brief add statistics to the data */
  inline void Add(GradType grad, GradType hess) {
    sum_grad += grad;
    sum_hess += hess;
  }
};

/*!
 * \brief OneAPI implementation of SplitEntryContainer for device compilation.
 *        Original structure cannot be used due to std::isinf usage, which is not supported
 */
template<typename GradientT>
struct SplitEntryContainerOneAPI {
  /*! \brief loss change after split this node */
  bst_float loss_chg {0.0f};
  /*! \brief split index */
  bst_feature_t sindex{0};
  bst_float split_value{0.0f};

  GradientT left_sum;
  GradientT right_sum;

  SplitEntryContainerOneAPI() = default;

  friend std::ostream& operator<<(std::ostream& os, SplitEntryContainerOneAPI const& s) {
    os << "loss_chg: " << s.loss_chg << ", "
       << "split index: " << s.SplitIndex() << ", "
       << "split value: " << s.split_value << ", "
       << "left_sum: " << s.left_sum << ", "
       << "right_sum: " << s.right_sum;
    return os;
  }
  /*!\return feature index to split on */
  bst_feature_t SplitIndex() const { return sindex & ((1U << 31) - 1U); }
  /*!\return whether missing value goes to left branch */
  bool DefaultLeft() const { return (sindex >> 31) != 0; }
  /*!
   * \brief decides whether we can replace current entry with the given statistics
   *
   *   This function gives better priority to lower index when loss_chg == new_loss_chg.
   *   Not the best way, but helps to give consistent result during multi-thread
   *   execution.
   *
   * \param new_loss_chg the loss reduction get through the split
   * \param split_index the feature index where the split is on
   */
  inline bool NeedReplace(bst_float new_loss_chg, unsigned split_index) const {
    if (sycl::isinf(new_loss_chg)) {  // in some cases new_loss_chg can be NaN or Inf,
                                      // for example when lambda = 0 & min_child_weight = 0
                                      // skip value in this case
      return false;
    } else if (this->SplitIndex() <= split_index) {
      return new_loss_chg > this->loss_chg;
    } else {
      return !(this->loss_chg > new_loss_chg);
    }
  }
  /*!
   * \brief update the split entry, replace it if e is better
   * \param e candidate split solution
   * \return whether the proposed split is better and can replace current split
   */
  inline bool Update(const SplitEntryContainerOneAPI &e) {
    if (this->NeedReplace(e.loss_chg, e.SplitIndex())) {
      this->loss_chg = e.loss_chg;
      this->sindex = e.sindex;
      this->split_value = e.split_value;
      this->left_sum = e.left_sum;
      this->right_sum = e.right_sum;
      return true;
    } else {
      return false;
    }
  }
  /*!
   * \brief update the split entry, replace it if e is better
   * \param new_loss_chg loss reduction of new candidate
   * \param split_index feature index to split on
   * \param new_split_value the split point
   * \param default_left whether the missing value goes to left
   * \return whether the proposed split is better and can replace current split
   */
  bool Update(bst_float new_loss_chg, unsigned split_index,
              bst_float new_split_value, bool default_left,
              const GradientT &left_sum,
              const GradientT &right_sum) {
    if (this->NeedReplace(new_loss_chg, split_index)) {
      this->loss_chg = new_loss_chg;
      if (default_left) {
        split_index |= (1U << 31);
      }
      this->sindex = split_index;
      this->split_value = new_split_value;
      this->left_sum = left_sum;
      this->right_sum = right_sum;
      return true;
    } else {
      return false;
    }
  }

  /*! \brief same as update, used by AllReduce*/
  inline static void Reduce(SplitEntryContainerOneAPI &dst,         // NOLINT(*)
                            const SplitEntryContainerOneAPI &src) { // NOLINT(*)
    dst.Update(src);
  }
};

template<typename GradType>
using SplitEntryOneAPI = SplitEntryContainerOneAPI<GradStatsOneAPI<GradType>>;

}
}
#endif  // XGBOOST_TREE_PARAM_H_
