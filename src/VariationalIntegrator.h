//
// Created by vscode on 11/27/24.
//

#ifndef VARIATIONALINTEGRATOR_H
#define VARIATIONALINTEGRATOR_H
#include <iostream>
#include <drake/systems/analysis/explicit_euler_integrator.h>

#include "Pendulum.h"
#include <drake/math/autodiff.h>
// #include <drake/multibody/math/spatial_vector.h>

namespace drake_external_examples {
  namespace partices {
    using drake::systems::IntegratorBase;
    using drake::systems::System;
    using drake::systems::Context;
    using drake::systems::ContinuousState;
    using drake::systems::VectorBase;
    using drake::systems::ExplicitEulerIntegrator;

    template <class T>
class VariationalIntegrator final : public IntegratorBase<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(VariationalIntegrator);

  ~VariationalIntegrator() override = default;

  /**
   * Constructs a fixed-step integrator for a given system using the given
   * context for initial conditions.
   * @param system A reference to the system to be simulated
   * @param max_step_size The maximum (fixed) step size; the integrator will
   *                      not take larger step sizes than this.
   * @param context Pointer to the context (nullptr is ok, but the caller
   *                must set a non-null context before Initialize()-ing the
   *                integrator).
   * @sa Initialize()
   */
  VariationalIntegrator(const particles::PendulumSystem<T>& system, const T& max_step_size,
                          Context<T>* context = nullptr)
      : IntegratorBase<T>(system, context) {
    IntegratorBase<T>::set_maximum_step_size(max_step_size);
    this->autodiff_clone = this->get_system().ToAutoDiffXd();
    this->autodiff_context = this->autodiff_clone->CreateDefaultContext();
  }
      std::unique_ptr<drake::systems::System<drake::AutoDiffXd>> autodiff_clone;
      std::unique_ptr<drake::systems::Context<drake::AutoDiffXd>> autodiff_context;

  /**
   * Explicit Euler integrator does not support error estimation.
   */
  bool supports_error_estimation() const override { return false; }

  /// Integrator does not provide an error estimate.
  int get_error_estimate_order() const override { return 0; }

 private:
  bool DoStep(const T& h) override;
};

/**
 * Integrates the system forward in time by h, starting at the current time t₀.
 * This value of h is determined by IntegratorBase::Step().
 */
template <class T>
bool VariationalIntegrator<T>::DoStep(const T& h) {
  Context<T>& context = *this->get_mutable_context();

  const auto& system = static_cast<const particles::PendulumSystem<T>&>(this->get_system());
  const auto& diff_system = static_cast<const particles::PendulumSystem<drake::AutoDiffXd>&>(*(this->autodiff_clone));
  const auto& state = context.get_continuous_state_vector();

  const T& q0 = context.get_numeric_parameter(0)[0];
  const T& v0 = context.get_continuous_state().get_vector()[1];
  const T& q1 = context.get_continuous_state().get_vector()[0];
  const T& v_estimate = (q1 - q0) / h;

  std::cout << "Kinetic energy is: " << system.CalcKineticEnergy(context) << std::endl;
  this->autodiff_context->SetTimeStateAndParametersFrom(context);
  auto &autodiff_xc = this->autodiff_context->get_mutable_continuous_state_vector();
  auto &autodiff_q0 = this->autodiff_context->get_mutable_numeric_parameter(0)[0];
  autodiff_q0.derivatives() = Eigen::MatrixXd::Identity(1, 1).col(0);
  std::cout << "autodiff_xc: " << autodiff_xc << std::endl;
  auto derivatives = Eigen::Vector2d(-1.0 / h, 1.0 / h);
  autodiff_xc[0].derivatives() = Eigen::MatrixXd::Identity(2, 2).col(0);
  autodiff_xc[1] = (q1 - q0) / h;
  autodiff_xc[1].derivatives() = derivatives;
  // autodiff_xc[1] = (q1 - q0) / h;
  auto autodiff_energy = diff_system.CalcKineticEnergy(*this->autodiff_context);
  std::cout << "Autodiff derivative is: " << autodiff_energy.derivatives() << std::endl;
  // auto diff_xc = drake::Vector<Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1>>, 3>();
  // diff_xc[0] = q0;
  // diff_xc[1] = q1;
  // diff_xc[2] = h;
  // std::cout << "diff_xc: " << diff_xc << std::endl;
  // diff_xc.derivatives()
  // diff_xc[0].derivatives() = Eigen::MatrixXd::Identity(1, 1).col(0);
  // diff_xc[1].derivatives() = Eigen::MatrixXd::Identity(1, 1).col(0);

  // this->autodiff_context->get_mutable_numeric_parameter(0)[0].derivatives() = Eigen::MatrixXd::Identity(1, 1).col(0);
  // std::cout << this->autodiff_context->get_mutable_numeric_parameter(0)[0].derivatives() << std::endl;
  // this->autodiff_context->get_mutable_continuous_state().get_mutable_vector()[0].derivatives() = Eigen::MatrixXd::Identity(1, 1).col(0);
  // this->autodiff_context->get_mutable_continuous_state().get_mutable_vector()[1].derivatives() = Eigen::MatrixXd::Identity(1, 1).col(0) / h;
  // std::cout << "autodiff_derivatives " << this->autodiff_context->get_continuous_state().get_vector()[1].derivatives() << std::endl;

  // std::cout << "ke, pe: " << system.CalcKineticEnergy(context) << " " << system.CalcPotentialEnergy(context) << std::endl;
  // std::cout << "lagrangian: " << system.DoCalcLagrangian(context) << std::endl;
  // std::cout << "discrete_lagrangian: " << system.DoCalcDiscreteLagrangian(context, h) << std::endl;
  std::cout << "Ld diff: " << diff_system.DoCalcLagrangian(*this->autodiff_context).derivatives() << std::endl;
  std::cout << "q0, q1: " << q0 << " " << q1 << std::endl;
  std::cout << "v0, v_estimate: " << v0 << " " << v_estimate << std::endl;

  // CAUTION: This is performance-sensitive inner loop code that uses dangerous
  // long-lived references into state and cache to avoid unnecessary copying and
  // cache invalidation. Be careful not to insert calls to methods that could
  // invalidate any of these references before they are used.
  // std::cout << "Called update\n";
  system.UpdateParametersWithCurrentState(context);
  // Evaluate derivative xcdot₀ ← xcdot(t₀, x(t₀), u(t₀)).
  const ContinuousState<T>& xc_deriv = this->EvalTimeDerivatives(context);
  const VectorBase<T>& xcdot0 = xc_deriv.get_vector();

  // Cache: xcdot0 references the live derivative cache value, currently
  // up to date but about to be marked out of date. We do not want to make
  // an unnecessary copy of this data.

  // Update continuous state and time. This call marks t- and xc-dependent
  // cache entries out of date, including xcdot0.

  VectorBase<T>& xc = context.SetTimeAndGetMutableContinuousStateVector(
      context.get_time() + h);  // t ← t₀ + h

  // Cache: xcdot0 still references the derivative cache value, which is
  // unchanged, although it is marked out of date.

  xc.PlusEqScaled(h, xcdot0);   // xc(t₀ + h) ← xc(t₀) + h * xcdot₀

  // This integrator always succeeds at taking the step.
  return true;
}
  } // partices
} // drake_external_examples

#endif //VARIATIONALINTEGRATOR_H
