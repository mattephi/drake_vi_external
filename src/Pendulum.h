//
// Created by vscode on 11/26/24.
//
#pragma once

#include <iostream>
#include <drake/common/drake_copyable.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/continuous_state.h>
#include <drake/systems/framework/leaf_system.h>

#include <drake/math/autodiff.h>


namespace drake_external_examples::particles {
  template <typename T>
  class PendulumSystem final : public drake::systems::LeafSystem<T> {
  public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PendulumSystem);
    explicit PendulumSystem(double length, double gravity, double damping)
      : length_(length), gravity_(gravity), damping_(damping), drake::systems::LeafSystem<T>(drake::systems::SystemTypeTag<PendulumSystem>{}) {
      this->DeclareContinuousState(2); // [theta, theta_dot]
      this->DeclareNumericParameter(drake::systems::BasicVector<T>(1)); // theta_{i - 1}
      // this->DeclarePerStepPublishEvent(&PendulumSystem::UpdateParametersWithCurrentState);
    }

    template <typename U>
    explicit PendulumSystem(const PendulumSystem<U> &other)
      : PendulumSystem<T>(other.length_, other.gravity_, other.damping_) {}

  protected:
    void DoCalcTimeDerivatives(
      const drake::systems::Context<T> &context,
      drake::systems::ContinuousState<T> *derivatives) const override {
      const auto &state = context.get_continuous_state_vector().CopyToVector();
      const T theta = state[0];
      const T theta_dot = state[1];
      const T theta_ddot = -(gravity_ / length_) * sin(theta) -
                                (damping_ / (length_ * length_)) * theta_dot;
      derivatives->SetFromVector((Eigen::Matrix<T, 2, 1>() << theta_dot, theta_ddot).finished());
      // derivatives->SetFromVector(Eigen::Vector4d<T>(theta_dot, theta_ddot));
    }

  public:
    T DoCalcKineticEnergy(const drake::systems::Context<T> &context) const override {
      // std::cout << "Tried to calculate kinetic energy" << std::endl;
      const auto &state = context.get_continuous_state_vector().CopyToVector();
      const T theta_dot = state[1];
      return 0.5 * length_ * length_ * theta_dot * theta_dot;
    }

    T DoCalcPotentialEnergy(const drake::systems::Context<T> &context) const override {
      // std::cout << "Tried to calculate potential energy" << std::endl;
      const auto &state = context.get_continuous_state_vector().CopyToVector();
      const T theta = state[0];
      return -length_ * gravity_ * cos(theta);
    }

    T DoCalcLagrangian(const drake::systems::Context<T> &context) const {
      return this->DoCalcKineticEnergy(context) - this->DoCalcPotentialEnergy(context);
    }

    T DoCalcDiscreteLagrangian(const drake::systems::Context<T> &context, const T &h) const {
      const T& q0 = context.get_numeric_parameter(0)[0];
      const T& q1 = context.get_continuous_state_vector().GetAtIndex(0);
      auto &mutable_context = const_cast<drake::systems::Context<T> &>(context);
      auto &mutable_state = mutable_context.get_mutable_continuous_state();
      // auto old_velocity = mutable_state.get_mutable_vector().GetAtIndex(1);
      // auto old_velocity_derivatives = old_velocity.derivatives();
      const auto& lagrangian = this->DoCalcLagrangian(mutable_context) * h;
      // restore context
      // mutable_state.get_mutable_vector().SetAtIndex(1, old_velocity);
      // mutable_state.get_mutable_vector().GetAtIndex(0).derivatives() = old_velocity_derivatives;
      return this->DoCalcKineticEnergy(mutable_context);
    }

  public:
    const double length_;
    const double gravity_;
    const double damping_;

  public:
    drake::systems::EventStatus UpdateParametersWithCurrentState(
    const drake::systems::Context<T> &context) const {
      // std::cout << "UpdateParametersWithCurrentState" << std::endl;
      const T theta = context.get_continuous_state_vector().GetAtIndex(0);
      auto &mutable_context = const_cast<drake::systems::Context<T> &>(context);
      auto &parameter = this->GetMutableNumericParameter(&mutable_context, 0);
      parameter.SetAtIndex(0, theta);
      return drake::systems::EventStatus::Succeeded();
    }
  };
}

