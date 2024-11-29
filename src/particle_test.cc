// SPDX-License-Identifier: MIT-0

#include "particle.h"  // IWYU pragma: associated
#include "Pendulum.h"
#include "VariationalIntegrator.h"
#include <memory>

#include <gtest/gtest.h>

#include <drake/common/eigen_types.h>
#include <drake/systems/framework/input_port.h>
#include <drake/systems/framework/system_output.h>
#include <drake/systems/framework/system.h>
#include <drake/systems/framework/vector_base.h>
#include <drake/systems/analysis/explicit_euler_integrator.h>

#include <pinocchio/multibody/model.hpp>

namespace drake_external_examples {
    namespace particles {
        namespace {
            ///
            /// A test fixture class for Particle systems.
            ///
            /// @tparam T must be a valid Eigen ScalarType.
            ///
            template<typename T>
            class ParticleTest : public ::testing::Test {
            protected:
                /// Arrange a Particle as the Device Under Test.
                void SetUp() override {
                    this->dut_ = std::make_unique<Particle<T> >();
                    this->context_ = this->dut_->CreateDefaultContext();
                    this->output_ = this->dut_->AllocateOutput();
                    this->derivatives_ = this->dut_->AllocateTimeDerivatives();
                }

                /// System (aka Device Under Test) being tested.
                std::unique_ptr<drake::systems::System<T> > dut_;
                /// Context for the given @p dut_.
                std::unique_ptr<drake::systems::Context<T> > context_;
                /// Outputs of the given @p dut_.
                std::unique_ptr<drake::systems::SystemOutput<T> > output_;
                /// Derivatives of the given @p dut_.
                std::unique_ptr<drake::systems::ContinuousState<T> > derivatives_;
            };

            TYPED_TEST_SUITE_P(ParticleTest);

            /// Makes sure a Particle output is consistent with its
            /// state (position and velocity).
            TYPED_TEST_P(ParticleTest, OutputTest) {
                // Initialize state.
                drake::systems::VectorBase<TypeParam> &continuous_state_vector =
                        this->context_->get_mutable_continuous_state_vector();
                continuous_state_vector.SetAtIndex(
                    0, static_cast<TypeParam>(10.0)); // x0 = 10 m.
                continuous_state_vector.SetAtIndex(
                    1, static_cast<TypeParam>(1.0)); // x1 = 1 m/s.
                // Compute outputs.
                this->dut_->CalcOutput(*this->context_, this->output_.get());
                drake::systems::BasicVector<TypeParam> *output_vector =
                        this->output_->GetMutableVectorData(0);
                // Check results.
                EXPECT_EQ(output_vector->GetAtIndex(0),
                          static_cast<TypeParam>(10.0)); // y0 == x0
                EXPECT_EQ(output_vector->GetAtIndex(1),
                          static_cast<TypeParam>(1.0)); // y1 == x1
            }

            /// Makes sure a Particle system state derivatives are
            /// consistent with its state and input (velocity and acceleration).
            TYPED_TEST_P(ParticleTest, DerivativesTest) {
                // Set input.
                const drake::systems::InputPort<TypeParam> &input_port =
                        this->dut_->get_input_port(0);
                drake::VectorX<TypeParam> u0(input_port.size());
                u0 << 1.0; // m/s^2
                input_port.FixValue(this->context_.get(), u0);
                // Set state.
                drake::systems::VectorBase<TypeParam> &continuous_state_vector =
                        this->context_->get_mutable_continuous_state_vector();
                continuous_state_vector.SetAtIndex(
                    0, static_cast<TypeParam>(0.0)); // x0 = 0 m
                continuous_state_vector.SetAtIndex(
                    1, static_cast<TypeParam>(2.0)); // x1 = 2 m/s
                // Compute derivatives.
                this->dut_->CalcTimeDerivatives(*this->context_, this->derivatives_.get());
                const drake::systems::VectorBase<TypeParam> &derivatives_vector =
                        this->derivatives_->get_vector();
                // Check results.
                EXPECT_EQ(derivatives_vector.GetAtIndex(0),
                          static_cast<TypeParam>(2.0)); // x0dot == x1
                EXPECT_EQ(derivatives_vector.GetAtIndex(1),
                          static_cast<TypeParam>(1.0)); // x1dot == u0
            }

            REGISTER_TYPED_TEST_SUITE_P(ParticleTest, OutputTest, DerivativesTest);

            INSTANTIATE_TYPED_TEST_SUITE_P(WithDoubles, ParticleTest, double);

            ///
        /// A test fixture class for Particle systems.
        /// @tparam T must be a valid Eigen ScalarType.
        ///
            TEST(PendulumIntegrationTest, BackupState) {
                // Create the pendulum system.
                auto pendulum = std::make_unique<PendulumSystem<double>>(1.0, 9.81, 0.0);
                {
                    auto autodiff_pendulum = pendulum->ToAutoDiffXd();
                    std::cout << "Conversion success" << std::endl;
                }

                double q0 = 3.14; // Initial angle in radians.
                double v0 = 0.0; // Initial angular velocity.
                // Create a context for the system.
                auto context = pendulum->CreateDefaultContext();

                // Set initial state: [theta, theta_dot].
                auto &state = context->get_mutable_continuous_state_vector();
                state.SetAtIndex(0, q0); // Initial angle in radians.
                state.SetAtIndex(1, v0); // Initial angular velocity.

                // Create a Runge-Kutta 2 integrator.
                partices::VariationalIntegrator<double> integrator(*pendulum, 0.01, context.get());
                integrator.set_maximum_step_size(0.01);
                integrator.set_fixed_step_mode(true);
                integrator.Initialize();

                // Integrate the system for 1.0 seconds with fixed step size.
                const double total_time = 0.10;
                const double step_size = 0.01;
                double q_prev_manual = q0;
                // double prev_x1 = v0;
                for (double time = 0.0; time < total_time; time += step_size) {
                    integrator.IntegrateWithSingleFixedStepToTime(time + step_size);
                    // Check the state.
                    const auto &current_state = state.CopyToVector();
                    const auto &q_prev = context->get_numeric_parameter(0)[0];
                    // Check the state.
                    EXPECT_NEAR(q_prev, q_prev_manual, 1e-2); // Check angle (theta).
                    // EXPECT_NEAR(prev_x1, backup_x1, 1e-2); // Check angular velocity (theta_dot).
                    // std::cout << "-------------------\n";
                    // std::cout << "time: " << time << "\n";
                    // std::cout << "backup: " << backup_state[0] << " " << backup_state[1] << " " << backup_state[2] << " " << backup_state[3] << "\n";
                    // std::cout << "prev: " << prev_x0 << " " << prev_x1 << "\n";

                    // std::cout << backup_x0 << " " << prev_x0 << " " << backup_state[0] << "\n";
                    q_prev_manual = q_prev;
                }

                // Get the final state.
                const auto &final_state = state.CopyToVector();
                // EXPECT_NEAR(final_state[0], 0.0, 1e-2); // Check angle (theta).
                // EXPECT_NEAR(final_state[1], 0.0, 1e-2); // Check angular velocity (theta_dot).
            }

            TEST(PendulumIntegrationTest, IntegrateState) {
                // Create the pendulum system.
                PendulumSystem<double> pendulum(1.0, 9.81, 0.0);

                double q0 = 1.57; // Initial angle in radians.
                double v0 = 0.0; // Initial angular velocity.
                // Create a context for the system.
                auto context = pendulum.CreateDefaultContext();
                auto &parameter = context->get_mutable_numeric_parameter(0);
                parameter[0] = q0;

                // Set initial state: [theta, theta_dot].
                auto &state = context->get_mutable_continuous_state_vector();
                state.SetAtIndex(0, q0); // Initial angle in radians.
                state.SetAtIndex(1, v0); // Initial angular velocity.

                // Create a Runge-Kutta 2 integrator.
                partices::VariationalIntegrator<double> integrator(pendulum, 0.01, context.get());
                integrator.set_maximum_step_size(0.01);
                integrator.set_fixed_step_mode(true);
                integrator.Initialize();

                // Integrate the system for 1.0 seconds with fixed step size.
                const double total_time = 0.10;
                const double step_size = 0.01;
                double q_prev_manual = q0;
                // double prev_x1 = v0;
                for (double time = 0.0; time < total_time; time += step_size) {
                    // pendulum.UpdateParametersWithCurrentState(context);
                    integrator.IntegrateWithSingleFixedStepToTime(time + step_size);
                    // Check the state.
                    const auto &current_state = state.CopyToVector();
                    const auto &q_prev = context->get_numeric_parameter(0)[0];
                    // Check the state.
                    std::cout << "-------------------\n";
                    EXPECT_NEAR(q_prev, q_prev_manual, 1e-2); // Check angle (theta).
                    // EXPECT_NEAR(prev_x1, backup_x1, 1e-2); // Check angular velocity (theta_dot).
                    // std::cout << "logged prev: " << q_prev_manual << std::endl;
                    // std::cout << "system prev: " << q_prev << std::endl;
                    // std::cout << "current: " << current_state[0] << std::endl;
                    // std::cout << "time: " << time << "\n";
                    // std::cout << "backup: " << backup_state[0] << " " << backup_state[1] << " " << backup_state[2] << " " << backup_state[3] << "\n";
                    // std::cout << "prev: " << prev_x0 << " " << prev_x1 << "\n";

                    // std::cout << backup_x0 << " " << prev_x0 << " " << backup_state[0] << "\n";
                    q_prev_manual = current_state[0];
                }

                // Get the final state.
                const auto &final_state = state.CopyToVector();
                // EXPECT_NEAR(final_state[0], 0.0, 1e-2); // Check angle (theta).
                // EXPECT_NEAR(final_state[1], 0.0, 1e-2); // Check angular velocity (theta_dot).
            }
        } // namespace
    } // namespace particles
} // namespace drake_external_examples

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
