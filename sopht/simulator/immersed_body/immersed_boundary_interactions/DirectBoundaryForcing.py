from typing import Type, Optional, Literal, Tuple

import numpy as np
from numba import njit

from sopht.simulator.immersed_body.immersed_boundary_interactions.ImmersedBoundaryInteraction import (
    ImmersedBoundaryInteraction,
)
from sopht.simulator.immersed_body.immersed_body_forcing_grid import (
    ImmersedBodyForcingGrid,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_set_fixed_val_pyst_kernel_2d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_set_fixed_val_pyst_kernel_3d,
)


class DirectBoundaryForcing(ImmersedBoundaryInteraction):
    def __init__(
        self,
        grid_dim: int,
        dx: float,
        eul_grid_position_field: np.ndarray,
        eul_grid_velocity_field: np.ndarray,
        forcing_grid_cls: Type[ImmersedBodyForcingGrid],
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        flow_density: float = 1.0,
        real_t: Type = np.float64,
        num_threads: int = 1,
        eul_grid_coord_shift: Optional[float] = None,
        interp_kernel_width: int = 2,
        start_time: float = 0.0,
        interp_kernel_type: Literal["peskin", "cosine"] = "cosine",
        explicit_forcing: bool = True,
        **forcing_grid_kwargs,
    ):
        super().__init__(
            grid_dim=grid_dim,
            dx=dx,
            eul_grid_velocity_field=eul_grid_velocity_field,
            forcing_grid_cls=forcing_grid_cls,
            body_flow_forces=body_flow_forces,
            body_flow_torques=body_flow_torques,
            real_t=real_t,
            eul_grid_coord_shift=eul_grid_coord_shift,
            interp_kernel_width=interp_kernel_width,
            start_time=start_time,
            interp_kernel_type=interp_kernel_type,
            **forcing_grid_kwargs,
        )
        self.interp_kernel_width = interp_kernel_width
        self.real_t = real_t
        self.eul_grid_position_field = eul_grid_position_field.view()
        self.lag_grid_position_mismatch_field = np.zeros_like(
            self.lag_grid_flow_velocity_field,
            dtype=real_t,
        )
        self.lag_grid_velocity_mismatch_field = np.zeros_like(
            self.lag_grid_flow_velocity_field,
            dtype=real_t,
        )
        self.dt = 0.0
        self.flow_density = flow_density
        self.explicit_forcing = explicit_forcing

        self.lag_grid_velocity_moment_field = np.zeros_like(
            self.lag_grid_body_velocity_field,
            dtype=real_t,
        )

        # For conjugate gradient method
        self.lag_grid_residual_field = np.zeros_like(
            self.lag_grid_body_velocity_field,
            dtype=real_t,
        )

        self.eul_buffer_field = np.zeros_like(
            self.eul_grid_position_field,
            dtype=real_t,
        )

        self.lag_grid_search_direction_field = np.zeros_like(
            self.lag_grid_body_velocity_field,
            dtype=real_t,
        )

        self.tol = real_t(1e3) * np.finfo(real_t).eps
        self.spd_inner_product = np.empty((self.grid_dim,), dtype=real_t)
        self.alpha = np.empty((self.grid_dim,), dtype=real_t)
        self.beta = np.empty((self.grid_dim,), dtype=real_t)
        self.num_iters = 0

        self.sum_axes: Tuple[int, ...]
        if self.grid_dim == 2:
            self._set_field = gen_set_fixed_val_pyst_kernel_2d(
                real_t=self.real_t,
                num_threads=num_threads,
                field_type="vector",
            )
            self.sum_axes = (1, 2)
        else:
            self._set_field = gen_set_fixed_val_pyst_kernel_3d(
                real_t=self.real_t,
                num_threads=num_threads,
                field_type="vector",
            )
            self.sum_axes = (1, 2, 3)

    def compute_lag_grid_forcing_field(self) -> None:
        """Compute the forcing field on Lagrangian grid

        We do nothing here as the forcing is propagated to the body
        from the previous time step.
        """

    def __call__(self) -> None:
        """Compute the Lagrangian forces and transfer to them to Eulerian grid

        This method should be invoked exactly once every flow step.
        """

        # 1. Update the Lagrangian body position and velocity
        self.forcing_grid.compute_lag_grid_position_field()
        self.forcing_grid.compute_lag_grid_velocity_field()

        # 2. Transfer the Eulerian flow velocity onto the Lagrangian grid
        self.transfer_flow_velocity_to_lag_grid()

        # 3. Compute velocity mismatch between flow and body
        self._compute_lag_grid_velocity_mismatch_field(
            lag_grid_velocity_mismatch_field=self.lag_grid_velocity_mismatch_field,
            lag_grid_flow_velocity_field=self.lag_grid_flow_velocity_field,
            lag_grid_body_velocity_field=self.lag_grid_body_velocity_field,
        )

        # 4. Solve for velocity moment
        if self.explicit_forcing:
            self._solve_for_velocity_moment_via_explicit_forcing()
        else:
            self._solve_for_velocity_moment_via_conjugate_gradient()

        # 5. Compute velocity correction on Eulerian grid
        self.reset_field(self.eul_buffer_field)
        self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            eul_grid_field=self.eul_buffer_field,
            lag_grid_field=self.lag_grid_velocity_moment_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )

        # 6. Compute Lagrangian grid forcing
        self.lag_grid_forcing_field[...] = (
            self.flow_density * self.lag_grid_velocity_moment_field / self.dt
        )

    def _solve_for_velocity_moment_via_explicit_forcing(self) -> None:
        self.lag_grid_velocity_moment_field[...] = self.real_t(1.0)
        self.reset_field(self.eul_buffer_field)
        self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            eul_grid_field=self.eul_buffer_field,
            lag_grid_field=self.lag_grid_velocity_moment_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )
        self.eul_lag_grid_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
            lag_grid_field=self.lag_grid_velocity_moment_field,
            eul_grid_field=self.eul_buffer_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )
        self.lag_grid_velocity_moment_field[
            ...
        ] = self.lag_grid_velocity_mismatch_field / (
            self.lag_grid_velocity_moment_field
        )

    def _solve_for_velocity_moment_via_conjugate_gradient(self) -> None:
        """Solve for DD^T (dU dX) = U_b - Du*"""

        self.num_iters = 0

        # Initial guess
        self.lag_grid_velocity_moment_field[...] = (
            self.lag_grid_velocity_mismatch_field
            * self.max_lag_grid_dx**self.grid_dim
        )

        # p0 = r0 = b - A x0
        self._conjugate_gradient_compute_residual()
        self.lag_grid_search_direction_field[...] = self.lag_grid_residual_field.copy()

        while np.amax(np.abs(self.lag_grid_residual_field)) > self.tol:
            self.num_iters += 1
            # D^T p
            self.reset_field(self.eul_buffer_field)
            self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
                eul_grid_field=self.eul_buffer_field,
                lag_grid_field=self.lag_grid_search_direction_field,
                interp_weights=self.interp_weights,
                nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
            )
            # p^T A p = norm(D^T p)^2
            self.spd_inner_product[...] = np.sum(
                self.eul_buffer_field**2, axis=self.sum_axes
            )
            self.spd_inner_product[...] *= self.dx**self.grid_dim
            # r^T r
            self.beta[...] = np.sum(self.lag_grid_residual_field**2, axis=1)
            # alpha = r^T r / p^T A p
            self.alpha[...] = np.nan_to_num(self.beta / self.spd_inner_product)
            # x = x + alpha * p
            self.lag_grid_velocity_moment_field[...] += (
                self.alpha.reshape((-1, 1)) * self.lag_grid_search_direction_field
            )
            # r = b - Ax
            self._conjugate_gradient_compute_residual()
            # beta = (r^T r)_{k + 1} / (r^T r)_{k}
            self.beta[...] = np.nan_to_num(
                np.sum(self.lag_grid_residual_field**2, axis=1) / self.beta
            )
            # p = r + beta * p
            self.lag_grid_search_direction_field[...] = (
                self.lag_grid_residual_field
                + self.beta.reshape((-1, 1)) * self.lag_grid_search_direction_field
            )

    def _conjugate_gradient_compute_residual(self) -> None:
        """Helper function that computes the residual for CG solve.

        Residual is evaluated as RHS - LHS, where the RHS must be pre-computed and
        stored in 'self.lag_grid_velocity_mismatch_field'. The LHS is computed by
        spreading the velocity moment onto a buffer Eulerian field (reset before
        every use) and then interpolate back onto the Lagrangian grid.
        """
        self.reset_field(self.eul_buffer_field)
        self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            eul_grid_field=self.eul_buffer_field,
            lag_grid_field=self.lag_grid_velocity_moment_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )
        self.eul_lag_grid_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
            lag_grid_field=self.lag_grid_residual_field,
            eul_grid_field=self.eul_buffer_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )
        self.lag_grid_residual_field[...] = (
            self.lag_grid_velocity_mismatch_field - self.lag_grid_residual_field
        )

    def reset_field(self, field) -> None:
        self._set_field(
            vector_field=field, fixed_vals=[self.real_t(0.0)] * self.grid_dim
        )

    def set_dt(self, dt: float) -> None:
        """Set class immersed boundary time step.

        Parameters
        ----------
        dt: float
            Immersed boundary time step. This is tentatively set as the flow time step.
            TODO: See if using flow time step is correct
        """
        self.dt = dt

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _compute_lag_grid_velocity_mismatch_field(
        lag_grid_velocity_mismatch_field,
        lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field,
    ) -> None:
        """Compute the Lagrangian velocity mismatch field.

        Numba-supported subtraction of the flow velocity field from
        the body velocity field on the Lagrangian.

        Parameters
        ----------
        lag_grid_velocity_mismatch_field: ndarray
            Lagrangian grid velocity mismatch field to be populated.
        lag_grid_flow_velocity_field: ndarray
            Lagrangian grid flow velocity field.
        lag_grid_body_velocity_field: ndarray
            Lagrangian grid body velocity field.
        """
        lag_grid_velocity_mismatch_field[...] = (
            lag_grid_body_velocity_field - lag_grid_flow_velocity_field
        )

    def get_grid_deviation_error_l2_norm(self) -> float:
        """Compute L2 norm of Lagrangian position mismatch field between flow
        and body

        Returns
        -------
        float:
            L2 norm of Lagrangian position mismatch field
        """
        return np.linalg.norm(self.lag_grid_position_mismatch_field) / np.sqrt(
            self.num_lag_nodes
        )
