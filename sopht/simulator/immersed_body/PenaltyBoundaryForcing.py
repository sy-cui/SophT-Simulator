import numpy as np
from numba import njit
from typing import Type, Optional, Literal

from sopht.simulator.immersed_body.immersed_body_forcing_grid import (
    ImmersedBodyForcingGrid,
)
from sopht.simulator.immersed_body.ImmersedBoundaryInteraction import (
    ImmersedBoundaryInteraction,
)


class PenaltyBoundaryForcing(ImmersedBoundaryInteraction):
    def __init__(
        self,
        virtual_boundary_stiffness_coeff: float,
        virtual_boundary_damping_coeff: float,
        grid_dim: int,
        dx: float,
        eul_grid_forcing_field: np.ndarray,
        eul_grid_velocity_field: np.ndarray,
        forcing_grid_cls: Type[ImmersedBodyForcingGrid],
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        real_t: Type = np.float64,
        eul_grid_coord_shift: Optional[float] = None,
        interp_kernel_width: int = 2,
        start_time: float = 0.0,
        interp_kernel_type: Literal["peskin", "cosine"] = "cosine",
        **forcing_grid_kwargs,
    ) -> None:
        """Base class for immersed boundary (IB) interaction.

        Parameters
        ----------
        virtual_boundary_stiffness_coeff: float
            Virtual boundary stiffness coefficient (alpha).
        virtual_boundary_damping_coeff: float
            Virtual boundary damping coefficient (beta).
        grid_dim: int
            Grid dimension (2 or 3).
        dx: float
            Uniform flow grid spacing for all axes.
            This should be generated during initialization of the flow domain.
        eul_grid_forcing_field: ndarray
            Eulerian grid forcing field. This is usually initialized in the flow
            simulator, and a view to the array is delegated here for IB interaction.
        eul_grid_velocity_field: ndarray
            Eulerian grid forcing field. This is usually initialized in the flow
            simulator, and a view to the array is delegated here for IB interaction.
        forcing_grid_cls: subclasses of ImmersedBodyForcingGrid
            Forcing grid class that translates properties between forcing grid
            and body's native grid.
        body_flow_forces: ndarray
            Interaction forces upon the body in its native representation.
            The shape of this array is independent of the Lagrangian forcing field.
            e.g. For a Cosserat rod, the object has shape (3, n_elems + 1)
                 For a rigid body, the object has shape (3, 1)
        body_flow_torques: ndarray
            Interaction torque upon the body in its native representation.
            The shape of this array is independent of the Lagrangian forcing field.
            e.g. For a Cosserat rod, the object has shape (3, n_elems)
                 For a rigid body, the object has shape (3, 1)
        real_t: type
            Numerical precision used for grid data. Defaults to float64.
        eul_grid_coord_shift: float, optional
            Shift of the Eulerian grid coordinate from 0. This is usually
            dx / 2, which is automatically enforced if no value is provided.
        interp_kernel_width: int
            Width of the interpolation kernel. Defaults to 2.
        start_time: float
            Start time of the simulation. Defaults to 0.0.
        interp_kernel_type: str
            Type of interpolation kernel function. Options include cosine and
            Peskin kernels. Defaults to "cosine".

        Other Parameters
        ----------------
        **forcing_grid_kwargs: dict
            Keyword arguments to pass into the forcing grid class.
        """

        super().__init__(
            grid_dim=grid_dim,
            dx=dx,
            eul_grid_forcing_field=eul_grid_forcing_field,
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

        self.alpha = virtual_boundary_stiffness_coeff * self.max_lag_grid_dx ** (
            grid_dim - 1
        )
        self.beta = virtual_boundary_damping_coeff * self.max_lag_grid_dx ** (
            grid_dim - 1
        )

        self.lag_grid_position_mismatch_field = np.zeros_like(
            self.lag_grid_flow_velocity_field,
            dtype=real_t,
        )
        self.lag_grid_velocity_mismatch_field = np.zeros_like(
            self.lag_grid_flow_velocity_field,
            dtype=real_t,
        )

    def compute_lag_grid_forcing_field(self) -> None:
        """Compute the forcing field on Lagrangian grid"""

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

        # 4. Compute penalty force on Lagrangian grid
        self._compute_lag_grid_forcing_field(
            lag_grid_forcing_field=self.lag_grid_forcing_field,
            lag_grid_position_mismatch_field=self.lag_grid_position_mismatch_field,
            lag_grid_velocity_mismatch_field=self.lag_grid_velocity_mismatch_field,
            virtual_boundary_stiffness_coeff=self.alpha,
            virtual_boundary_damping_coeff=self.beta,
        )

    def __call__(self) -> None:
        """Compute the Lagrangian forces and transfer to them to Eulerian grid

        This method should be invoked exactly once every flow step.
        """

        # 1. Compute forcing on Lagrangian grid
        self.compute_lag_grid_forcing_field()

        # 2. Transfer Lagrangian grid forces to Eulerian grid
        self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            eul_grid_field=self.eul_grid_forcing_field,
            lag_grid_field=self.lag_grid_forcing_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _compute_lag_grid_velocity_mismatch_field(
        lag_grid_velocity_mismatch_field,
        lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field,
    ) -> None:
        """Compute the Lagrangian velocity mismatch field.

        Numba-supported subtraction of the body velocity field from
        the flow velocity field on the Lagrangian.

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
            lag_grid_flow_velocity_field - lag_grid_body_velocity_field
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _update_lag_grid_position_mismatch_field_via_euler_forward(
        lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field,
        dt,
    ):
        """Update the Lagrangian grid position mismatch field.

        Euler forward time stepping in used. This method is invoked
        within the time_step method.

        Parameters
        ----------
        lag_grid_position_mismatch_field: ndarray
            Lagrangian grid position mismatch field to be updated.
        lag_grid_velocity_mismatch_field: ndarray
            Lagrangian grid velocity mismatch field.
        dt: float
            Time step. This is usually the body time step, when applicable.
        """
        lag_grid_position_mismatch_field[...] = (
            lag_grid_position_mismatch_field + dt * lag_grid_velocity_mismatch_field
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _compute_lag_grid_forcing_field(
        lag_grid_forcing_field,
        lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
    ):
        """Compute penalty force on Lagrangian grid, defined via virtual boundary method.

        Refer to Goldstein 1993, JCP for details on the penalty force computation.
        We can use pystencils for this but seems like it will be O(N) work, and won't be
        the limiter at least for few rods.

        """
        lag_grid_forcing_field[...] = (
            virtual_boundary_stiffness_coeff * lag_grid_position_mismatch_field
            + virtual_boundary_damping_coeff * lag_grid_velocity_mismatch_field
        )

    def time_step(self, dt: float):
        """

        Parameters
        ----------
        dt: float
            Time step. This is usually the body time step when applicable.

        Returns
        -------
        float
            Current immersed boundary class time
        """
        self._update_lag_grid_position_mismatch_field_via_euler_forward(
            lag_grid_position_mismatch_field=self.lag_grid_position_mismatch_field,
            lag_grid_velocity_mismatch_field=self.lag_grid_velocity_mismatch_field,
            dt=dt,
        )
        self.time += dt

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
