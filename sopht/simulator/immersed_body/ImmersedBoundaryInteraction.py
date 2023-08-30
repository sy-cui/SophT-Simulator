"""Generic immersed boundary interaction class for flow-body interaction"""
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Type, Optional, Literal

from sopht.numeric.immersed_boundary_ops.EulerianLagrangianGridCommunicator2D import (
    EulerianLagrangianGridCommunicator2D,
)
from sopht.numeric.immersed_boundary_ops.EulerianLagrangianGridCommunicator3D import (
    EulerianLagrangianGridCommunicator3D,
)
from sopht.simulator.immersed_body.immersed_body_forcing_grid import (
    ImmersedBodyForcingGrid,
)


class ImmersedBoundaryInteraction(ABC):
    def __init__(
        self,
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

        assert grid_dim == 2 or grid_dim == 3, "Invalid grid dimension."
        assert (
            interp_kernel_type == "peskin" or interp_kernel_type == "cosine"
        ), "Invalid interpolation kernel type."
        self.grid_dim = grid_dim
        self.time = start_time
        self.dx = dx

        if eul_grid_coord_shift is None:
            eul_grid_coord_shift = real_t(dx / 2)

        # References to the flow and body properties
        self.eul_grid_forcing_field = eul_grid_forcing_field.view()
        self.eul_grid_velocity_field = eul_grid_velocity_field.view()
        self.body_flow_forces = body_flow_forces.view()
        self.body_flow_torques = body_flow_torques.view()

        # Forcing grid objet creation
        self.forcing_grid = forcing_grid_cls(
            grid_dim=grid_dim,
            **forcing_grid_kwargs,
        )
        self.lag_grid_body_position_field = self.forcing_grid.position_field.view()
        self.lag_grid_body_velocity_field = self.forcing_grid.velocity_field.view()
        self.num_lag_nodes = self.forcing_grid.num_lag_nodes
        self.max_lag_grid_dx = self.forcing_grid.get_maximum_lagrangian_grid_spacing()

        # Check the relative grid resolution
        self._check_eul_lag_grid_relative_resolution()

        # Buffer creation
        self.nearest_eul_grid_index_to_lag_grid = np.empty(
            (grid_dim, self.num_lag_nodes),
            dtype=int,
        )

        eul_grid_support_of_lag_grid_shape = (
            (grid_dim,) + (2 * interp_kernel_width,) * grid_dim + (self.num_lag_nodes,)
        )
        self.local_eul_grid_support_of_lag_grid = np.empty(
            eul_grid_support_of_lag_grid_shape, dtype=real_t
        )

        interp_weights_shape = (2 * interp_kernel_width,) * grid_dim + (
            self.num_lag_nodes,
        )
        self.interp_weights = np.empty(interp_weights_shape, dtype=real_t)

        self.lag_grid_flow_velocity_field = np.zeros(
            (grid_dim, self.num_lag_nodes), dtype=real_t
        )
        self.lag_grid_forcing_field = np.zeros_like(
            self.lag_grid_flow_velocity_field,
            dtype=real_t,
        )

        self.eul_lag_grid_communicator: EulerianLagrangianGridCommunicator2D | EulerianLagrangianGridCommunicator3D

        if grid_dim == 2:
            self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicator2D(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                num_lag_nodes=self.num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                real_t=real_t,
                n_components=grid_dim,
                interp_kernel_type=interp_kernel_type,
            )

        else:
            self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicator3D(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                num_lag_nodes=self.num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                real_t=real_t,
                n_components=grid_dim,
                interp_kernel_type=interp_kernel_type,
            )

    @abstractmethod
    def compute_lag_grid_forcing_field(self) -> None:
        """Compute the forcing field on Lagrangian grid"""

    @abstractmethod
    def __call__(self) -> None:
        """Compute the Lagrangian forces and transfer to them to Eulerian grid

        This method should be invoked exactly once every flow step.
        """

    def compute_flow_forces_and_torques(self) -> None:
        """Compute flow forces and torques on the body from Lagrangian grid forces.

        For Cosserat rods, this method is called within the forcing class, which is
        invoked every rod time step.
        """
        self.compute_lag_grid_forcing_field()
        self.forcing_grid.transfer_forcing_from_grid_to_body(
            body_flow_forces=self.body_flow_forces,
            body_flow_torques=self.body_flow_torques,
            lag_grid_forcing_field=self.lag_grid_forcing_field,
        )

    def transfer_flow_velocity_to_lag_grid(self) -> None:
        """Interpolate the Eulerian grid flow velocity onto the Lagrangian grid

        This method should be invoked exactly once every flow step.
        """

        # 1. Find Eulerian grid local support of the Lagrangian grid
        self.eul_lag_grid_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
            local_eul_grid_support_of_lag_grid=self.local_eul_grid_support_of_lag_grid,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
            lag_positions=self.lag_grid_body_position_field,
        )

        # 2. Compute interpolation weights based on local Eulerian grid support
        self.eul_lag_grid_communicator.interpolation_weights_kernel(
            interp_weights=self.interp_weights,
            local_eul_grid_support_of_lag_grid=self.local_eul_grid_support_of_lag_grid,
        )

        # 3. Interpolate the Eulerian flow velocity on the Lagrangian grid
        self.eul_lag_grid_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
            lag_grid_field=self.lag_grid_flow_velocity_field,
            eul_grid_field=self.eul_grid_velocity_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )

    def _check_eul_lag_grid_relative_resolution(self) -> None:
        """Check the relative resolution of Eulerian and Lagrangian grids.

        Peskin (2002) said to avoid leaks, lag_h < h / 2 ???
        """
        log = logging.getLogger()
        grid_type = self.forcing_grid.__class__.__name__

        log.warning(
            "==========================================================\n"
            f"For {grid_type}:"
        )
        if (
            self.max_lag_grid_dx > 2 * self.dx
        ):  # 2 here since the support of delta function is 2 grid points
            log.warning(
                f"Eulerian grid spacing (dx): {self.dx}"
                f"\nMax Lagrangian grid spacing: {self.max_lag_grid_dx} > 2 * dx"
                "\nThe Lagrangian grid of the body is too coarse relative to"
                "\nthe Eulerian grid of the flow, which can lead to unexpected"
                "\nconvergence. Please make the Lagrangian grid finer."
            )
        elif (
            self.max_lag_grid_dx < 0.5 * self.dx
        ):  # reverse case of the above condition
            log.warning(
                "==========================================================\n"
                f"Eulerian grid spacing (dx): {self.dx}"
                f"\nMax Lagrangian grid spacing: {self.max_lag_grid_dx} < 0.5 * dx"
                "\nThe Lagrangian grid of the body is too fine relative to"
                "\nthe Eulerian grid of the flow, which corresponds to redundant"
                "\nforcing points. Please make the Lagrangian grid coarser."
            )
        else:
            log.warning(
                "Lagrangian grid is resolved almost the same as the Eulerian"
                "\ngrid of the flow."
            )
        log.warning("==========================================================")
