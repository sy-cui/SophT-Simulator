from elastica import RigidBodyBase
import numpy as np
from sopht.simulator.immersed_body.immersed_body_forcing_grid import (
    ImmersedBodyForcingGrid,
)
from sopht.simulator.immersed_body.immersed_boundary_interactions.PenaltyBoundaryForcing import (
    PenaltyBoundaryForcing,
)
from typing import Type, Optional, Literal


class RigidBodyFlowInteraction(PenaltyBoundaryForcing):
    """Class for rigid body (from pyelastica) flow interaction."""

    def __init__(
        self,
        rigid_body: RigidBodyBase,
        eul_grid_forcing_field: np.ndarray,
        eul_grid_velocity_field: np.ndarray,
        virtual_boundary_stiffness_coeff: float,
        virtual_boundary_damping_coeff: float,
        dx: float,
        grid_dim: int,
        forcing_grid_cls: Type[ImmersedBodyForcingGrid],
        real_t: type = np.float64,
        eul_grid_coord_shift: Optional[float] = None,
        interp_kernel_width: int = 2,
        start_time: float = 0.0,
        interp_kernel_type: Literal["peskin", "cosine"] = "cosine",
        **forcing_grid_kwargs,
    ) -> None:
        """Class initialiser."""
        body_flow_forces = np.zeros((3, 1))
        body_flow_torques = np.zeros((3, 1))
        forcing_grid_kwargs["rigid_body"] = rigid_body

        # initialising super class
        super().__init__(
            virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
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
