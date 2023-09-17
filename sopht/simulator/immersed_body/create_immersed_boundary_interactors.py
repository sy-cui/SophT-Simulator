import numpy as np
from typing import Dict, Iterable, Any, Optional, Literal, List
from elastica import CosseratRod, RigidBodyBase
from sopht.simulator.immersed_body.immersed_body_forcing_grid import (
    ImmersedBodyForcingGrid,
)
from sopht.simulator.immersed_body.immersed_boundary_interactions.ImmersedBoundaryInteraction import (
    ImmersedBoundaryInteraction,
)
from sopht.simulator.immersed_body.immersed_boundary_interactions.DirectBoundaryForcing import (
    DirectBoundaryForcing,
)
from sopht.simulator.immersed_body.immersed_boundary_interactions.PenaltyBoundaryForcing import (
    PenaltyBoundaryForcing,
)
from sopht.simulator.flow.navier_stokes_flow_simulators import (
    UnboundedNavierStokesFlowSimulator2D,
    UnboundedNavierStokesFlowSimulator3D,
)


def create_immersed_boundary_communicators(
    interaction_pair_list: Iterable[Dict[str, Any]],
    real_t: type = np.float64,
    eul_grid_coord_shift: Optional[float] = None,
    interp_kernel_width: int = 2,
    start_time: float = 0.0,
    interp_kernel_type: Literal["peskin", "cosine"] = "cosine",
    **kwargs,
) -> List:
    interactors = []
    for idx, interaction_pair in enumerate(interaction_pair_list):
        keys = interaction_pair.keys()
        assert (
            "flow_sim" in keys
        ), f"Interaction pair {idx}: Missing 'flow_sim' keyword."
        assert (
            "body_sim" in keys
        ), f"Interaction pair {idx}: Missing 'body_sim' keyword."
        assert (
            "forcing_grid" in keys
        ), f"Interaction pair {idx}: Missing 'forcing_grid' keyword."
        assert (
            "interaction_method" in keys
        ), f"Interaction pair {idx}: Missing 'interaction_method' keyword."

        flow_simulator = interaction_pair["flow_sim"]
        immersed_body = interaction_pair["body_sim"]
        forcing_grid_cls = interaction_pair["forcing_grid"]
        interaction_cls = interaction_pair["interaction_method"]

        assert isinstance(
            flow_simulator, UnboundedNavierStokesFlowSimulator2D
        ) or isinstance(flow_simulator, UnboundedNavierStokesFlowSimulator3D), (
            f"Interaction pair {idx}: Value associated with 'flow_sim' must be a "
            f"Navier Stokes Flow Simulator. "
        )

        assert issubclass(forcing_grid_cls, ImmersedBodyForcingGrid), (
            f"Interaction pair {idx}: Value associated with 'forcing_grid' must be a "
            f"sub-class of 'ImmersedBodyForcingGrid'."
        )

        assert issubclass(interaction_cls, ImmersedBoundaryInteraction), (
            f"Interaction pair {idx}: Value associated with 'interaction_method' must be a "
            f"sub-class of 'ImmersedBoundaryInteraction'."
        )

        body_flow_forces: np.ndarray
        body_flow_torques: np.ndarray

        if isinstance(immersed_body, RigidBodyBase):
            body_flow_forces = np.zeros(
                (3, 1),
                dtype=real_t,
            )
            body_flow_torques = np.zeros(
                (3, 1),
                dtype=real_t,
            )
            kwargs["rigid_body"] = immersed_body

        elif isinstance(immersed_body, CosseratRod):
            body_flow_forces = np.zeros(
                (3, immersed_body.n_elems + 1),
                dtype=real_t,
            )
            body_flow_torques = np.zeros(
                (3, immersed_body.n_elems),
                dtype=real_t,
            )
            kwargs["cosserat_rod"] = immersed_body

        else:
            try:
                rod, _ = immersed_body.communicate()
                n_elems = int(rod.n_elems[0])
                body_flow_forces = np.zeros(
                    (3, n_elems + 1),
                    dtype=real_t,
                )
                body_flow_torques = np.zeros(
                    (3, n_elems),
                    dtype=real_t,
                )
                kwargs["cosserat_rod_simulator"] = immersed_body

            except AttributeError:
                raise ValueError(
                    f"Interaction pair {idx}: Unrecognized immersed body type "
                    f"{type(immersed_body)}."
                )

        interactor: ImmersedBoundaryInteraction
        if interaction_cls.__name__ == "PenaltyBoundaryForcing":
            assert (
                "virtual_boundary_stiffness_coeff" in kwargs.keys()
                and "virtual_boundary_stiffness_coeff" in kwargs.keys()
            ), (
                f"Interaction pair {idx}: Missing 'virtual_boundary_stiffness_coeff' "
                f"or 'virtual_boundary_stiffness_coeff' in 'kwargs'."
            )
            interactor = PenaltyBoundaryForcing(
                grid_dim=flow_simulator.grid_dim,
                dx=flow_simulator.dx,
                eul_grid_forcing_field=flow_simulator.eul_grid_forcing_field,
                eul_grid_velocity_field=flow_simulator.velocity_field,
                forcing_grid_cls=forcing_grid_cls,
                body_flow_forces=body_flow_forces,
                body_flow_torques=body_flow_torques,
                real_t=flow_simulator.real_t,
                eul_grid_coord_shift=eul_grid_coord_shift,
                interp_kernel_width=interp_kernel_width,
                start_time=start_time,
                interp_kernel_type=interp_kernel_type,
                **kwargs,
            )

        elif interaction_cls.__name__ == "DirectBoundaryForcing":
            interactor = DirectBoundaryForcing(
                grid_dim=flow_simulator.grid_dim,
                dx=flow_simulator.dx,
                eul_grid_position_field=flow_simulator.position_field,
                eul_grid_velocity_field=flow_simulator.velocity_field,
                forcing_grid_cls=forcing_grid_cls,
                body_flow_forces=body_flow_forces,
                body_flow_torques=body_flow_torques,
                real_t=flow_simulator.real_t,
                eul_grid_coord_shift=eul_grid_coord_shift,
                interp_kernel_width=interp_kernel_width,
                start_time=start_time,
                interp_kernel_type=interp_kernel_type,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Interaction pair {idx}: Unsupported immersed body type "
                f"{interaction_cls.__name__}."
            )

        interactors.append(interactor)

    return interactors
