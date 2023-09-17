import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
import click
from elastic_fish_case import ElasticFishSimulator
from elastic_fish_utils.fish_grid import FishSurfaceForcingGrid
from elastic_fish_utils.fish_geometry import create_fish_geometry


def elastic_fish_swimming_case(
    non_dim_final_time: float,
    n_elem: int,
    grid_size: tuple[int, int, int],
    surface_grid_density_for_largest_element: int,
    slenderness_ratio: float,
    mass_ratio: float,
    non_dim_bending_stiffness: float,
    actuation_reynolds_number: float,
    coupling_stiffness: float = -2e4 / 20 / 2 / 2 * 128,
    coupling_damping: float = -1e1 / 20 / 2 / 2 * 256,
    num_threads: int = 4,
    precision: str = "single",
    save_data: bool = False,
) -> None:
    # =================COMMON SIMULATOR STUFF=======================
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    rho_f = 1.0
    base_length = 1.0
    x_range = 4 * base_length
    y_range = grid_size_y / grid_size_x * x_range
    z_range = grid_size_z / grid_size_x * x_range
    # =================PYELASTICA STUFF BEGIN=====================
    period = 1
    final_time = non_dim_final_time * period
    vel_scale = base_length / period
    rho_s = mass_ratio * rho_f
    base_diameter = base_length / slenderness_ratio
    base_radius = base_diameter / 2
    moment_of_inertia = np.pi / 4 * base_radius**4
    youngs_modulus = (
        non_dim_bending_stiffness
        * rho_f
        * vel_scale**2
        * base_length**3
        * base_diameter
        / moment_of_inertia
    )

    origin = np.array(
        [0.85 * x_range - 0.5 * base_length, 0.5 * y_range, 0.5 * z_range]
    )
    fish_sim = ElasticFishSimulator(
        final_time=final_time,
        period=period,
        n_elements=n_elem,
        rod_density=rho_s,
        youngs_modulus=youngs_modulus,
        base_length=base_length,
        origin=origin,
        normal=np.array([0.0, 0.0, 1.0]),
        plot_result=True,
        damping_constant=0.16,
    )

    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    kinematic_viscosity = base_length * vel_scale / actuation_reynolds_number
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
        filter_vorticity=True,
        filter_setting_dict={"order": 5, "type": "convolution"},
    )
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=fish_sim.shearable_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=FishSurfaceForcingGrid,
        surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
    )
    fish_sim.simulator.add_forcing_to(fish_sim.shearable_rod).using(
        sps.FlowForces,
        cosserat_rod_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    # =================TIMESTEPPING====================
    fish_sim.finalize()

    if save_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        io_origin = np.array(
            [
                flow_sim.position_field[z_axis_idx].min(),
                flow_sim.position_field[y_axis_idx].min(),
                flow_sim.position_field[x_axis_idx].min(),
            ]
        )
        io_dx = flow_sim.dx * np.ones(grid_dim)
        io_grid_size = np.array(grid_size)
        io = spu.IO(dim=grid_dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )
        # Initialize sphere IO
        rod_io = spu.IO(dim=grid_dim, real_dtype=real_t)
        rod_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=cosserat_rod_flow_interactor.forcing_grid.position_field,
            lagrangian_grid_name="fish",
            vector_3d=cosserat_rod_flow_interactor.lag_grid_forcing_field,
        )

    foto_timer = 0.0
    foto_timer_limit = period / 30
    timescale = period
    time_history = []
    force_history = []
    com_velocity_history = []
    com_position_history = []

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    # iterate
    while flow_sim.time < final_time:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(
                f"Vorticity z, time: {flow_sim.time / timescale:.2f}, "
                f"distance: {(fish_sim.shearable_rod.position_collection[0, 0] - origin[0]):.6f}"
            )
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx, grid_size_z // 2, :, :],
                flow_sim.position_field[y_axis_idx, grid_size_z // 2, :, :],
                flow_sim.vorticity_field[z_axis_idx, grid_size_z // 2, :, :],
                levels=np.linspace(-25, 25, 100),
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            element_position = 0.5 * (
                fish_sim.shearable_rod.position_collection[:, 1:]
                + fish_sim.shearable_rod.position_collection[:, :-1]
            )
            # for plotting rod with correct radii for reference see
            # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot
            # -axes-scatter-markersize-by-x-scale/48174228#48174228
            scaling_factor = (
                ax.get_window_extent().width
                / max(flow_sim.x_range, flow_sim.y_range)
                * 72.0
                / fig.dpi
            )
            ax.scatter(
                element_position[x_axis_idx],
                element_position[y_axis_idx],
                s=4 * (scaling_factor * fish_sim.shearable_rod.radius) ** 2,
                c="k",
            )
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.5d" % (flow_sim.time * 100)) + ".png",
            )
            time_history.append(flow_sim.time)
            forces = np.sum(cosserat_rod_flow_interactor.lag_grid_forcing_field, axis=1)
            force_history.append(forces.copy())
            com_velocity_history.append(
                fish_sim.shearable_rod.compute_velocity_center_of_mass().copy()
            )
            com_position_history.append(
                fish_sim.shearable_rod.compute_position_center_of_mass().copy()
            )
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f},"
                " grid deviation L2 error: "
                f"{cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm():.6f},"
                f" fish pos: {fish_sim.shearable_rod.position_collection[0, 0]:.6f},"
                f" total force: {forces},"
                f" force norm: {np.linalg.norm(forces)}"
                f"flow_dt: {flow_sim.compute_stable_timestep(dt_prefac=0.125):.6f}"
            )
            with open("fish_velocity_vs_time.csv", "ab") as f:
                np.savetxt(
                    f,
                    np.c_[
                        np.hstack(
                            (
                                flow_sim.time,
                                com_velocity_history[-1],
                                np.linalg.norm(com_velocity_history[-1]),
                            )
                        )
                    ].T,
                    delimiter=",",
                )

            with open("fish_com_position_vs_time.csv", "ab") as f:
                np.savetxt(
                    f,
                    np.c_[np.hstack((flow_sim.time, com_position_history[-1]))].T,
                    delimiter=",",
                )

            with open("fish_forces_vs_time.csv", "ab") as f:
                np.savetxt(
                    f,
                    np.c_[
                        np.hstack(
                            (
                                flow_sim.time,
                                force_history[-1],
                                np.linalg.norm(force_history[-1]),
                            )
                        )
                    ].T,
                    delimiter=",",
                )

            if save_data:
                io.save(
                    h5_file_name="sopht_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                rod_io.save(
                    h5_file_name="rod_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.125 * 2)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, fish_sim.dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            rod_time = fish_sim.time_step(rod_time, local_rod_dt)
            # timestep the cosserat_rod_flow_interactor
            cosserat_rod_flow_interactor.time_step(dt=local_rod_dt)
        # evaluate feedback/interaction between flow and rod
        cosserat_rod_flow_interactor()

        flow_sim.time_step(dt=flow_dt)

        # update timer
        foto_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=30
    )

    # Save data
    np.savetxt(
        "fish_forces_vs_time.csv",
        np.c_[
            np.array(time_history),
            np.array(force_history),
            np.linalg.norm(np.array(force_history), axis=1),
        ],
        delimiter=",",
        header="time, force x, force y, force z, force norm",
    )
    np.savetxt(
        "fish_velocity_vs_time.csv",
        np.c_[
            np.array(time_history),
            np.array(com_velocity_history),
            np.linalg.norm(np.array(com_velocity_history), axis=1),
        ],
        delimiter=",",
        header="time, vel x, vel y, vel z, vel norm",
    )
    np.savetxt(
        "fish_com_position_vs_time.csv",
        np.c_[
            np.array(time_history),
            np.array(com_position_history),
        ],
        delimiter=",",
        header="time, pos x, pos y, pos z",
    )


if __name__ == "__main__":

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option("--nx", default=320, help="Number of grid points in x direction.")
    def simulate_fish_swimming(num_threads: int, nx: int) -> None:
        ny = nx // 2
        nz = nx // 8
        # in order Z, Y, X
        grid_size = (nz, ny, nx)
        surface_grid_density_for_largest_element = nx // 16  # 8
        n_elem = nx // 4  # int(nx // 4 / 1.4)  # 32

        exp_activation_period = 1.0
        final_time = 6.0 * exp_activation_period

        exp_base_length = 1.0
        exp_rho_s = 1e3 / 15  # kg/m3
        exp_rho_f = 1e3 / 15  # kg/m3
        exp_youngs_modulus = 15e5  # Pa
        exp_kinematic_viscosity = 1.4e-4
        exp_mass_ratio = exp_rho_s / exp_rho_f
        width, _ = create_fish_geometry(exp_base_length / n_elem * np.ones(n_elem))
        exp_base_radius = width[0]
        exp_base_diameter = 2 * exp_base_radius
        exp_slenderness_ratio = exp_base_length / exp_base_diameter
        exp_velocity_scale = exp_base_length / exp_activation_period
        exp_moment_of_inertia = np.pi / 4 * exp_base_radius**4
        exp_bending_rigidity = exp_youngs_modulus * exp_moment_of_inertia
        exp_non_dim_bending_stiffness = exp_bending_rigidity / (
            exp_rho_f
            * exp_velocity_scale**2
            * exp_base_length**3
            * exp_base_diameter
        )
        exp_actuation_reynolds_number = (
            exp_base_length * exp_velocity_scale / exp_kinematic_viscosity
        )
        exp_non_dimensional_final_time = final_time / exp_activation_period

        elastic_fish_swimming_case(
            non_dim_final_time=exp_non_dimensional_final_time,
            n_elem=n_elem,
            grid_size=grid_size,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
            slenderness_ratio=exp_slenderness_ratio,
            mass_ratio=exp_mass_ratio,
            non_dim_bending_stiffness=exp_non_dim_bending_stiffness,
            actuation_reynolds_number=exp_actuation_reynolds_number,
            save_data=False,
            num_threads=num_threads,
        )

    simulate_fish_swimming()
