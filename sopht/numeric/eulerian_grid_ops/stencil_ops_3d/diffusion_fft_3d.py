import numpy as np
import pyfftw
import sopht.numeric.eulerian_grid_ops as spne


class DiffusionNumerical3D:
    def __init__(self, dx, diffusion_coefficient, scalar):
        self.dx = dx
        self.nu = diffusion_coefficient
        self.scalar = scalar
        self.temp = scalar.copy()

    def time_step(self, dt):
        self.temp[1:-1, 1:-1, 1:-1] = self.scalar[
            1:-1, 1:-1, 1:-1
        ] + self.nu * dt / 2 / self.dx**2 * (
            self.scalar[2:, 1:-1, 1:-1]
            + self.scalar[:-2, 1:-1, 1:-1]
            + self.scalar[1:-1, 2:, 1:-1]
            + self.scalar[1:-1, :-2, 1:-1]
            + self.scalar[2:, 1:-1, 1:-1]
            + self.scalar[:-2, 1:-1, 1:-1]
            - 6.0 * self.scalar[1:-1, 1:-1, 1:-1]
        )
        self.scalar[1:-1, 1:-1, 1:-1] += (
            self.nu
            * dt
            / self.dx**2
            * (
                self.temp[2:, 1:-1, 1:-1]
                + self.temp[:-2, 1:-1, 1:-1]
                + self.temp[1:-1, 2:, 1:-1]
                + self.temp[1:-1, :-2, 1:-1]
                + self.temp[1:-1, 1:-1, 2:]
                + self.temp[1:-1, 1:-1, :-2]
                - 6.0 * self.temp[1:-1, 1:-1, 1:-1]
            )
        )


class DSTPyFFTW3D:
    def __init__(
        self,
        grid_size: tuple[int, int, int],
        num_threads: int = 1,
        real_t: type = np.float64,
    ) -> None:
        """Class initializer."""
        self.grid_size_z, self.grid_size_y, self.grid_size_x = grid_size
        self.num_threads = num_threads
        self.real_t = real_t
        self.complex_dtype = np.complex64 if real_t == np.float32 else np.complex128
        self._create_fftw_plan()

    def _create_fftw_plan(self) -> None:
        """Create FFTW plan objects necessary for executing FFT later."""
        self.field_pyfftw_buffer = pyfftw.empty_aligned(
            (self.grid_size_z - 2, self.grid_size_y - 2, self.grid_size_x - 2),
            dtype=self.real_t,
        )
        self.fourier_field_pyfftw_buffer = pyfftw.empty_aligned(
            (self.grid_size_z - 2, self.grid_size_y - 2, self.grid_size_x - 2),
            dtype=self.real_t,
        )
        self.dst_plan = pyfftw.FFTW(
            self.field_pyfftw_buffer,
            self.fourier_field_pyfftw_buffer,
            axes=(0, 1, 2),
            direction=("FFTW_RODFT00", "FFTW_RODFT00", "FFTW_RODFT00"),
            flags=("FFTW_MEASURE", "FFTW_MEASURE", "FFTW_MEASURE"),
            threads=self.num_threads,
        )
        self.idst_plan = pyfftw.FFTW(
            self.fourier_field_pyfftw_buffer,
            self.field_pyfftw_buffer,
            axes=(0, 1, 2),
            direction=("FFTW_RODFT00", "FFTW_RODFT00", "FFTW_RODFT00"),
            flags=("FFTW_MEASURE", "FFTW_MEASURE", "FFTW_MEASURE"),
            threads=self.num_threads,
        )


class DiffusionDST3D:
    def __init__(
        self,
        flow_dx: float,
        grid_size: tuple[int, int, int],
        kinematic_viscosity: float,
        num_threads: int = 1,
        real_t: type = np.float64,
    ) -> None:
        grid_size_z, grid_size_y, grid_size_x = grid_size
        pyfftw_construct = DSTPyFFTW3D(
            grid_size=grid_size,
            num_threads=num_threads,
            real_t=real_t,
        )
        self.dst = pyfftw_construct.dst_plan
        self.idst = pyfftw_construct.idst_plan
        self.domain_buffer = pyfftw_construct.field_pyfftw_buffer
        self.domain_fourier_buffer = pyfftw_construct.fourier_field_pyfftw_buffer

        # Elementwise copy kernel
        self.elementwise_copy_pyst_kernel_3d = spne.gen_elementwise_copy_pyst_kernel_3d(
            real_t=real_t,
            num_threads=num_threads,
            fixed_grid_size=(grid_size_z - 2, grid_size_y - 2, grid_size_x - 2),
        )

        fourier_mode_x = np.arange(1, grid_size_x - 1) / (grid_size_x - 1)
        fourier_mode_y = np.arange(1, grid_size_y - 1) / (grid_size_y - 1)
        fourier_mode_z = np.arange(1, grid_size_z - 1) / (grid_size_z - 1)
        fourier_mode_zz, fourier_mode_yy, fourier_mode_xx = np.meshgrid(
            fourier_mode_z, fourier_mode_y, fourier_mode_x, indexing="ij"
        )
        self.inv_time_const = (
            -kinematic_viscosity
            * (np.pi / flow_dx) ** 2
            * (fourier_mode_xx**2 + fourier_mode_yy**2 + fourier_mode_zz**2)
        )
        self.normalization_factor = (
            1 / (2 * grid_size_x - 2) / (2 * grid_size_y - 2) / (2 * grid_size_z - 2)
        )

    def diffusion_timestep_dst(
        self,
        vector_field: np.ndarray,
        flow_dt: float,
    ) -> None:
        self._diffusion_timestep_dst_sub_step(
            field=vector_field[0],
            flow_dt=flow_dt,
        )
        self._diffusion_timestep_dst_sub_step(
            field=vector_field[1],
            flow_dt=flow_dt,
        )
        self._diffusion_timestep_dst_sub_step(
            field=vector_field[2],
            flow_dt=flow_dt,
        )

    def _diffusion_timestep_dst_sub_step(
        self,
        field: np.ndarray,
        flow_dt: float,
    ) -> None:
        self.elementwise_copy_pyst_kernel_3d(
            field=self.domain_buffer,
            rhs_field=field[1:-1, 1:-1, 1:-1],
        )

        self.dst()
        self.domain_fourier_buffer[...] *= self.normalization_factor * np.exp(
            self.inv_time_const * flow_dt
        )
        self.idst()

        self.elementwise_copy_pyst_kernel_3d(
            field=field[1:-1, 1:-1, 1:-1], rhs_field=self.domain_buffer
        )


if __name__ == "__main__":
    lx = 1.0
    grid_size = (64, 64, 64)
    dx = lx / grid_size[2]
    ly = grid_size[1] * dx
    lz = grid_size[0] * dx
    x = np.linspace(dx / 2, lx - dx / 2, grid_size[2])
    y = np.linspace(dx / 2, ly - dx / 2, grid_size[1])
    z = np.linspace(dx / 2, lz - dx / 2, grid_size[0])
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    u_fft = np.exp(
        -1000 * ((xx - lx / 2) ** 2 + (yy - ly / 2) ** 2 + (zz - lz / 2) ** 2)
    )
    nu = 1e-2
    u_num = u_fft.copy()
    num_cls = DiffusionNumerical3D(dx, nu, u_num)
    fft_cls = DiffusionDST3D(
        flow_dx=dx,
        grid_size=grid_size,
        kinematic_viscosity=nu,
    )

    t_end = 5.0
    dt_num = 0.9 * dx**2 / (6 * nu)
    t_num = 0.0
    while t_num < t_end:
        if t_end - t_num < dt_num:
            dt_num = t_end - t_num
        num_cls.time_step(dt_num)
        t_num += dt_num

    fft_cls._diffusion_timestep_dst_sub_step(field=u_fft, flow_dt=t_end)
    error = (u_fft - u_num) ** 2

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.use("macosx")

    index = int(grid_size[0] / 2)
    fig, ax = plt.subplots(ncols=2)
    ax[0].contourf(xx[index, :, :], yy[index, :, :], u_num[index, :, :])
    ax[1].contourf(xx[index, :, :], yy[index, :, :], u_fft[index, :, :])
    ax[0].set_title("Numerical")
    ax[1].set_title("FFT")
    plt.show()

    error_2norm = np.sqrt(np.sum(error) * dx**3 / (lx * ly * lz))
    print(error_2norm)
