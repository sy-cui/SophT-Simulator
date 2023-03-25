import numpy as np
import pyfftw
import sopht.numeric.eulerian_grid_ops as spne


class DSTPyFFTW2D:
    def __init__(
        self,
        grid_size: tuple[int, int],
        num_threads: int = 1,
        real_t: type = np.float64,
    ) -> None:
        """Class initializer."""
        self.grid_size_y, self.grid_size_x = grid_size
        self.num_threads = num_threads
        self.real_t = real_t
        self.complex_dtype = np.complex64 if real_t == np.float32 else np.complex128
        self._create_fftw_plan()

    def _create_fftw_plan(self) -> None:
        """Create FFTW plan objects necessary for executing FFT later."""
        self.field_pyfftw_buffer = pyfftw.empty_aligned(
            (self.grid_size_y - 2, self.grid_size_x - 2),
            dtype=self.real_t,
        )
        self.fourier_field_pyfftw_buffer = pyfftw.empty_aligned(
            (self.grid_size_y - 2, self.grid_size_x - 2),
            dtype=self.real_t,
        )
        self.dst_plan = pyfftw.FFTW(
            self.field_pyfftw_buffer,
            self.fourier_field_pyfftw_buffer,
            axes=(0, 1),
            direction=("FFTW_RODFT00", "FFTW_RODFT00"),
            flags=("FFTW_MEASURE", "FFTW_MEASURE"),
            threads=self.num_threads,
        )
        self.idst_plan = pyfftw.FFTW(
            self.fourier_field_pyfftw_buffer,
            self.field_pyfftw_buffer,
            axes=(0, 1),
            direction=("FFTW_RODFT00", "FFTW_RODFT00"),
            flags=("FFTW_MEASURE", "FFTW_MEASURE"),
            threads=self.num_threads,
        )


class DiffusionDST2D:
    def __init__(
        self,
        flow_dx: float,
        grid_size: tuple[int, int],
        kinematic_viscosity: float,
        num_threads: int = 1,
        real_t: type = np.float64,
    ):
        # Initialize pyFFTW plans and field buffers
        grid_size_y, grid_size_x = grid_size
        pyfftw_construct = DSTPyFFTW2D(
            grid_size=grid_size,
            num_threads=num_threads,
            real_t=real_t,
        )

        self.dst = pyfftw_construct.dst_plan
        self.idst = pyfftw_construct.idst_plan
        self.domain_buffer = pyfftw_construct.field_pyfftw_buffer
        self.domain_fourier_buffer = pyfftw_construct.fourier_field_pyfftw_buffer

        # Elementwise copy kernel
        self.elementwise_copy_pyst_kernel_2d = spne.gen_elementwise_copy_pyst_kernel_2d(
            real_t=real_t,
            num_threads=num_threads,
            fixed_grid_size=(grid_size_y - 2, grid_size_x - 2),
        )

        fourier_mode_x = np.arange(1, grid_size_x - 1) / (grid_size_x - 1)
        fourier_mode_y = np.arange(1, grid_size_y - 1) / (grid_size_y - 1)
        fourier_mode_xx, fourier_mode_yy = np.meshgrid(fourier_mode_x, fourier_mode_y)
        self.inv_time_const = (
            -kinematic_viscosity
            * (np.pi / flow_dx) ** 2
            * (fourier_mode_xx**2 + fourier_mode_yy**2)
        )
        self.normalization_factor = 1 / (2 * grid_size_x - 2) / (2 * grid_size_y - 2)

    def diffusion_timestep_dst(self, field, flow_dt):
        # Copy from diffused field to pyFFTW domain buffer
        self.elementwise_copy_pyst_kernel_2d(
            field=self.domain_buffer, rhs_field=field[1:-1, 1:-1]
        )

        self.dst()
        self.domain_fourier_buffer[...] *= self.normalization_factor * np.exp(
            self.inv_time_const * flow_dt
        )
        self.idst()

        # Copy from pyFFTW domain buffer to diffused field
        self.elementwise_copy_pyst_kernel_2d(
            field=field[1:-1, 1:-1],
            rhs_field=self.domain_buffer,
        )
