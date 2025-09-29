"""TODO(jmdm): description of script.

Author:     jmdm
Date:       yyyy-mm-dd
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     In progress ⚙️
Status:     Paused ⏸️
Status:     Completed ✅
Status:     Incomplete ❌
Status:     Broken ⚠️
Status:     To Improve ⬆️

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.traceback import install

# Local libraries
# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP ---
SEED = 42
RNG = np.random.default_rng(SEED)

# --- TERMINAL OUTPUT SETUP ---
install(show_locals=True)
console = Console()


import torch


class NA_CPG_Matrix_Form(torch.nn.Module):
    """
    Implements the Normalized Asymmetric CPG (NA-CPG) in a matrix form,
    based on the model described in the paper.
    """

    def __init__(
        self,
        inherent_params,
        adjustable_params,
    ) -> None:
        """
        Initializes the NA-CPG network.

        Args:
            inherent_params (dict): Dictionary of inherent parameters:
                - 'alpha': learning rate
                - 'h': coupling coefficient
                - 'dt
                - 'n
            adjustable_params (dict): Dictionary of adjustable parameters:
                - 'omega': angular frequency (tensor of shape [n_oscillators, 1])
                - 'A': amplitude (tensor of shape [n_oscillators, 1])
                - 'b': offset ratio (tensor of shape [n_oscillators, 1])
                - 'h_a': asymmetry parameter (tensor of shape [n_oscillators, 1])
                - 'phase_diffs': phase differences between oscillators (tensor of shape [n_oscillators, n_oscillators])
            dt (float): Control period (time interval).
        """
        super().__init__()

        # Inherent parameters: do not change during learning
        self.n = inherent_params["n_oscillators"]
        self.alpha = inherent_params["alpha"]
        self.h = inherent_params["h"]
        self.dt = inherent_params["dt"]

        # Adjustable parameters: do change during learning
        self.omega = adjustable_params["omega"]
        self.A = adjustable_params["A"]
        self.b = adjustable_params["b"]
        self.h_a = adjustable_params["h_a"]
        self.phase_diffs = adjustable_params["phase_diffs"]

        # In case of mistake in initialization of phase differences
        for i in range(self.n):
            self.phase_diffs[i, i] = 0.0

        # Initialize the state of the CPGs (x, y)
        #epsilon = 0.001 
        # self.x = torch.zeros(self.n, 1)# - 0.5) * epsilon
        # self.y = torch.zeros(self.n, 1) #- 0.5) * epsilon
        self.x = 0.01 * torch.randn(self.n, 1)
        self.y = 0.01 * torch.randn(self.n, 1)

    def forward(
        self,
        x_old,
        y_old,
        x_old_dot=None,
    ):
        """
        Perform a single step of the CPG network update.

        Args:
            x_old (torch.Tensor): Tensor of old x-states of shape [n, 1].
            y_old (torch.Tensor): Tensor of old y-states of shape [n, 1].
            x_old_dot (torch.Tensor): Optional tensor of old x-dot values for asymmetry.

        Returns
        -------
            torch.Tensor: The new output angles (theta) of shape [n, 1].
        """
        # Compute r^2 for each oscillator [cite: 246]
        r_sq = x_old**2 + (y_old - self.b) ** 2
        
        
        #print("r_sq\n", r_sq)

        # Compute asymmetry parameter zeta [cite: 182, 254]
        # Using a small epsilon to prevent division by zero
        zeta = (
            1 - self.h_a * (x_old / (torch.abs(x_old_dot) + 1e-9))
            if x_old_dot is not None
            else torch.ones_like(x_old)
        )
        
        #print("zeta\n", zeta)

        # Main diagonal of the K matrix [cite: 279, 280]
        k_diag = self.alpha * (1 - r_sq)
        
        #print("k_diag\n", k_diag)

        # The main K matrix (2n x 2n)
        # k_matrix = torch.zeros(2 * self.n, 2 * self.n)
        # for i in range(self.n):
        #     k_matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = torch.tensor([
        #         [k_diag[i].item(), -self.omega[i].item() / zeta[i].item()],
        #         [self.omega[i].item() / zeta[i].item(), k_diag[i].item()],
        #     ])
        
        k_matrix = torch.zeros(2 * self.n, 2 * self.n)
        for i in range(self.n):
            k_matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = torch.tensor([
                [k_diag[i].squeeze(), -self.omega[i].squeeze() / zeta[i].squeeze()],
                [self.omega[i].squeeze() / zeta[i].squeeze(), k_diag[i].squeeze()],
            ])

        #print("k_matrix\n", k_matrix)

        # The rotation matrices R_ij for the coupling terms [cite: 281]
        r_matrices = torch.zeros(self.n, self.n, 2, 2)
        for i in range(self.n):
            for j in range(self.n):
                phase_diff = self.phase_diffs[i, j]
                cos_phi = torch.cos(phase_diff)
                sin_phi = torch.sin(phase_diff)

                # Create rotation matrix for all cases (diagonal will be identity since phase_diffs[i,i] = 0)
                r_matrices[i, j] = torch.tensor([
                    [cos_phi, -sin_phi],
                    [sin_phi, cos_phi],
                ])


        #print("r_matrices\n", r_matrices)

        # Construct the full (2n x 2n) R matrix from the R_ij matrices
        r_matrix = torch.zeros(2 * self.n, 2 * self.n)
        for i in range(self.n):
            for j in range(self.n):
                r_matrix[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
                    self.h * r_matrices[i, j]
                )

        #print("r_matrix\n", r_matrix)

        # State vector [x1, y1-b1, x2, y2-b2, ...]
        state_vector = torch.cat((x_old, y_old - self.b), dim=1).T.reshape(
            -1,
            1,
        )

        # Calculate state derivatives (x_dot, y_dot) [cite: 288, 294]
        state_dot_vector = torch.matmul(k_matrix + r_matrix, state_vector)
        state_dot_vector = self.constraint_function(state_dot_vector)

        # Update states
        self.x += state_dot_vector[0::2] * self.dt
        self.y += (state_dot_vector[1::2] + self.b) * self.dt

        # The output angle is given by A * y [cite: 243]
        
        output = self.A * self.y
        
        #print("angles\n", output)
        
        return output

    def constraint_function(self, state_dot_vector):
        # state_dot_vector shape: [2*n, 1]
        # omega shape: [n, 1]
        # For each oscillator, limit x_dot and y_dot based on omega
        k = 2.0  # scaling factor, tune as needed
        max_dot = k * self.omega.squeeze()  # shape: [n]
        # Expand to [2*n] for x_dot and y_dot
        max_dot_full = torch.repeat_interleave(max_dot, repeats=2).reshape(-1, 1)
        # Clamp each element
        return torch.clamp(state_dot_vector, -max_dot_full, max_dot_full)

if __name__ == "__main__":
    import numpy as np

    # Set   the outer grid size
    n = 3
    # Set the inner block size
    b = 2

    # A_4D has shape (n, n, b, b) -> (3, 3, 3, 3)
    A_4D = np.arange(n**2 * b**2).reshape(n, n, b, b)
    
    print(A_4D)

    # Transpose: (0, 1, 2, 3) -> (0, 2, 1, 3)
    A_transposed = A_4D.transpose(0, 2, 1, 3)

    # Reshape: (n*b) x (n*b) -> 9x9
    A_flattened = A_transposed.reshape(n * b, n * b)
    
    print(A_flattened)

    print(f"Flattened shape: {A_flattened.shape}")
    # ... print 9x9 matrix ...
    
#     """Entry point."""
#     # Example usage of the NA_CPG_Matrix_Form class
#     n_oscillators = 1
    
#     inherent_params = {
#         "n_oscillators": n_oscillators,
#         "alpha": 30,
#         "h": 5,
#         "dt": 0.001,
#     }
    
#     # Phase differences in DEGREES: [phi_2 - phi_1, phi_3 - phi_2, phi_4 - phi_3]
#     initial_phase_diffs_degrees = [30.0, 15.0, 15.0]

#     # Calculate absolute phases relative to the first oscillator (phi_1 = 0)
#     absolute_phases_deg = [0.0] + list(np.cumsum(initial_phase_diffs_degrees))

#     # Calculate the full phase difference matrix Phi_ij = phi_i - phi_j
#     phase_diffs_deg = np.array(absolute_phases_deg)[:, np.newaxis] - np.array(absolute_phases_deg)

#     # Convert to RADIANS
#     phase_diffs_rad = torch.tensor(phase_diffs_deg * (np.pi / 180.0), dtype=torch.float32)

#     adjustable_params = {
#         "omega": torch.ones(4, 1) * (2.0 * torch.pi),  # 1 Hz oscillation
#         "A": torch.tensor([10.0, 15.0, 20.0, 25.0]).reshape(4, 1),  # Amplitudes (paper uses these)
#         "b": torch.zeros(4, 1),                        # Offset, start with zero
#         "h_a": torch.ones(4, 1) * 0.1,                 # Asymmetry, 0.1 is moderate
#         "phase_diffs": phase_diffs_rad,                # Your phase difference matrix
#     }

#     # adjustable_params = {
#     #     # Fixed frequency of 2*pi rad/s
#     #     "omega": torch.ones(n_oscillators, 1) * (2.0 * torch.pi),
#     #     "A": torch.tensor([10.0, 15.0, 20.0, 25.0]).reshape(n_oscillators, 1),
#     #     "b": torch.zeros(n_oscillators, 1),
#     #     "h_a": torch.ones(n_oscillators, 1) * 0.1,
#     #     "phase_diffs": phase_diffs_rad,
#     # }
    
#     # adjustable_params = {
#     #         "omega": torch.ones(n_oscillators, 1) * 2.0,
#     #         "A": torch.ones(n_oscillators, 1) * 1.0,
#     #         "b": torch.zeros(n_oscillators, 1),
#     #         "h_a": torch.ones(n_oscillators, 1) * 0.1,
#     #         "phase_diffs": torch.zeros(n_oscillators, n_oscillators),
#     # }
    
    
#     cpg_network = NA_CPG_Matrix_Form(inherent_params, adjustable_params)
#     x_old = cpg_network.x
#     y_old = cpg_network.y
    
#     runs = 10000
#     outputs = []

#     for _ in range(runs):
#         out = cpg_network.forward(x_old, y_old)
#         outputs.append(out.detach().numpy())    
        
#         x_old, y_old = cpg_network.x, cpg_network.y

#     outputs = np.squeeze(np.array(outputs))

#     time_steps = np.arange(runs) * inherent_params["dt"]


#     plt.figure(figsize=(8, 4))
#     for i in range(n_oscillators):
#         plt.plot(time_steps, outputs[:, i], label=f"Oscillator {i+1}")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Output")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('test.png')
# # 
