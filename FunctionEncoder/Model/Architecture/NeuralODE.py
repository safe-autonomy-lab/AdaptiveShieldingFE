import math
from typing import Union, Tuple, Optional

import torch

# Assuming these imports are in your project structure
from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.Util import get_encoder
from FunctionEncoder.Model.Architecture.MLP import get_activation, MLP


def rk4_difference_only(model, xs, dt):
    """
    Runge-Kutta 4th order method for solving ODEs.
    This method correctly computes the change in state.
    """
    k1 = model(xs)
    k2 = model(xs + dt / 2 * k1)
    k3 = model(xs + dt / 2 * k2)
    k4 = model(xs + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class NeuralODE(BaseArchitecture):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=77, n_layers=4, ode_state_size=77, learn_basis_functions=True, *args, **kwargs):
        """Modified to account for encoder and decoder."""
        input_dim = input_size[0]
        output_dim = output_size[0]
        
        # Encoder params
        n_params = (input_dim + 1) * ode_state_size
        
        # Dynamics model params (input=output=ode_state_size)
        dynamics_params = (ode_state_size + 1) * hidden_size + \
                          (hidden_size + 1) * hidden_size * (n_layers - 2) + \
                          (hidden_size + 1) * ode_state_size
        
        if learn_basis_functions:
            n_params += dynamics_params * n_basis
        else:
            n_params += dynamics_params
            
        # Decoder params
        n_params += (ode_state_size + 1) * output_dim
        
        return n_params

    def __init__(self,
                 input_size: Tuple[int],
                 output_size: Tuple[int],
                 n_basis: int = 100,
                 hidden_size: int = 77,
                 # New parameter for the internal dimension of the ODE
                 ode_state_size: int = 77,
                 n_layers: int = 4,
                 activation: str = "relu",
                 learn_basis_functions=True,
                 dt: float = 0.1,
                 encoder_type: Optional[str] = None,
                 encoder_kwargs: dict = {},
                 **kwargs):
        super(NeuralODE, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        
        self.history_encoder = get_encoder(encoder_type, **encoder_kwargs) if encoder_type is not None else None
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions
        self.dt = dt

        if not self.learn_basis_functions:
            n_basis = 1
            self.n_basis = 1

        # 1. Encoder maps from input_size to the internal ODE state size
        self.encoder = MLP(
            input_size=input_size, 
            output_size=(ode_state_size,), 
            n_basis=1, 
            hidden_size=hidden_size, 
            n_layers=2,
            activation=activation, 
            learn_basis_functions=False
        )

        # 2. The core dynamics models now operate on the internal state size
        self.dynamics_models = torch.nn.ModuleList([
            MLP(
              input_size=(ode_state_size,), # Input is the internal state
              output_size=(ode_state_size,),# Output is also the internal state
              n_basis=1,
              hidden_size=hidden_size,
              n_layers=n_layers,
              activation=activation,
              learn_basis_functions=False,
            )
            for _ in range(n_basis)
        ])
        
        # 3. Decoder maps from the internal ODE state size back to the desired output_size
        self.decoder = MLP(
            input_size=(ode_state_size,), 
            output_size=output_size, 
            n_basis=1, 
            hidden_size=hidden_size,
            n_layers=2,
            activation=activation, 
            learn_basis_functions=False
        )

    def forward(self, x):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"
        
        if self.history_encoder is not None:
            if len(x.shape) == 4:
                original_shape = x.shape
                batch_size, history_length, feature_dim = original_shape[0] * original_shape[1], original_shape[2], original_shape[3]
                x_reshaped = x.view(batch_size, history_length, feature_dim)
                x = self.history_encoder(x_reshaped).view(original_shape[0], original_shape[1], -1)
            else:
                x = self.history_encoder(x)

        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)

        # 1. Encode the input to the initial hidden state for the ODE
        h0 = self.encoder(x)
        
        # 2. Solve the ODE for the change in state (delta_h)
        delta_h_list = [rk4_difference_only(model, h0, self.dt) for model in self.dynamics_models]
        delta_h_stacked = torch.stack(delta_h_list, dim=-1) # Shape: (..., ode_state_size, n_basis)
        
        # 3. Calculate the final hidden state
        h_final = h0.unsqueeze(-1) + delta_h_stacked # Add the change to the initial state
        
        # 4. Decode the final hidden state to the desired output dimension
        batch_dims = h_final.shape[:-2]
        ode_dim = h_final.shape[-2]
        num_bases = h_final.shape[-1]
        
        # Permute and reshape for batch decoding: (..., S, N) -> (..., N, S) -> (B*N, S)
        h_final_reshaped = h_final.permute(*range(len(batch_dims)), -1, -2).reshape(-1, ode_dim)

        # Apply decoder
        decoded_reshaped = self.decoder(h_final_reshaped)
        
        # Reshape and permute back to get the final output Gs
        output_dim = self.output_size[0]
        # (B*N, O) -> (..., N, O) -> (..., O, N)
        Gs = decoded_reshaped.view(*batch_dims, num_bases, output_dim).permute(*range(len(batch_dims)), -1, -2)

        if not self.learn_basis_functions:
            Gs = Gs.view(*x.shape[:2], *self.output_size)
        
        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs