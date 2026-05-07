# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:48:53 2025

@author: pedro
"""

#%% Import packages

# Standard functions
import os
import dill
import numpy as np
import matplotlib.pyplot as plt

# JAX functions
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn

# Import functions from packages
from tqdm import tqdm
from typing import Sequence, Callable
from CoolProp.CoolProp import PropsSI, PhaseSI
from sklearn.model_selection import train_test_split
from CoolSurrogateProp.SaveAndLoad_EoS import afns_to_str, save_jax_model

import CoolProp.CoolProp as CoolProp
# Add REFPROP database in case it is necessary
CoolProp.set_config_string(CoolProp.ALTERNATIVE_REFPROP_PATH,'c:\\Program Files (x86)\\REFPROP\\')

#%% Model and optimizer settings

"Class storing surrogate and optimizer settings"
class Settings:
    # Define core settings
    def __init__(self,
                 # --- ANN settings --- #
                 n_layers, 
                 hidden_dim, 
                 activation_fns,
                 # --- Training data settings --- #
                 test_size = 0.2, 
                 train_size = 0.8,
                 N_data_points = 100,
                 # --- Optimizer settings --- #
                 N_epochs = int(1e5), 
                 N_save = int(50), 
                 eta0 = 1e-4,
                 deriv_penalty = 1e-2,
                 batch_frac = 0.5,
                 weight_decay = 1e-4,
                 # Random seed
                 seed = 42
                 ):
        # --- ANN settings --- #
        self.key = jax.random.PRNGKey(seed) 
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation_fns = activation_fns
        # --- Training data batch settings --- #
        self.test_size = test_size
        self.train_size = train_size
        self.train_batch_seed = seed
        self.N_data_points = N_data_points
        # --- Optimizer settings --- #
        self.N_epochs = N_epochs
        self.N_save = N_save
        self.eta0 = eta0
        self.deriv_penalty = deriv_penalty
        self.batch_frac = batch_frac
        self.weight_decay = weight_decay
        return

#%% Surrogate architecture

"ANN architecture for CoolProp surrogate"
class PropNN(nn.Module):
    # Define class variables, i.e., model inputs
    n_layers: int # No. of hidden layers
    activation_fns: Sequence[Callable[[jnp.ndarray], jnp.ndarray]] # List of activation functions
    hidden_dim: Sequence[int] # Depth of each hidden layer
    
    "Network architecture: model.apply(w,x), where x in [np, ns]"
    # !!! X CANNOT BE A 1D-Array
    @nn.compact
    def __call__(self, x):
        # Loop through each layer
        for i in range(self.n_layers):
            # Linear transformation
            x = nn.Dense(self.hidden_dim[i])(x)
            # Activation function
            x = self.activation_fns[i](x)
        # Linear transfrom to the output node
        x = nn.Dense(1)(x)
        # Return output quantity
        return x.squeeze()
    
    "Scaling parameters"
    def surrogate_params(self,w_opt,x_hat_params,y_hat_params):
        # Store optimal weights
        self.w_opt = w_opt
        # Scaling parameters for ANN input and output
        self.x_hat_params = x_hat_params
        self.y_hat_params = y_hat_params
        return
    
    "Evaluate fluid property"
    def compute(self, x):
        # Scale inputs
        x_hat = (x - self.x_hat_params[0])/self.x_hat_params[1]
        # Scaled outputs
        y_hat = self.apply(self.w_opt, x_hat)
        # Rescale outputs
        y = self.y_hat_params[1]*y_hat + self.y_hat_params[0]
        return y
    
#%% Generate training data

"Generate X, Y, dYdx training data"
def Training_Data(Y_OUT, # Output fluid property
                  Phase, # Target fluid phase
                  Fluid, # Working fluid
                  P_max, T_max, P_min, T_min, # Pressure-temperature limits
                  X_IN=['P','T'], # Inputs to evaluate fluid property
                  N = 100, # Number of data points
                  ):
    # =========================================================================
    # Pressure-temperature range (fluid-specific)
    # =========================================================================
    # Assign minimum pressure and temperature
    if P_min is None: P_min = PropsSI('PTRIPLE', Fluid) + 0.1e5
    if T_min is None: T_min = PropsSI('TTRIPLE', Fluid) + 0.150
    
    # Number of inputs
    Nx = len(X_IN)
    
    # Vapor OR Liquid phase
    if Nx == 2:
        # Pressure and temperature range
        P_1d = np.linspace(P_min, P_max, num=N)
        T_1d = np.linspace(T_min, T_max, num=N)
        
        # =========================================================================
        # Generate pressure-temperature grid within min-max values
        # =========================================================================
        # Pressure and temperature mesh-grid
        P_2d, T_2d = np.meshgrid(P_1d, T_1d, indexing='xy')
        # Flatten grid into one-dimensional pressure-temperature pairs
        P_ = P_2d.flatten(); T_ = T_2d.flatten()
        # Initialize fluid phase
        Phi_ = []
        # Loop through each P,T pair
        for k in tqdm(range(N*N)):
            # Evaluate phase for the pressure-temperature pair
            Phi_.append(PhaseSI('P', P_[k], 'T', T_[k], Fluid))
        # Convert phase list to array
        Phi_ = np.array(Phi_)
        
        # =========================================================================
        # Identify indices of the desired fluid phase
        # =========================================================================
        # Input phase: gas
        if Phase.lower() == 'gas':
            idx_Phi_g  = np.where((Phi_ == 'gas'))[0]
            idx_Phi_sg = np.where((Phi_ == 'supercritical_gas'))[0]
            idx_Phi = np.hstack((idx_Phi_g,idx_Phi_sg))
        # Input phase: liquid
        elif Phase.lower() == 'liquid':
            idx_Phi = np.where((Phi_ == 'liquid'))[0]
        # Not-valid phase
        else:
            print('Input phase NOT valid!')
            
        # =========================================================================
        # Generate training data arrays
        # =========================================================================
        # Generate pressure-temperature training data
        P_Train = P_[idx_Phi]; T_Train = T_[idx_Phi]
        # Evaluate training data, i.e., X_train = [X0, X1]
        X0 = P_Train if X_IN[0] == 'P' else PropsSI(X_IN[0], 'P', P_Train, 'T', T_Train, Fluid)
        X1 = T_Train if X_IN[1] == 'T' else PropsSI(X_IN[1], 'P', P_Train, 'T', T_Train, Fluid)
        # Group pressure-temperature training data
        X_Train = np.hstack((X0[:,None], X1[:,None]))
        # Training fluid property data
        Y_Train = PropsSI(Y_OUT, X_IN[0], X0, X_IN[1], X1, Fluid)
        # Transport properties have no partial derivatives in CoolProp
        if "conductivity" in Y_OUT.lower() or "viscosity" in Y_OUT.lower() or "surface_tension" in Y_OUT.lower():
            dYdX0_Train = np.ones_like(X0)
            dYdX1_Train = np.ones_like(X0)
        # Compute thermodynamic partial derivatives
        else:
            # Thermodynamic partial derivatives
            dYdX0_Train = PropsSI('d(%s)/d(%s)|%s' %(Y_OUT, X_IN[0], X_IN[1]), X_IN[0], X0, X_IN[1], X1, Fluid)
            dYdX1_Train = PropsSI('d(%s)/d(%s)|%s' %(Y_OUT, X_IN[1], X_IN[0]), X_IN[0], X0, X_IN[1], X1, Fluid)
        # Group partial derivatives
        dYdX_Train = np.hstack((dYdX0_Train[:,None], dYdX1_Train[:,None]))
    
    # Saturated vapor OR saturated liquid phase
    elif Nx == 1:
        # =========================================================================
        # Generate pressure-temperature grid within min-max values
        # =========================================================================
        # Overwrite maximum pressure and temperature to the critical point
        P_max = PropsSI('P_CRITICAL', Fluid) - 1.0e5
        T_max = PropsSI('T_CRITICAL', Fluid) - 1.000
        # Input training pressure-temperature
        P_Train = np.linspace(P_min, P_max, num=N)
        T_Train = np.linspace(T_min, T_max, num=N)
        
        # =========================================================================
        # Identify "quality" for the desired phase
        # =========================================================================
        Q = 1 if Phase.lower() == 'saturated_vapor' else 0
        
        # Evaluate training data, i.e., X_train
        if X_IN == 'P':
            X_Train = P_Train[:,None]
        elif X_IN == 'T':
            X_Train = T_Train[:,None]
        else:
            X_Train = PropsSI(X_IN, 'P', P_Train.ravel(), 'Q', Q, Fluid)[:,None]
        # Training fluid property data
        Y_Train = PropsSI(Y_OUT, X_IN, X_Train.ravel(), 'Q', Q, Fluid)
        # Transport properties have no partial derivatives in CoolProp
        if "surface_tension" in Y_OUT.lower():
            dYdX_Train = np.ones_like(X_Train.ravel())
        # Compute thermodynamic partial derivatives
        else:
            # Saturation partial derivatives
            try:
                dYdX_Train = PropsSI('d(%s)/d(%s)|sigma' %(Y_OUT, X_IN), X_IN, X_Train.ravel(), 'Q', Q, Fluid)
            except:
                dYdX_Train = dYdX_numeric(Y_OUT, X_IN, X_Train.ravel(), Q, Fluid)
    else:
        print('Invalid number of inputs!')
        print('Nx = 2 (liquid, vapor) OR Nx = 1 (saturation line)')
    
    # Return data
    return X_Train, Y_Train, dYdX_Train

def dYdX_numeric(Y_OUT, X_IN, X, Q, Fluid, h=1e-5):
    Xp = X*(1 + h)
    Xm = X*(1 - h)
    Yp = PropsSI(Y_OUT, X_IN, Xp, 'Q', Q, Fluid)
    Ym = PropsSI(Y_OUT, X_IN, Xm, 'Q', Q, Fluid)
    return (Yp - Ym) / (2*h)

#%% Surrogate training

"Train Neural CoolProp Surrogate"
def Model_Training(X_, Y_, dYdX_, settings, w0 = None):
    # =========================================================================
    # Initialize model
    # =========================================================================
    model = PropNN(n_layers       = settings.n_layers,
                   hidden_dim     = settings.hidden_dim,
                   activation_fns = settings.activation_fns)
    # Initialize weights
    if w0 is None: w0 = model.init(settings.key, X_)
    
    # =========================================================================
    # Setup training
    # =========================================================================
    
    # Define loss function (MSE)
    def Loss(w, X, Y, dYdX, lmda, epoch, eps=1e-8, alpha=0.3, warmup_start=200, warmup_end=1000):
        # Define model output for a single sample
        def model_output(x):
            return model.apply(w, x[None]).squeeze()

        # Predictions
        Y_pred, dYdX_pred = jax.vmap(jax.value_and_grad(model_output))(X)
        
        # Errors
        Err_Y    = Y_pred    - Y
        Err_dYdX = dYdX_pred - dYdX
        
        # Losses
        L_Y_abs    = jnp.mean(Err_Y**2)           / jnp.mean(Y**2)
        L_dYdX_abs = jnp.mean(Err_dYdX**2,axis=0) / jnp.mean(dYdX**2,axis=0)
        
        # Scale gradient penalty as iterations progress
        lmda_eff = settings.deriv_penalty
        
        # Gradually enforce gradient penalty
        # Annealing schedule: 0 before warmup_start, linear ramp, then flat
        progress  = (epoch - warmup_start) / (warmup_end - warmup_start)  # 0 -> 1
        progress  = jnp.clip(progress, 0.0, 1.0)                          # clamp
        lmda_eff  = progress * settings.deriv_penalty
        
        # Return
        return L_Y_abs + lmda_eff * jnp.sum( L_dYdX_abs ) / L_dYdX_abs.size 
    
    
    # Training iteration
    @jax.jit
    def Train_step(w, opt_state, X_train, Y_train, dYdX_train, penalty, epoch):
        # Evaluate loss value and gradient
        loss, grads = jax.value_and_grad(Loss)(w, X_train, Y_train, dYdX_train, penalty, epoch)
        # Optimizer step
        updates, opt_state = optimizer.update(grads, opt_state, w)
        # Update weights
        w = optax.apply_updates(w, updates)
        # Return weights, optimizer state, loss value
        return w, opt_state, loss
    
    # =========================================================================
    # Setup optimizer
    # =========================================================================
    # Initialize loss history
    loss_train = [] # Training data
    loss_valid = [] # Validation data
    # Initial weight guess
    wp = w0
    # Train with JAX's optimizers
    optimizer = optax.adamw(settings.eta0, weight_decay=settings.weight_decay)
    opt_state = optimizer.init(w0)
    
    # =========================================================================
    # Training loop
    # =========================================================================
    # Setup tqdm bar
    pbar = tqdm(range(settings.N_epochs))
    # Clean data from NaN and infs
    # Replace inf with nan, then drop any row containing nan
    X    = np.where(np.isinf(X_), np.nan, X_)
    Y    = np.where(np.isinf(Y_), np.nan, Y_)
    dYdX = np.where(np.isinf(dYdX_), np.nan, dYdX_)

    # Extract bad rows
    bad_rows_X    = np.any(np.isnan(X), axis=1) if X.ndim == 2 else np.isnan(X)
    bad_rows_Y    = np.any(np.isnan(Y), axis=1) if Y.ndim == 2 else np.isnan(Y)
    bad_rows_dYdX = np.any(np.isnan(dYdX.reshape(len(dYdX), -1)), axis=1)
    bad_rows      = bad_rows_X | bad_rows_Y | bad_rows_dYdX
    # Clean bad rows
    X    = X[~bad_rows]
    Y    = Y[~bad_rows]
    dYdX = dYdX[~bad_rows]
    
    # Data scaling quantities
    X_mean = np.mean(X,axis=0); X_std = np.std(X,axis=0); X_data = (X - X_mean)/X_std
    Y_mean = np.mean(Y,axis=0); Y_std = np.std(Y,axis=0); Y_data = (Y - Y_mean)/Y_std
    # X_std: (n_inputs,)   -> (1, 1, n_inputs)
    # Y_std: (n_outputs,)  -> (1, n_outputs, 1)
    if dYdX.ndim == 1:  # X is 1D: shape (n_samples,)
        dYdX_data = dYdX * (X_std / Y_std)
    elif dYdX.ndim == 2:  # X is 2D: shape (n_samples, n_inputs)
        dYdX_data = dYdX * (X_std[None, :] / Y_std) 
        
    # Store scaling parameters in dictionary
    X_hat_params = np.vstack((X_mean,X_std))
    Y_hat_params = np.vstack((Y_mean,Y_std))
    # Initial batch
    X_train, X_test, Y_train, Y_test, dYdX_train, dYdX_test = train_test_split(
        X_data,Y_data, dYdX_data,
        test_size=settings.test_size, 
        train_size=settings.train_size, 
        random_state=settings.train_batch_seed
        )
    # No. of train and test points
    N_train  = Y_train.shape[0]
    N_test   = Y_test.shape[0]
    N_batch  = int(N_train * settings.batch_frac)
    N_sample = N_batch // 2
    # JIT loss functional (for the validation loss)
    Loss_JIT = jax.jit(Loss)
    # Train for a few epochs
    for epoch in pbar:
        # Mini-batch indices
        key = jax.random.PRNGKey(epoch)
        idx = jax.random.choice(key, N_train, shape=(N_batch,), replace=False)
        # Evaluate mini-batch of my training data
        X_batch_     = X_train[idx]
        Y_batch_     = Y_train[idx]
        dYdX_batch_  = dYdX_train[idx]
        
        # Non-uniform sampling weighted by derivative magnitude
        # Reduce to per-sample scalar by taking the norm across inputs
        if dYdX_data.ndim == 2:
            dYdX_magnitude_ = np.linalg.norm(dYdX_batch_, axis=1)  # shape: (n_samples,)
        else:
            dYdX_magnitude_ = np.abs(dYdX_batch_)  # shape: (n_samples,)
        # Power transform to emphasize large weights
        dYdX_magnitude = jnp.abs( jnp.power(dYdX_magnitude_, 1.0) )
            
        # Evaluate weights and sampling stuff based on largest gradients
        weights = dYdX_magnitude / np.sum(dYdX_magnitude)
        idx_sampled = np.random.choice(N_batch, size=N_sample, replace=False, p=weights.flatten())
        X_batch    = X_batch_[idx_sampled]
        Y_batch    = Y_batch_[idx_sampled]
        dYdX_batch = dYdX_batch_[idx_sampled]
        
        # Apply training step
        wp, opt_state, loss = Train_step(wp, opt_state, X_batch, Y_batch, dYdX_batch, settings.deriv_penalty, epoch)
        
        # # =============================================================================
        # # TODO: DEBUG  
        # # =============================================================================
        # loss_val, w_grads = jax.value_and_grad(Loss)(wp, X_batch, Y_batch, dYdX_batch, settings.deriv_penalty, epoch)
        # # Outside JIT — regular print is fine here
        # grad_norms = jax.tree_util.tree_map(lambda g: jnp.linalg.norm(g), w_grads)
        # print(f"Loss: {loss_val:.6e}")
        # print(f"Grad norms: {jax.tree_util.tree_leaves(grad_norms)}")
        # # =============================================================================
        # # TODO: DEBUG  
        # # =============================================================================
        
        # Update loss history
        loss_train.append(float(loss))
        loss_valid.append(float(Loss_JIT(wp, X_test, Y_test, dYdX_test, settings.deriv_penalty, epoch)))
        
        # Print/save training data
        if epoch % settings.N_save == 0:
            # Batch data
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_data,Y_data,
                test_size    = settings.test_size, 
                train_size   = settings.train_size, 
                random_state = settings.train_batch_seed + epoch
                )
        # Update progress bar
        pbar.set_postfix(loss=f"{loss_train[-1]:.4e}")
    # Return optimal weights, model, and loss history
    return wp, loss_train, loss_valid, X_hat_params, Y_hat_params, model

#%% Plots for benchmarking CoolProp and CoolSurrogateProp

"Plot CoolProp and CoolSurrogateProp"
def Plot_CoolSurrogateProp(Xt, Yt, Yp, 
                           dYdXt, dYdXp, 
                           Err_Y, Err_dYdX0, Err_dYdX1, 
                           X_IN, Y_OUT):
    # Number of inputs
    Nx = len(X_IN)
    # Two-inputs
    if Nx == 2:
        # Setup figure
        fig,ax = plt.subplots(nrows=3,ncols=3,tight_layout=True,figsize=(14,10))
        # Y
        a00 = ax[0,0].scatter(Xt[:,0], Xt[:,1], c=Yt)
        a01 = ax[0,1].scatter(Xt[:,0], Xt[:,1], c=Yp)
        a02 = ax[0,2].scatter(Xt[:,0], Xt[:,1], c=Err_Y)
        # dYdX0
        a10 = ax[1,0].scatter(Xt[:,0], Xt[:,1], c=dYdXt[:,0])
        a11 = ax[1,1].scatter(Xt[:,0], Xt[:,1], c=dYdXp[:,0])
        a12 = ax[1,2].scatter(Xt[:,0], Xt[:,1], c=Err_dYdX0)
        # dYdX1
        a20 = ax[2,0].scatter(Xt[:,0], Xt[:,1], c=dYdXt[:,1])
        a21 = ax[2,1].scatter(Xt[:,0], Xt[:,1], c=dYdXp[:,1])
        a22 = ax[2,2].scatter(Xt[:,0], Xt[:,1], c=Err_dYdX1)
        # Clip errors
        a02.set_clim(0,100)
        a12.set_clim(0,100)
        a22.set_clim(0,100)
        # --- Titles --- #
        # Y
        ax[0,0].set_title('%s (CoolProp)' %Y_OUT)
        ax[0,1].set_title('%s (Surrogate)' %Y_OUT)
        ax[0,2].set_title('%s (Error [%%])' %Y_OUT)
        # dYdX0
        ax[1,0].set_title('d(%s)/d(%s)|%s (CoolProp)'   %(Y_OUT,X_IN[0],X_IN[1]) )
        ax[1,1].set_title('d(%s)/d(%s)|%s (Surrogate)'  %(Y_OUT,X_IN[0],X_IN[1]) )
        ax[1,2].set_title('d(%s)/d(%s)|%s (Error [%%])' %(Y_OUT,X_IN[0],X_IN[1]) )
        # dYdX1
        ax[2,0].set_title('d(%s)/d(%s)|%s (CoolProp)'   %(Y_OUT,X_IN[1],X_IN[0]) )
        ax[2,1].set_title('d(%s)/d(%s)|%s (Surrogate)'  %(Y_OUT,X_IN[1],X_IN[0]) )
        ax[2,2].set_title('d(%s)/d(%s)|%s (Error [%%])' %(Y_OUT,X_IN[1],X_IN[0]) )
        # --- Colorbars --- #
        # Y
        fig.colorbar(a00, ax=ax[0,0])
        fig.colorbar(a01, ax=ax[0,1])
        fig.colorbar(a02, ax=ax[0,2])
        # dYdX0
        fig.colorbar(a10, ax=ax[1,0])
        fig.colorbar(a11, ax=ax[1,1])
        fig.colorbar(a12, ax=ax[1,2])
        # dYdX1
        fig.colorbar(a20, ax=ax[2,0])
        fig.colorbar(a21, ax=ax[2,1])
        fig.colorbar(a22, ax=ax[2,2])
        # --- Axes --- #
        # X1
        ax[0,0].set_ylabel(X_IN[1])
        ax[1,0].set_ylabel(X_IN[1])
        ax[2,0].set_ylabel(X_IN[1])
        # X0
        ax[2,0].set_xlabel(X_IN[0])
        ax[2,1].set_xlabel(X_IN[0])
        ax[2,2].set_xlabel(X_IN[0])
    # One-input
    elif Nx == 1:
        # Setup figure
        fig,ax = plt.subplots(nrows=2,ncols=2,tight_layout=True,figsize=(10,6))
        # Y
        ax[0,0].plot(Xt[:,0], Yt, label='%s (CoolProp)'  %Y_OUT)
        ax[0,0].plot(Xt[:,0], Yp, label='%s (Surrogate)' %Y_OUT)
        # Error-Y
        a01 = ax[0,1].plot(Xt[:,0], Err_Y, label='%s (Error [%%])'  %Y_OUT)
        # dYdX
        ax[1,0].plot(Xt[:,0], dYdXt, label='d(%s)/d(%s)|sat (CoolProp)' %(Y_OUT,X_IN))
        ax[1,0].plot(Xt[:,0], dYdXp, label='d(%s)/d(%s)|sat (Surrogate)' %(Y_OUT,X_IN))
        # Error-dYdX
        a11 = ax[1,1].plot(Xt[:,0], Err_dYdX0, label='d(%s)/d(%s)|sat (Error [%%])' %(Y_OUT,X_IN))
        # --- Legends --- #
        ax[0,0].legend(); ax[0,1].legend()
        ax[1,0].legend(); ax[1,1].legend()
        # --- Axes --- #
        # Y-labels
        ax[0,0].set_ylabel(Y_OUT)
        ax[0,1].set_ylabel('%s (Error [%%])'  %Y_OUT)
        ax[1,0].set_ylabel('d(%s)/d(%s)|sat' %(Y_OUT,X_IN))
        ax[1,1].set_ylabel('d(%s)/d(%s)|sat (Error [%%])' %(Y_OUT,X_IN))
        # X-labels
        ax[0,0].set_xlabel(X_IN); ax[0,1].set_xlabel(X_IN)
        ax[1,0].set_xlabel(X_IN); ax[1,1].set_xlabel(X_IN)
    else:
        print('Invalid number of inputs!')
        print('Nx = 2 (liquid, vapor) OR Nx = 1 (saturation line)')
    # Finish function
    return fig

"Logarithmic plot of training and validation losses"
def Plot_Loss(loss_train, loss_valid):
    # Initialize plot
    fig,ax = plt.subplots(figsize=(6,4),tight_layout=True)
    # Plot loss functionals
    ax.plot(loss_train, color='k', label='Training')
    ax.plot(loss_valid, color='b', label='validation')
    # Plot customization
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Iterations [-]'); ax.set_ylabel('Loss [-]')
    ax.minorticks_on(); ax.grid(which='both',axis='both',alpha=0.3)
    ax.legend()
    # Close function
    return fig

#%% Benchmark CoolProp and CoolSurrogate Prop

"Evaluate surrogate errors and plots"
def Benchmark_CoolSurrogateProp(model, w_opt, loss_train, loss_valid, Xt, Yt, dYdXt, X_IN, Y_OUT):
    # =========================================================================
    # Forward pass
    # =========================================================================
    # Evaluate surrogate property
    Yp = model.compute(Xt)
    # Create property partial derivatives
    dYdX_fn = jax.jit(jax.jacrev(model.compute, argnums=0)) 
    
    # =========================================================================
    # Compute errors
    # =========================================================================
    # Error on Y
    eps   = np.max((np.abs(Yt).min(), 1e-8))
    Err_Y = 100 * np.abs(Yp - Yt)/(np.mean(Yt) + eps)
    # Number of inputs
    Nx = len(X_IN)
    # Error on dYdX
    if Nx == 2:
        # Evaluate thermodynamic partial derivatives
        dYdXp = jax.vmap(dYdX_fn, in_axes=[0])(Xt)[:,0,:]
        # Epsilon
        eps_d0 = np.max((np.abs(dYdXt[:,0]).min(), 1e-8))
        eps_d1 = np.max((np.abs(dYdXt[:,1]).min(), 1e-8))
        # Gradient
        Err_dYdX0 = 100*np.abs(dYdXp[:,0] - dYdXt[:,0])/(np.mean(dYdXt[:,0]) + eps_d0)
        Err_dYdX1 = 100*np.abs(dYdXp[:,1] - dYdXt[:,1])/(np.mean(dYdXt[:,1]) + eps_d1)
    elif Nx == 1:
        # Evaluate thermodynamic partial derivatives
        dYdXp = jax.vmap(dYdX_fn, in_axes=[0])(Xt)[:,0,0]
        # Epsilon
        eps_d0 = np.max((np.abs(dYdXt).min(), 1e-8))
        eps_d1 = 0.0
        # Gradient
        Err_dYdX0 = 100*np.abs(dYdXp - dYdXt)/(np.mean(dYdXt) + eps_d0)
        Err_dYdX1 = 0.0
    else:
        print('Invalid number of inputs!')
        print('Nx = 2 (liquid, vapor) OR Nx = 1 (saturation line)')
    
    # =========================================================================
    # Plot surrogate and CoolProp evaluation
    # =========================================================================
    # Scatter plot
    fig1 = Plot_CoolSurrogateProp(
        Xt, Yt, Yp, dYdXt, dYdXp, 
        jnp.abs(Err_Y), jnp.abs(Err_dYdX0), jnp.abs(Err_dYdX1), 
        X_IN, Y_OUT
        )
    # Loss functionals
    fig2 = Plot_Loss(loss_train, loss_valid)
    # Close function
    return fig1, fig2

#%% Train EoS Y = f(X0, X1)

"Train surrogate model"
def CalibrateAndSave(
        Y, X_IN, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max = 10.0e5, # Max. pressure [Pa]
        T_max = 300.15, # Max. temperature [K]
        P_min = None, # Min. pressure [Pa]
        T_min = None,  # Min. temperature [K],
        REFPROP=False
        ):
    # Fix fluid name (in case it is part of the REFPROP database)
    if REFPROP:
        Fluid = "REFPROP::%s" %Fluid
    # Number of inputs
    Nx = len(X_IN)
    # Loop through fluid phase
    for Phase in PHASE:
        # Loop through properties
        for Y_OUT in Y:
            # Output root string
            BASE_OUT_ = '%s_%s(%s,%s)' %(Phase, Y_OUT,X_IN[0],X_IN[1]) if Nx == 2 else '%s_%s(%s)' %(Phase, Y_OUT,X_IN)
            # Replace "/" with "-" (dash)
            # Replace "|" with "_" (underscore)
            BASE_OUT = BASE_OUT_.replace('/','-').replace('|','_')
            # Divider string
            div = '-'*len(BASE_OUT)
            print(div,'\n',BASE_OUT,'\n',div)
            
            # =====================================================================
            # Generate training data
            # =====================================================================
            Xt, Yt, dYdXt = Training_Data(
                Y_OUT, Phase, Fluid, 
                P_max = P_max, T_max = T_max, P_min = P_min, T_min = T_min, 
                X_IN=X_IN, N=settings.N_data_points
                )
            
            # =====================================================================
            # Train CoolProp surrogate
            # =====================================================================
            # Train surrogate model
            w_opt, loss_train, loss_valid, X_hat_params, Y_hat_params, model = Model_Training(Xt, Yt, dYdXt, settings)
            # Assign model parameters
            model.surrogate_params(w_opt, X_hat_params, Y_hat_params)
            # Benchmark surrogate against CoolProp
            fig1, fig2 = Benchmark_CoolSurrogateProp(model, w_opt, loss_train, loss_valid, Xt, Yt, dYdXt, X_IN, Y_OUT)
            
            # =================================================================
            # Save model data
            # =================================================================
            
            # Group model data
            surrogate_params = {# Architecture
                                "n_layers": model.n_layers,
                                "hidden_dim": model.hidden_dim,
                                "activation_fns": afns_to_str(model.activation_fns), # model.activation_fns,
                                # Surrogate parameters
                                "weights": model.w_opt,
                                "x_hat_params": model.x_hat_params,
                                "y_hat_params": model.y_hat_params
                                }
            # Save plots
            fig1.savefig("%s/%s_Benchmark.png"   %(FOL_OUT_I,BASE_OUT), dpi=350)
            fig2.savefig("%s/%s_LossHistory.png" %(FOL_OUT_I,BASE_OUT), dpi=350)
            # Close plots
            plt.close("all")
            # Save data
            save_jax_model("%s/%s"  %(FOL_OUT,BASE_OUT), surrogate_params)
            
            # with open("%s/%s.pkl"  %(FOL_OUT,BASE_OUT), 'wb') as file: 
            #     dill.dump(surrogate_params, file)
    # Finish function
    return