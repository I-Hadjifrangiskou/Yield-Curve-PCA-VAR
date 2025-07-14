# Yield Curve Factor Modeling using PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg

# Load the yield data from csv
# Columns: Date, 1M, 3M, 6M, 1Y, 2Y, 5Y, 7Y, 10Y, 20Y, 30Y (Bond maturities)
# Date format: YYYY-MM-DD
df = pd.read_csv('yields.csv', parse_dates = ['Date']) 
df.set_index('Date', inplace = True)

# Drop rows with missing values
df = df.dropna()


# Standardize the yields across maturities (each column) to have zero mean and unit variance
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

# Perform PCA 
pca   = PCA() # Finds orthognal components that capture the largest variance in the yield curve 
X_pca = pca.fit_transform(X_scaled) # PCA scores

# Create DataFrame of principal components (scores)
df_pca = pd.DataFrame(X_pca, index = df.index, 
                      columns = [f'PC{i+1}' for i in range(X_pca.shape[1])])

# Analyze the explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize = (10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker = 'o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


# Loadings (eigenvectors) tell us how each original feature (maturity) contributes
loadings = pd.DataFrame(pca.components_.T, 
                        index = df.columns, 
                        columns = [f'PC{i+1}' for i in range(X_pca.shape[1])])

plt.figure(figsize = (12, 6))

# Plot the loadings (or weights) of each of the PCs. More precisely, Loading_{i,j} = contribution of maturity j to PC_{i}
for i in range(3):  # Plot first 3 PCs
    plt.plot(df.columns, loadings.iloc[:, i], marker = 'o', label = f'PC{i+1}')
plt.title('PCA Loadings')
plt.xlabel('Maturity')
plt.ylabel('Loading Value')
plt.legend()
plt.show()

# Plot time series of PCs
df_pca[['PC1', 'PC2', 'PC3']].plot(figsize = (12, 6))
plt.title('Time Series of First 3 Principal Components')
plt.ylabel('PCA Score')
plt.xlabel('Date')
plt.show()

# Reconstruct approximate yield curve using first k PCs
k = 3
X_reduced        = np.dot(X_pca[:, :k], pca.components_[:k, :])
X_reconstructed  = scaler.inverse_transform(X_reduced)
df_reconstructed = pd.DataFrame(X_reconstructed, index = df.index, columns = df.columns)

# Plot actual vs reconstructed on a sample date
sample_date = df.index[100]
plt.figure(figsize = (10, 5))
plt.plot(df.columns, df.loc[sample_date], marker = 'o', label = 'Original')
plt.plot(df.columns, df_reconstructed.loc[sample_date], marker = 'x', label = f'Reconstructed (PC1-{k})')
plt.title(f'Yield Curve Reconstruction on {sample_date.date()}')
plt.xlabel('Maturity')
plt.ylabel('Yield')
plt.legend()
plt.show()

# Compute relative reconstruction error for each date (row-wise)
errors = np.linalg.norm(df.values - df_reconstructed.values, axis = 1) / np.linalg.norm(df.values, axis = 1)

# Create a Series with Date index
error_series = pd.Series(errors, index = df.index)

# Plot the reconstruction error over time
plt.figure(figsize = (10, 5))
plt.plot(error_series, label = 'Relative Reconstruction Error', color = 'darkred')
plt.title('Relative Reconstruction Error over Time (Using PC1–3)')
plt.xlabel('Date')
plt.ylabel('Relative Error')
plt.axhline(0.01, color = 'gray', linestyle = '--', label = '1% threshold')
plt.legend()
plt.show()


pcs_used = ['PC1', 'PC2', 'PC3']   # PCs that will be used
delta_pcs = df_pca.diff().dropna() # Daily changes in PC scores, ΔPC_t = PC_t - PC_{t-1}

# Fit AR(1) models to PC changes, where t is measured daily
# ΔPC_t = μ + φ * ΔPC_{t-1} + ε_t, where μ is the intercept, φ the AR(1) coefficient and ε_t the random noise, taken to be normally distributed with mean 0, and standard deviation equal to the std of residuals
models = {}
for pc in pcs_used:
    models[pc] = AutoReg(delta_pcs[pc], lags = 1).fit()

# Simulate 1-day-ahead PC changes, we will do this in 2 cases using an AR(1) model: 1) Include a drift term to capture yield rates and 2) Remove the drift term to only consider shocks
n_sims = 10000 # Number of Monte Carlo simulations
simulated_delta_pcs = np.zeros((n_sims, 3)) # ΔPC_sim with drift term μ 
simulated_delta_pcs_mu0 = np.zeros((n_sims, 3)) # ΔPC_sim without drift term μ 

for i, pc in enumerate(pcs_used):

    # Model and parameters
    model    = models[pc]
    last_val = delta_pcs[pc].iloc[-1]
    mu       = model.params[0]
    phi      = model.params[1]
    sigma    = model.resid.std()
    
    # Get simulated ΔPC_{i}
    simulated_delta_pcs[:, i]     = mu + phi * last_val + np.random.normal(0, sigma, n_sims)
    simulated_delta_pcs_mu0[:, i] = phi * last_val + np.random.normal(0, sigma, n_sims)

# Add simulated deltas to last observed PC values, this gives n_sims simulated 1-day-ahead PC vectors
last_pcs = df_pca[pcs_used].iloc[-1].values
simulated_pcs = simulated_delta_pcs + last_pcs
simulated_pcs_mu0 = simulated_delta_pcs_mu0 + last_pcs

# Reconstruct simulated yield curves
components = pca.components_[:3, :]
X_simulated_std = np.dot(simulated_pcs, components) # The yield curve X_sim = PC_sim . loadings
X_simulated_std_mu0 = np.dot(simulated_pcs_mu0, components)

# Add to last standardized yield curve
last_curve_std = X_scaled[-1] # Get last element of the standardized yield curves
X_simulated_full_std = X_simulated_std + last_curve_std # Add the next day yield
X_simulated_full_std_mu0 = X_simulated_std_mu0 + last_curve_std


# Inverse transform to true yield space
X_simulated_curves = scaler.inverse_transform(X_simulated_full_std)
X_simulated_curves_mu0 = scaler.inverse_transform(X_simulated_full_std_mu0)

# Assume a single 5Y zero-coupon bond, P = 1/(1 + y)^T
# ΔP \approx - D . P . Δy for small Δy, we take P = 1
# 5Y bond duration = 5 years
duration = 5  # in years
idx_5Y = df.columns.get_loc("5Y")
yield_diff = X_simulated_curves[:, idx_5Y] - df.iloc[-1, idx_5Y]
price_changes = - duration * yield_diff / 100  # divide by 100 for percentage yields
yield_diff_mu0 = X_simulated_curves_mu0[:, idx_5Y] - df.iloc[-1, idx_5Y]
price_changes_mu0 = - duration * yield_diff_mu0 / 100

# Compute 1-day VaR 5th and 1st percentiles (95%, 99% confidence)
VaR_95 = np.percentile(price_changes, 5)
VaR_99 = np.percentile(price_changes, 1)
VaR_95_mu0 = np.percentile(price_changes_mu0, 5)
VaR_99_mu0 = np.percentile(price_changes_mu0, 1)


# Plot histogram of simulated losses (losses are expected because yields have been rising in the AR(1) model)
plt.figure(figsize = (10, 5))
plt.hist(price_changes, bins = 50, color = 'steelblue', edgecolor = 'black')
plt.axvline(VaR_95, color = 'orange', linestyle = '--', label = '95% VaR')
plt.axvline(VaR_99, color = 'red', linestyle = '--', label = '99% VaR')
plt.title('Simulated Portfolio Loss Distribution (1-day Horizon)')
plt.xlabel('Change in Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot histogram of simulated losses/gains (can have both in this case because we neglected drift term)
plt.figure(figsize = (10, 5))
plt.hist(price_changes_mu0, bins = 50, color = 'steelblue', edgecolor = 'black')
plt.axvline(VaR_95_mu0, color = 'orange', linestyle = '--', label = '95% VaR (AR μ=0)')
plt.axvline(VaR_99_mu0, color = 'red', linestyle = '--', label = '99% VaR (AR μ=0)')
plt.title('Simulated Portfolio Loss Distribution (1-day Horizon) w/ μ = 0')
plt.xlabel('Change in Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()