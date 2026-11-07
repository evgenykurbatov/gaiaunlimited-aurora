#%%

import sqlutilpy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import config
from domain import Domain

# %%
# Load domain
#

print("\n* Load domain")

dom = Domain().load(config.data_path + f"/domain_hpx{config.hpxorder}.npz")

# %%
# Query a 2D histogram on a HEALPix vs G plane
#
# Manual to the built-in functions:
# https://neon.tech/postgresql/postgresql-math-functions/postgresql-width_bucket
#

# Use HEALPix order 7
hpxorder = 7
hpxbase = 2**35 * 4**(12 - hpxorder)
nside = 2**hpxorder
npix = 12 * 4**hpxorder
print(f"hpxorder={hpxorder} -> (nside={nside}, npix={npix})")

query = \
f"SELECT \
  source_id/{hpxbase} AS hpx7, \
  g_bin, \
  COUNT(*) AS n, \
  SUM(selection) AS k, \
  PERCENTILE_CONT(0.5) WITHIN GROUP ( ORDER BY parallax_error ) AS parallax_error, \
  PERCENTILE_CONT(0.5) WITHIN GROUP ( ORDER BY visibility_periods_used ) AS visibility_periods_used \
  FROM ( SELECT source_id, parallax_error, visibility_periods_used, phot_g_n_obs, astrometric_n_good_obs_al, \
      WIDTH_BUCKET(phot_g_mean_mag,{dom.G_edges[0]},{dom.G_edges[-1]},{dom.G.size}) AS g_bin, \
      (CASE WHEN has_xp_continuous THEN 1 ELSE 0 END) AS selection \
      FROM gaia_dr3.gaia_source ) AS subquery \
GROUP BY hpx7, g_bin \
"

res = sqlutilpy.get(query, asDict=True, \
                    db='wsdb', host=config.wsdb_host, user=config.wsdb_user, password=config.wsdb_password)
df = pd.DataFrame.from_dict(res)
del(res)
print("Columns:", df.columns)
print("  records:", len(df))
print("  memory: %.1f MiB" % (df.memory_usage().sum()/(1024**2)))

# %%

print()
print("Stars in GS:  ", df['n'].sum())
print("Stars with XP:", df['k'].sum())

# %%

mask = (df['g_bin'] <= 0) | (df['g_bin'] >= len(dom.G)+1)
df = df[~mask].copy()
df['g_bin'] -= 1

# %%

print("min and max G bins:", (dom.G.min(), dom.G.max()))
print("min and max occupied G bins:", (dom.G[df['g_bin']].min(), dom.G[df['g_bin']].max()))


plt.figure()

hist, edges = np.histogram(dom.G[df['g_bin']], weights=df['k'], bins=dom.G_edges, density=True)
N = df['k'].sum()
print(N)

plt.stairs(N*hist, edges)
plt.xlabel("G [mag]")
plt.yscale('log')

plt.show()
plt.close()

# %%
# Save
#

print("\n* Save")

df.to_hdf(config.data_path + '/stats.hdf', key='df', format='table', data_columns=True, mode='w', index=False)
