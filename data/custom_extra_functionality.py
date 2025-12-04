# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


def custom_extra_functionality(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints.
    """
    import pandas as pd
    
    gen_p_nom = pd.read_csv('D:/Ablage/mast/PythonSkripts/EntsoeTransparancyData/flh_generation_DE_2024.csv',index_col=0,sep=';')
    flh = gen_p_nom.rename(columns={'real_market_flh':'flh'})
    flh_carrier = ['biomass','lignite','coal']
    
    m = n.model

    p = m.variables["Generator-p"]
    dt = n.snapshot_weightings["generators"]
    
    # Filter: nur Generatoren deren Carrier in flh_carrier stehen
    gens = n.generators.index[
        n.generators.carrier.isin(flh_carrier)
    ]
    # Für jeden Carrier ein globales Limit setzen
    for g in gens:
        carrier = n.generators.at[g, "carrier"]
        p_nom = n.generators.at[g, "p_nom"]

        # Limit für genau diesen Generator
        limit = flh.at[carrier,'flh'] * p_nom   # MWh

        # Summe über alle Zeitpunkte
        expr = sum(p[g, t] * dt[t] for t in snapshots)

        # Constraint hinzufügen
        m.add_constraints(
            expr <= limit,
            name=f"limit_flh_{g}"
        )
