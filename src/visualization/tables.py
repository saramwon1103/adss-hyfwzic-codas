def winner_names(result):
    """List every competition-rank-1 alternative; never invent a unique tie-break."""
    table=result.ranking
    return table.loc[table['rank']==1,'alternative'].tolist()
