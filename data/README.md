## networkrepository.com.csv
Contains the analysis results of all analyzed graphs which is the output of the
`networkrepository_run_analysis.py` script.

## networkrepository.com_nodup.csv
Contains the results after manually removing duplicates. Duplicates were identified
by sorting the data using discriminating information like the number of orbits +
the numer of generators. On equality of these two values, the remaining values
(espacially density, numer of ndoes and edges) could be compared.
This allowed to remove duplicates that had slightly differing names.
The `networkrepository_clean_dups.py` script achieves this.

## networkrepository.com_simplified.csv
As above, but simplified versions of non-simple graphs were used.

## networkrepository.com_simplified_nodup.csv
As above, but simplified versions of non-simple graphs were used.

## stats.csv
Statistics of all graphs on networkrepository.com. It is the result of
`networkrepository_download_stats.py`.

