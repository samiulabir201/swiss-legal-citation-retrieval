# Gold parent-link check

A gold citation is counted as having no own text when it does not appear as a `citation` row in either `laws_de.csv` or `court_considerations.csv`.

## Summary

| Split | No-own-text gold mentions | Same-query gold parent found | Has parents, none in same-query gold | No parent edge found |
|---|---:|---:|---:|---:|
| train | 1,340 | 73 | 1,105 | 162 |
| val | 0 | 0 | 0 | 0 |

## Exclusive Unique Counts

| Split | No-own-text unique gold | Has parent edges | Ever has same-query gold parent | Never has same-query gold parent | No parent edge found |
|---|---:|---:|---:|---:|---:|
| train | 767 | 650 | 56 | 594 | 117 |
| val | 0 | 0 | 0 | 0 | 0 |
| combined | 767 | 650 | 56 | 594 | 117 |

## Examples Where Parent Is Not In Same Query Gold

- `train:train_0002` `Art. 975 ZGB` parents include: `1C_664/2024 E. 4.2.2`, `5A.6/2005 17.03.2005 E. 3`, `5A_1007/2020 E. 2.3.1`, `5A_134/2020 E. 1`, `5A_194/2013 E. 3.1`
- `train:train_0002` `Art. 973 ZGB` parents include: `1C_151/2010 21.06.2010 E. 2`, `1C_340/2016 E. 3.1`, `1C_585/2015 E. 2`, `2C_49/2014 E. 4.3`, `4A_393/2018 E. 2.4.4`
- `train:train_0002` `Art. 956a ZGB` parents include: `5A_1024/2020 E. 1`, `5A_12/2021 E. 1`, `5A_12/2021 E. 2`, `5A_237/2018 E. 2.2`, `5A_237/2018 E. 2.3`
- `train:train_0002` `Art. 976a ZGB` parents include: `5A_726/2021 E. 3.2`, `Art. 69 Abs. 4 GBV`
- `train:train_0002` `Art. 60 OR` parents include: `1C_315/2018 10.04.2019 E. 2.3`, `1C_353/2007 30.10.2008 E. 7.3`, `2A.29/2000 12.05.2000 E. 3`, `2A.373/1998 21.01.2000 E. 3`, `2A.553/2002 22.08.2003 E. 4`
- `train:train_0003` `Art. 264m StGB` parents include: `1B_1/2023 E. 3.8`, `1B_271/2017 E. 6.5`, `1B_417/2017 E. 7`, `1B_465/2018 E. 3.5`, `6B_466/2015 E. 1.4.1`
- `train:train_0005` `Art. 50 BV` parents include: `1C_100/2020 E. 1.3`, `1C_101/2007 26.02.2008 E. 1`, `1C_11/2008 25.09.2008 E. 1`, `1C_11/2008 25.09.2008 E. C`, `1C_119/2023 E. 1.1`
- `train:train_0007` `Art. 630 ZGB` parents include: `5A_145/2013 E. 4`, `5A_180/2022 E. 3.2`, `5A_180/2022 E. 3.4.2`, `5A_326/2016 E. 4.2.2`, `5A_587/2010 11.02.2011 E. 4`
- `train:train_0009` `Art. 337 OR` parents include: `1C_245/2008 02.03.2009 E. 5`, `1C_277/2007 30.06.2008 E. 6`, `1C_42/2007 29.11.2007 E. 3.5`, `1C_514/2023 E. 5.1`, `1C_514/2023 E. 7.1`
- `train:train_0009` `Art. 361 OR` parents include: `4A_112/2023 E. 4.1`, `4A_183/2012 11.09.2012 E. 4`, `4A_194/2013 E. 4.3`, `4A_196/2013 E. 4.3`, `4A_198/2013 E. 4.3`

## Examples With No Parent Edge Found

- `train:train_0001` `Art. 10a Abs. 1 UVG`
- `train:train_0012` `Art. 156 FinfraG`
- `train:train_0043` `Art. 56 Abs. 1 DBG`
- `train:train_0044` `Art. 19 FINIV`
- `train:train_0044` `Art. 5 Abs. 1 FIDLEG`
- `train:train_0044` `Art. 17 FIDLEV`
- `train:train_0044` `Art. 27 FIDLEV`
- `train:train_0060` `Art. 1 LugÜ`
- `train:train_0060` `Art. 2 Abs. 1 LugÜ`
- `train:train_0068` `Art. 647 Abs. 1 ZG`
