# BTP based on Bloom filters for facial images

Biometric Template Protection based on Bloom filters for facial images. Scripts designed for the FERET DB 
(<https://www.nist.gov/programs-projects/face-recognition-technology-feret>), following the protocol included in [[IF17]](http://www.sciencedirect.com/science/article/pii/S1566253516301233).

## License
This work is licensed under license agreement provided by Hochschule Darmstadt ([h_da-License](/hda-license.pdf)).

## Instructions

### Dependencies
* bob.io.base
* bob.bio.face
* numpy
* scipy.op
* PIL
* math
* argparser
* os

### Usage

1. Install bob packages (<https://www.idiap.ch/software/bob/docs/bob/bob/stable/install.html>)
2. Run BF_LGBPHS_extraction_FERET.py to extract both unprotected and protected templates from the 'fa', 'fb' and 'dup1' partitions of FERET

    ```python
	usage: BF_LGBPHS_extraction_FERET.py [-h] [--DBdir_png [DBDIR_PNG]]
                                         [--DBtemplates [DBTEMPLATES]]
                                         [--DB_BFtemplates [DB_BFTEMPLATES]]
                                         [--grayDB [GRAYDB]] [--tanDB [TANDB]]
                                         DBdir
    
    Extract unprotected LGBPHS and protected Bloom filter templates from the FERET
    DB.
    
    positional arguments:
      DBdir                 directory where the compressed face DB is stored
    
    optional arguments:
      -h, --help            show this help message and exit
      --DBdir_png [DBDIR_PNG]
                            directory where the uncompressed face DB will be
                            stored
      --DBtemplates [DBTEMPLATES]
                            directory where the unprotected face templates will be
                            stored
      --DB_BFtemplates [DB_BFTEMPLATES]
                            directory where the protected BF face templates will
                            be stored
      --grayDB [GRAYDB]     directory where the intermediate face gray images will
                            be stored (for debug)
      --tanDB [TANDB]       directory where the intermediate face Tan-Triggs
                            processed images will be stored (for debug)
    ```
	1. Input: folders containing the FERET DB and where to store the templates as well as some intermediate information 
(arguments can be modified at the top of the script to use other folders)
	2. Output: extracted templates and intermediate results (for debugging purposes)
	3. Other parameters for the LGBPHS and Bloom filter template extraction might be changed at the top of the script. The values used in [[IF18]] are included as default.
3. Run computeScores.py to compute the mated and non-mated scores

    ```python
    usage: computeScores.py [-h] [--scoresDir [SCORESDIR]]
                            DBtemplates DB_BFtemplates
    
    Compute unprotected LGBPHS and protected Bloom filter scores from the FERET
    DB.
    
    positional arguments:
      DBtemplates           directory where the unprotected LGBPHS templates are
                            stored
      DB_BFtemplates        directory where the protected BF templates are stored
      partitionsDir         directory where the FERET partition files are stored
    
    optional arguments:
      -h, --help            show this help message and exit
      --scoresDir [SCORESDIR]
                            directory where unprotected and protected scores will
                            be stored
    ```
	1. Input: folders with the unprotected and protected templates (arguments can be modified at the top of the script to use other folders), as well as the folder where the scores will be stored
	2. Output: mated and non-mated scores, stored in text files with one score per row

## References

More details in:

- [[IS16]](http://www.sciencedirect.com/science/article/pii/S0020025516304753) M. Gomez-Barrero, C. Rathgeb, J. Galbally, C. Busch, J. Fierrez, "Unlinkable and Irreversible Biometric Template
Protection based on Bloom Filters", in Elsevier Information Sciences, vol. 370-371, pp. 18-32, 2016

- [[IF18]](http://www.sciencedirect.com/science/article/pii/S1566253516301233) M. Gomez-Barrero, C. Rathgeb, G. Li, R. Raghavendra, J. Galbally and C. Busch, "Multi-Biometric Template Protection 
Based on Bloom Filters", in Information Fusion, vol. 42, pp. 37-50, 2018.

Please remember to reference articles [IS16] and [IF18] on any work made public, whatever the form,
based directly or indirectly on these scripts.
