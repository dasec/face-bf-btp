'''
Score computation (unprotected and protected domain) using the fa and fb partitions of the FERET databases.
More details on the Bloom filter based BTP scheme in:

[IS16] M. Gomez-Barrero, C. Rathgeb, J. Galbally, C. Busch, J. Fierrez, "Unlinkable and Irreversible Biometric Template
Protection based on Bloom Filters", in Elsevier Information Sciences, vol. 370-371, pp. 18-32, 2016

[IF18] M. Gomez-Barrero, C. Rathgeb, G. Li, R. Raghavendra, J. Galbally and C. Busch
        "Multi-Biometric Template Protection Based on Bloom Filters", in Information Fusion, vol. 42, pp. 37-50, 2018.

Please remember to reference articles [IS16] and [IF18] on any work made public, whatever the form,
based directly or indirectly on these metrics.
'''

__author__ = "Marta Gomez-Barrero"
__copyright__ = "Copyright (C) 2017 Hochschule Darmstadt"
__license__ = "License Agreement provided by Hochschule Darmstadt (https://github.com/dasec/face-bf-btp/blob/master/hda-license.pdf)"
__version__ = "1.0"

import bob.bio.face
import numpy
import os
import argparse

######################################################################
### Parameter and arguments definition

# location of source templates and score files
parser = argparse.ArgumentParser(description='Compute unprotected LGBPHS and protected Bloom filter scores from the FERET DB.')

parser.add_argument('DBtemplates', help='directory where the unprotected LGBPHS templates are stored', type=str)
parser.add_argument('DB_BFtemplates', help='directory where the protected BF templates are stored', type=str)
parser.add_argument('partitionsDir', help='directory where the FERET partition files are stored', type=str)
parser.add_argument('--scoresDir', help='directory where unprotected and protected scores will be stored', type=str, nargs='?', default = './scores/')

args = parser.parse_args()
DBtemplates = args.DBtemplates
DB_BFtemplates = args.DB_BFtemplates
scoresDir = args.scoresDir
partitionsDir = args.partitionsDir

if not os.path.exists(scoresDir):
    os.mkdir(scoresDir)

####################################################################
### Some auxiliary functions

def hamming_distance(X, Y):
    '''Computes the noralised Hamming distance between two Bloom filter templates'''
    dist = 0

    N_BF = X.shape[0]
    for i in range(N_BF):
        A = X[i, :]
        B = Y[i, :]

        suma = sum(A) + sum(B)
        if suma > 0:
            dist += float(sum(A ^ B)) / float(suma)

    return dist / float(N_BF)

####################################################################
### Score computation

# load protocol files from FERET
text_file = open(partitionsDir + "fa.txt", "r")
references = text_file.read().split('\n')[0:994]

text_file = open(partitionsDir + "fb.txt", "r")
probe_fb = text_file.read().split('\n')[0:992]

text_file = open(partitionsDir + "dup1.txt", "r")
probe_dup1 = text_file.read().split('\n')[0:736]

# pre-allocate score arrays
genScoresBF_fb = []
genScoresBF_dup1 = []
impScoresBF_fb = []

genScores_fb = []
genScores_dup1 = []
impScores_fb = []

# compute scores for each reference template and save at each iteration
for ref in references:
    r = ref.split()

    print(r[0])
    
    filename = r[1]
    aBF = numpy.loadtxt(DB_BFtemplates + filename[0:-4] + '.txt').astype(int)
    a = numpy.loadtxt(DBtemplates + filename[0:-4] + '.txt').astype(int)

    for prob_fb in probe_fb:
        p = prob_fb.split(' ')
        filename = p[1]
        if r[0] in prob_fb:
            bBF = numpy.loadtxt(DB_BFtemplates + filename[0:-4] + '.txt').astype(int)
            genScoresBF_fb.append(hamming_distance(aBF,bBF))

            b = numpy.loadtxt(DBtemplates + filename[0:-4] + '.txt').astype(int)
            genScores_fb.append(bob.math.chi_square(a.flatten(), b.flatten()))

        elif p[0] > r[0]:
            bBF = numpy.loadtxt(DB_BFtemplates + filename[0:-4] + '.txt').astype(int)
            impScoresBF_fb.append(hamming_distance(aBF, bBF))

            b = numpy.loadtxt(DBtemplates + filename[0:-4] + '.txt').astype(int)
            impScores_fb.append(bob.math.chi_square(a.flatten(), b.flatten()))

    for prob_dup1 in probe_dup1:
        p = prob_dup1.split(' ')
        if r[0] in prob_dup1:
            filename = prob_dup1.split(' ')[1]
            bBF = numpy.loadtxt(DB_BFtemplates + filename[0:-4] + '.txt').astype(int)
            genScoresBF_dup1.append(hamming_distance(aBF, bBF))

            b = numpy.loadtxt(DBtemplates + filename[0:-4] + '.txt').astype(int)
            genScores_dup1.append(bob.math.chi_square(a.flatten(), b.flatten()))

    # update score file after each reference is compared with all probes
    numpy.savetxt(scoresDir+"genScores_fb.txt", genScores_fb)
    numpy.savetxt(scoresDir+"genScores_dup1.txt", genScores_dup1)
    numpy.savetxt(scoresDir+"impScores_fb.txt", impScores_fb)
    numpy.savetxt(scoresDir+"genScoresBF_fb.txt", genScoresBF_fb)
    numpy.savetxt(scoresDir+"genScoresBF_dup1.txt", genScoresBF_dup1)
    numpy.savetxt(scoresDir+"impScoresBF_fb.txt", impScoresBF_fb)
