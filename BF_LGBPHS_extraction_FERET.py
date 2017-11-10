'''
Implementation of the Bloom filter based biometric template protection for facial images. More details in:

[IS16] M. Gomez-Barrero, C. Rathgeb, J. Galbally, C. Busch, J. Fierrez, "Unlinkable and Irreversible Biometric Template
Protection based on Bloom Filters", in Elsevier Information Sciences, vol. 370-371, pp. 18-32, 2016

[IF18] M. Gomez-Barrero, C. Rathgeb, G. Li, R. Raghavendra, J. Galbally and C. Busch
        "Multi-Biometric Template Protection Based on Bloom Filters", in Information Fusion, vol. 42, pp. 37-50, 2018.

Please remember to reference articles [IS16] and [IF18] on any work made public, whatever the form,
based directly or indirectly on these metrics.
'''

__author__ = "Marta Gomez-Barrero"
__copyright__ = "Copyright (C) 2017 Hochschule Darmstadt"
__license__ = "License Agreement provided by Hochschule Darmstadt (https://share.nbl.nislab.no/g03-06-btp/face-bf-btp/blob/master/hda-license.pdf)"
__version__ = "1.0"


import bob.io.base
import bob.bio.face
import numpy
import math
from PIL import Image
import os
import argparse


######################################################################
### Parameter and arguments definition

parser = argparse.ArgumentParser(description='Extract unprotected LGBPHS and protected Bloom filter templates from the FERET DB.')

# location of source images, final templates and intermediate steps (the latter for debugging purposes)
parser.add_argument('DBdir', help='directory where the compressed face DB is stored', type=str)
parser.add_argument('--DBdir_png', help='directory where the uncompressed face DB will be stored', type=str, nargs='?', default = './FERET_DB_png/')
parser.add_argument('--DBtemplates', help='directory where the unprotected face templates will be stored', type=str, nargs='?', default = './FERET_LGBPHSTemplates/')
parser.add_argument('--DB_BFtemplates', help='directory where the protected BF face templates will be stored', type=str, nargs='?', default = './FERET_BFTemplates_full/')
parser.add_argument('--grayDB', help='directory where the intermediate face gray images will be stored (for debug)', type=str, nargs='?', default = './FERET_DB_cropped/')
parser.add_argument('--tanDB', help='directory where the intermediate face Tan-Triggs processed images will be stored (for debug)', type=str, nargs='?', default = './FERET_DB_TanTriggs/')

args = parser.parse_args()
DBdir = args.DBdir
DBdir_png = args.DBdir_png
DBtemplates = args.DBtemplates
DB_BFtemplates = args.DB_BFtemplates
grayDB = args.grayDB
tanDB = args.tanDB

if not os.path.exists(DBdir_png):
    os.mkdir(DBdir_png)
if not os.path.exists(DBtemplates):
    os.mkdir(DBtemplates)
if not os.path.exists(DB_BFtemplates):
    os.mkdir(DB_BFtemplates)
if not os.path.exists(grayDB):
    os.mkdir(grayDB)
if not os.path.exists(tanDB):
    os.mkdir(tanDB)


# image resolution of the preprocessed images
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = 64

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT / 4, CROPPED_IMAGE_WIDTH / 4 - 1)
LEFT_EYE_POS  = (CROPPED_IMAGE_HEIGHT / 4, CROPPED_IMAGE_WIDTH / 4 * 3)

# Parameters of LGBPHS and Bloom filter extraction
N_BLOCKS = 80 # number of blocks the facial image is divided into, also for LGBPHS algorihtm

N_HIST = 40  # parameters fixed by LGBPHS
N_BINS = 59

THRESHOLD = 0  # binarization threshold for LGBPHS features

N_BITS_BF = 4  # parameters for BF extraction
N_WORDS_BF = 20
BF_SIZE = int(math.pow(2, N_BITS_BF))
N_BF_Y = N_HIST//N_BITS_BF
N_BF_X = (N_BINS+1)//N_WORDS_BF

# define facial LGBPHS feature extractor using bob face library
feature_extractor = bob.bio.face.extractor.LGBPHS(
    # block setup
    block_size = 8,
    block_overlap = 0,
    # Gabor parameters
    gabor_sigma = math.sqrt(2.) * math.pi,
    # LBP setup (we use the defaults)
    # histogram setup
    sparse_histogram = False,
    split_histogram = 'blocks'
)


####################################################################
### Some auxiliary functions

def extract_LGBPHS_features(filename):
    '''Extracts unprotected template from image file'''

    im = Image.open(DBdir + filename + '.ppm')
    im.save(DBdir_png + filename + '.png')
    image = bob.io.base.load(DBdir_png + filename + '.png')

    if image.ndim == 3:
        gray_image = bob.ip.color.rgb_to_gray(image)

    face_detector = bob.bio.face.preprocessor.FaceDetect(
        face_cropper='face-crop-eyes',
        use_flandmark=True,
        cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
        cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
    )

    tan_triggs_offset_preprocessor = bob.bio.face.preprocessor.TanTriggs(
        face_cropper = face_detector,
    )

    image2 = face_detector(gray_image)
    bob.io.base.save(image2.astype('uint8'), grayDB + filename + '_cropped.png')
    gray_image = tan_triggs_offset_preprocessor(gray_image)
    bob.io.base.save(gray_image.astype('uint8'), tanDB + filename + '_TanTriggs.png')
    return feature_extractor(gray_image)


def extract_BFs_from_LGBPHS_features(feat):
    '''Extracts BF protected template from an unprotected template'''
    template = numpy.zeros(shape=[N_BLOCKS * N_BF_X * N_BF_Y, BF_SIZE], dtype=int)

    index = 0
    for i in range(N_BLOCKS):
        block = feat[i, :]
        block = numpy.reshape(block, [N_HIST, N_BINS + 1])  # add column of 0s -> now done on features!

        block = (block > THRESHOLD).astype(int)

        for x in range(N_BF_X):
            for y in range(N_BF_Y):
                bf = numpy.zeros(shape=[BF_SIZE])

                ini_x = x * N_WORDS_BF
                fin_x = (x + 1) * N_WORDS_BF
                ini_y = y * N_BITS_BF
                fin_y = (y + 1) * N_BITS_BF
                new_hists = block[ini_y: fin_y, ini_x: fin_x]

                for k in range(N_WORDS_BF):
                    hist = new_hists[:, k]
                    location = int('0b' + ''.join([str(a) for a in hist]), 2)
                    bf[location] = int(1)

                template[index] = bf
                index += 1

    return template


####################################################################
### Template extraction

# Define permutation key to provide unlinkability
key4 = numpy.zeros(shape=[N_BF_Y, N_BF_X, N_BITS_BF * N_BLOCKS], dtype=int)
for j in range(N_BF_Y):
    for k1 in range(N_BF_X):
        key = numpy.random.permutation(N_BITS_BF * N_BLOCKS)
        key4[j, k1, :] = key

print("Extracting LGBPHS features and BF templates for the DB...")

for filename in os.listdir(DBdir):
    if ('fa' in filename) or ('fb' in filename): # extract templates only for the fa and fb partitions
        print(filename[0:-4])

        features = extract_LGBPHS_features(filename[0:-4])
        numpy.savetxt(DBtemplates + filename[0:-4] + '.txt', features, fmt='%d')

        features = numpy.reshape(features, newshape=[N_BLOCKS, N_HIST, N_BINS])
        feat = numpy.zeros([N_BLOCKS, N_HIST, N_BINS + 1]) # to add a 0 at the end and round the N_BINS to 60
        feat[:, :, 0:N_BINS] = features

        # permute features to provide unlinkability
        features2 = numpy.zeros([N_BLOCKS, N_HIST, N_BINS + 1])
        for j in range(N_BF_Y):
            for k1 in range(N_BF_X):
                permKey = key4[j, k1, :]
                aux = feat[:, j * N_BITS_BF: (j + 1) * N_BITS_BF, k1 * N_WORDS_BF: (k1 + 1) * N_WORDS_BF]
                aux = numpy.reshape(aux, [N_BLOCKS * N_BITS_BF, N_WORDS_BF])
                aux = aux[permKey, :]
                aux = numpy.reshape(aux, [N_BLOCKS, N_BITS_BF, N_WORDS_BF])
                features2[:, j * N_BITS_BF: (j + 1) * N_BITS_BF, k1 * N_WORDS_BF: (k1 + 1) * N_WORDS_BF] = aux

        # extract BFs
        bfs = extract_BFs_from_LGBPHS_features(features2)
        numpy.savetxt(DB_BFtemplates + filename[0:-4] + '.txt', bfs, fmt='%d')
