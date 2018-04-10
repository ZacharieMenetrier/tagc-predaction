#! /bin/bash

# The local root folder of the predgetic project
LOCAL_ROOT=~/Desktop/tagc-predaction

# The remote root folder of the predgetic project
REMOTE_ROOT=predaction

# The user name on the mesocenter
USER=zmenetrier

# Path to singularity image
SINGULARITY_IMAGE=/scratch/zmenetrier/images/keras.img

# Name of the script launching the python script
CALL_PYTHON_FILE=call_python.sh

# Python command to launch
PYTHON_COMMAND='python3 src/fr/tagc/predaction/predaction.py > test.txt'

# Get the date for folder naming
DATE=`date '+%Y-%m-%d-%H-%M-%S'`

# Build file name and folder name with variable values
OAR_FILE=OAR_$DATE
JOB_NAME=JOB_$DATE
REMOTE_PROJECT_FOLDER=/scratch/$USER/$REMOTE_ROOT/$DATE
REMOTE_SCRIPT_FOLDER=$REMOTE_PROJECT_FOLDER/Script

# Create the OAR file
echo '#!/bin/bash' > "${OAR_FILE}"
echo '#OAR -n '$JOB_NAME >> "${OAR_FILE}"
echo '#OAR -O '$JOB_NAME'.%jobid%.out' >> "${OAR_FILE}"
echo '#OAR -E '$JOB_NAME'.%jobid%.err' >> "${OAR_FILE}"
echo '#OAR -p singularity="YES"' >> "${OAR_FILE}"
# En mode CPU
#echo '#OAR -p smp AND nodetype like "SMP512Gb"' >> "${OAR_FILE}"
#echo '#OAR -l nodes=1/core=4,walltime=30:00:00' >> "${OAR_FILE}"
# En mode GPU
echo '#OAR -l node=1/gpunum=1,walltime=00:10:00' >> "${OAR_FILE}"
echo '#OAR -p gpu' >> "${OAR_FILE}"

echo 'singularity exec '$SINGULARITY_IMAGE $REMOTE_PROJECT_FOLDER/$CALL_PYTHON_FILE >> "${OAR_FILE}"

# Create the python call file
echo '#!/bin/bash' > $CALL_PYTHON_FILE
echo 'PYTHONPATH='$REMOTE_SCRIPT_FOLDER >> $CALL_PYTHON_FILE
echo $PYTHON_COMMAND >> $CALL_PYTHON_FILE

# Copy the files to mesocentre
ssh $USER@login.ccamu.u-3mrs.fr mkdir -p "$REMOTE_SCRIPT_FOLDER"
scp -r $LOCAL_ROOT/src/* $USER@login.ccamu.u-3mrs.fr:$REMOTE_SCRIPT_FOLDER
scp $CALL_PYTHON_FILE $USER@login.ccamu.u-3mrs.fr:$REMOTE_PROJECT_FOLDER
scp $OAR_FILE $USER@login.ccamu.u-3mrs.fr:$REMOTE_PROJECT_FOLDER

rm $CALL_PYTHON_FILE
rm $OAR_FILE
