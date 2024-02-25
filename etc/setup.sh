#!/bin/bash

echo Setting up environment with OS type: $OSTYPE

if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    export CONFIG=`dirname $SCRIPT`
    export WORKINGDIR=`dirname $CONFIG`

elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
    export WORKINGDIR=`dirname $SCRIPT`
    export CONFIG=$WORKINGDIR/etc
fi

# Creates redundancy in python path when sourced out of integreated shell but makes sure works for external shell aswell
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTHONPATH=$PYTHONPATH:$WORKINGDIR/model/slate
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    export PYTHONPATH=$PYTHONPATH; $WORKINGDIR\\model\\slate
fi

# Creates redundancy in python path when sourced out of integreated shell but makes sure works for external shell aswell
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTHONPATH=$PYTHONPATH:$WORKINGDIR/model/vanilla_slot_attention
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    export PYTHONPATH=$PYTHONPATH; $WORKINGDIR\\model\\vanilla_slot_attention
fi

# Creates redundancy in python path when sourced out of integreated shell but makes sure works for external shell aswell
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTHONPATH=$PYTHONPATH:$WORKINGDIR
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    export PYTHONPATH=$PYTHONPATH; $WORKINGDIR
fi

