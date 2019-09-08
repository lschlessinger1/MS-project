#!/bin/bash

### Metadata specification
# All this metadata is pulled from the Compute Engine instance metadata server.
SCRIPT=$(curl http://metadata/computeMetadata/v1/instance/attributes/script -H "Metadata-Flavor: Google")
ARGS=$(curl http://metadata/computeMetadata/v1/instance/attributes/args -H "Metadata-Flavor: Google")
CONTAINER_TAG=$(curl http://metadata/computeMetadata/v1/instance/attributes/container_tag -H "Metadata-Flavor: Google")
GOOGLE_APPLICATION_CREDENTIALS=$(curl http://metadata/computeMetadata/v1/instance/attributes/google_application_credentials -H "Metadata-Flavor: Google")

### Run training script
docker run --log-driver=gcplogs --log-opt gcp-log-cmd=true --rm \
 -e "GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS" \
 "$CONTAINER_TAG" python3 "$SCRIPT" "$ARGS"

### Shutdown GCE instance
sudo shutdown -h now