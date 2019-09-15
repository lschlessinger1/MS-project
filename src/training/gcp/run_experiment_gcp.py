import argparse
import json
import os
import time
from pathlib import Path

import googleapiclient.discovery


def run_experiment_group(experiments_filename: str,
                         project: str,
                         bucket: str,
                         zone: str,
                         instance_name_prefix: str,
                         compute):
    """Run and save experiment group on the Google Cloud Platform (GCP).

    :param experiments_filename:
    :param project:
    :param bucket:
    :param zone:
    :param instance_name_prefix:
    :param compute:
    :return:
    """
    with open(experiments_filename) as f:
        experiments_config = json.load(f)
    n_experiments = len(experiments_config['experiments'])
    exp_instance_names = []
    create_operations = []

    for i in range(n_experiments):
        instance_name = '-'.join((instance_name_prefix, str(i)))
        experiment_config = experiments_config['experiments'][i]
        experiment_config['experiment_group'] = experiments_config['experiment_group']

        print('Creating instance.')

        operation = create_instance(compute, project, zone, instance_name, bucket, json.dumps(experiment_config))

        create_operations.append(operation)
        exp_instance_names.append(instance_name)

    return exp_instance_names, create_operations


def list_instances(compute, project: str, zone: str):
    result = compute.instances().list(project=project, zone=zone).execute()
    return result['items'] if 'items' in result else None


def create_instance(compute,
                    project: str,
                    zone: str,
                    name: str,
                    bucket: str,
                    exp_args: str):
    image_response = compute.images().get(project="gce-uefi-images", image="cos-stable-76-12239-60-0").execute()
    source_disk_image = image_response['selfLink']
    # Configure the machine
    machine_type = f"projects/{project}/zones/{zone}/machineTypes/c2-standard-4"
    image = 'gcr.io/automated-kernel-search/experiment-runner'

    parent_dir = Path(__file__).resolve().parents[0]
    startup_script = parent_dir / "startup-script.sh"
    startup_script = startup_script.open().read()

    exp_script = "src/training/run_experiment.py"
    subnetwork_region = "-".join(zone.split('-')[:2])

    config = {
        "canIpForward": False,
        "cpuPlatform": "Unknown CPU Platform",
        "deletionProtection": False,
        "description": "",
        "disks": [
            {
                "autoDelete": True,
                "boot": True,
                'initializeParams': {
                    'sourceImage': source_disk_image,
                },
                "deviceName": name,
                "guestOsFeatures": [
                    {
                        "type": "VIRTIO_SCSI_MULTIQUEUE"
                    },
                    {
                        "type": "UEFI_COMPATIBLE"
                    }
                ],
                "index": 0,
                "interface": "SCSI",
                "kind": "compute#attachedDisk",
                "licenses": [
                    "projects/cos-cloud-shielded/global/licenses/shielded-cos",
                    "projects/cos-cloud/global/licenses/cos",
                    "projects/cos-cloud/global/licenses/cos-pcid"
                ],
                "mode": "READ_WRITE",
                "type": "PERSISTENT"
            }
        ],
        "displayDevice": {
            "enableDisplay": False
        },
        "kind": "compute#instance",
        "labelFingerprint": "ba4cB9EpDYo=",
        "labels": {
            "container-vm": "cos-stable-76-12239-60-0"
        },
        "machineType": machine_type,
        "metadata": {
            "fingerprint": "fppbS2H8msc=",
            "items": [
                {
                    'key': 'container_tag',
                    'value': image
                },
                {
                    'key': 'script',
                    'value': exp_script
                },
                {
                    'key': 'args',
                    'value': exp_args
                },
                {
                    'key': 'google_application_credentials',
                    'value': Path(os.environ['GOOGLE_APPLICATION_CREDENTIALS']).name
                },
                {
                    # Startup script is automatically executed by the
                    # instance upon startup.
                    'key': 'startup-script',
                    'value': startup_script
                },
                {
                    'key': 'bucket',
                    'value': bucket
                },
                {
                    "key": "google-logging-enabled",
                    "value": "true"
                },
                {
                    "key": "gce-container-declaration",
                    "value": f"spec:\n  containers:\n    - name: {name}\n      image: "
                             f"{image}\n      "
                             f"stdin: "
                             "false\n      tty: true\n  restartPolicy: Always\n\n"
                             "# This container declaration format "
                             "is not public API and may change without notice. Please\n# use gcloud command-line tool "
                             "or Google Cloud Console to run Containers on Google Compute Engine. "
                }
            ],
            "kind": "compute#metadata"
        },
        "name": name,
        "networkInterfaces": [
            {
                "accessConfigs": [
                    {
                        "kind": "compute#accessConfig",
                        "name": "External NAT",
                        "networkTier": "PREMIUM",
                        "type": "ONE_TO_ONE_NAT"
                    }
                ],
                "fingerprint": "ulZrnCE3FLk=",
                "kind": "compute#networkInterface",
                "name": "nic0",
                "network": f"projects/{project}/global/networks/default",
                "subnetwork": f"projects/{project}/regions/{subnetwork_region}/subnetworks/default"
            }
        ],
        "reservationAffinity": {
            "consumeReservationType": "ANY_RESERVATION"
        },
        "scheduling": {
            "automaticRestart": True,
            "onHostMaintenance": "MIGRATE",
            "preemptible": False
        },
        "selfLink": f"projects/{project}/zones/{zone}/instances/{name}",
        "serviceAccounts": [
            {
                "email": "default",
                "scopes": [
                    "https://www.googleapis.com/auth/devstorage.read_write",
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring.write",
                    "https://www.googleapis.com/auth/servicecontrol",
                    "https://www.googleapis.com/auth/service.management.readonly",
                    "https://www.googleapis.com/auth/trace.append"
                ]
            }
        ],
        "shieldedInstanceConfig": {
            "enableIntegrityMonitoring": True,
            "enableSecureBoot": False,
            "enableVtpm": True
        },
        "tags": {
            "fingerprint": "42WmSpB8rSM="
        },
    }

    return compute.instances().insert(
        project=project,
        zone=zone,
        body=config).execute()


def delete_instance(compute, project: str, zone: str, name: str):
    return compute.instances().delete(
        project=project,
        zone=zone,
        instance=name).execute()


def wait_for_operation(compute, project: str, zone: str, operation):
    print('Waiting for operation to finish...')
    while True:
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation).execute()

        if result['status'] == 'DONE':
            print("done.")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'project_id',
        help='Your Google Cloud project ID.'
    )
    parser.add_argument(
        'bucket_name',
        help='Your Google Cloud Storage bucket name.'
    )
    parser.add_argument(
        '--zone',
        default='us-central1-b',
        help='Compute Engine zone to deploy to.'
    )
    parser.add_argument(
        '--name',
        default='exp-runner',
        help='New instance name.'
    )

    parser.add_argument(
        "--n_repeats",
        default=1,
        type=int,
        dest='n_repeats',
        help="The experiment will be repeated `n_repeats` times"
    )

    parser.add_argument("experiments_filename", type=str, help="Filename of JSON file of experiments to run.")

    args = parser.parse_args()

    compute = googleapiclient.discovery.build('compute', 'v1')
    wait: bool = True

    all_instance_names = []
    all_create_operations = []
    for i in range(args.n_repeats):
        name = '-'.join((args.name, str(i)))
        instance_names, create_operations = run_experiment_group(args.experiments_filename, args.project_id,
                                                                 args.bucket_name, args.zone, name, compute)
        for instance_name in instance_names:
            all_instance_names.append(instance_name)
        for create_operation in create_operations:
            all_create_operations.append(create_operation)

    for operation in all_create_operations:
        wait_for_operation(compute, args.project_id, args.zone, operation['name'])

    print(all_instance_names)

    instances = list_instances(compute, args.project_id, args.zone)

    print('Instances in project %s and zone %s:' % (args.project_id, args.zone))
    for instance in instances:
        print(' - ' + instance['name'])

    print("""
    Instances created.
    It will take several minutes for the instances to complete the work.
    View instances: https://console.cloud.google.com/compute/instances?project={}
    View results: https://console.cloud.google.com/storage/browser/{}/?project={}
    Once the experiment has finished, press enter to delete all instances.
    """.format(args.project_id, args.bucket_name, args.project_id))

    if wait:
        input()

    print('Deleting instances.')

    all_delete_operations = []
    for instance_name in all_instance_names:
        operation = delete_instance(compute, args.project_id, args.zone, instance_name)
        all_delete_operations.append(operation)

    for operation in all_delete_operations:
        wait_for_operation(compute, args.project_id, args.zone, operation['name'])


if __name__ == '__main__':
    main()
