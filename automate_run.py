import os
import argparse
import fsspec
from fsspec import AbstractFileSystem
import time
import subprocess
import json
import logging
import datetime

logger = None

def create_tpu_vm(tpu_name, tpu_type, zone):
    tpu_created = False

    while not tpu_created:
        tpu_create_issue = subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus", 
                "tpu-vm",
                "create",
                f"{tpu_name}",
                f"--zone={zone}",
                f"--accelerator-type={tpu_type}",
                f"--version=tpu-ubuntu2204-base",
                f"--preemptible"
            ], 
            capture_output=True, 
            text=True
        )

        if tpu_create_issue.stdout != '':
            tpu_created = True
            logger.info(f'{datetime.datetime.now()} - TPU Create Succesfully')
            return tpu_create_issue.stdout
        else: 
            logger.error(f'{datetime.datetime.now()}-TPU CREATION ERROR: {tpu_create_issue.stderr}')
            time.sleep(120)
            continue

def describe_tpu_vm(tpu_name, zone):
    
    return subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "describe",
                tpu_name,
                f"--zone={zone}",
                "--quiet",
            ],
            capture_output=True, 
            text=True
        )
    

def delete_tpu_vm(tpu_name, zone):
    return subprocess.run(
        [
            "gcloud", 
            "compute", 
            "tpus", 
            "tpu-vm", 
            "delete",
            f"{tpu_name}", 
            f"--zone={zone}"
        ], 
        capture_output=True, 
        text=True
    )

def run_inference(tpu_name, tpu_type, zone, config, trainer_id, load_checkpoint_path):
    return subprocess.run(
        [
            "sudo", 
            "python3", 
            "infra/launch.py", 
            "--tpu_name",
            f"{tpu_name}", 
            "--tpu_type", 
            f"{tpu_type}", 
            "--zone",
            f"{zone}", 
            "--", 
            "python3", 
            "src/levanter/main/train_lm.py", 
            "--config_path", 
            f"{config}", 
            "--trainer.load_checkpoint_path", 
            f"{load_checkpoint_path}", 
            "--trainer.wandb.resume", 
            "True", 
            "--trainer.id", 
            f"{trainer_id}"
        ], 
        capture_output=True, 
        text=True
    )

def parse_tpu_status(tpu_describe_query_output):
    # TODO: Add enum for tpu status

    if tpu_describe_query_output.stdout != '':
        if 'DELETING' in tpu_describe_query_output.stdout:
            return 'DELETING'
        elif 'READY' in tpu_describe_query_output.stdout:
            return 'READY'
        elif 'PREEMPTING' in tpu_describe_query_output.stdout:
            return 'PREEMPTING'
        elif 'PREEMPTED' in tpu_describe_query_output.stdout:
            return 'PREEMPTED'
        elif 'DELETED' in tpu_describe_query_output.stdout:
            return 'DELETED'
        elif 'CREATING' in tpu_describe_query_output.stdout:
            return 'CREATING'
        elif 'READY' in tpu_describe_query_output.stdout:
            return 'READY'
    # TODO: Verify the below logic and remove boiler plate
    elif tpu_describe_query_output.stderr != '':
        if 'DELETING' in tpu_describe_query_output.stdout:
            return 'DELETING'
        elif 'READY' in tpu_describe_query_output.stdout:
            return 'READY'
        elif 'PREEMPTING' in tpu_describe_query_output.stdout:
            return 'PREEMPTING'
        elif 'PREEMPTED' in tpu_describe_query_output.stdout:
            return 'PREEMPTED'
        elif 'DELETED' in tpu_describe_query_output.stdout:
            return 'DELETED'
        elif 'CREATING' in tpu_describe_query_output.stdout:
            return 'CREATING'
        elif 'READY' in tpu_describe_query_output.stdout:
            return 'READY'
        else:
            return 'NOTFOUND'
    
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)

    args = parser.parse_args()

    config_file = args.config_file

    with open(config_file, 'r') as f:
        inference_args = json.load(f)

    print(inference_args)
    tpu_name = inference_args['tpu_name']
    tpu_type = inference_args['tpu_type']
    zone = inference_args['zone']
    config = inference_args['config']
    checkpoint_path = inference_args['checkpoint_path']
    base_bucket_path = inference_args['base_bucket_path']

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{tpu_name}_inference.log", encoding='utf-8', level=logging.DEBUG)

    fs: AbstractFileSystem = fsspec.core.url_to_fs(checkpoint_path)[0]

    # print(checkpoint_path)
    checkpoints = fs.ls(checkpoint_path)

    if len(checkpoints) == 0:
        print("No checkpoint found")
        exit(0)
    elif len(checkpoints) > 1:
        print("multiple checkpoints found")
        exit(0)

    trainer_id = checkpoints[0].split('/')[-1]

    checkpoints_by_step = fs.ls(f'{checkpoint_path}/{trainer_id}')
    checkpoint_steps = [checkpoint.split('/')[-1] for checkpoint in checkpoints_by_step]
    steps = [int(step.split('-')[-1]) for step in checkpoint_steps]
    steps.sort(reverse=True)

    latest_step = steps[-1]
    load_checkpoint_path = f'{checkpoint_path}/{trainer_id}/step-{latest_step}'

    logger.info(f'load_checkpoint_path: {load_checkpoint_path}')

    # TODO: Add inference running status enum
    inference_running = False

    logger.info(f'Started Training at {datetime.datetime.now()}')
    logger.info(f'TIME: {datetime.datetime.now()}')

    while True:
        tpu_status = parse_tpu_status(describe_tpu_vm(tpu_name, zone))

        logger.info(f'TIME: {datetime.datetime.now()}')
        logger.info(f'TPU STATUS: {tpu_status}')
        
        if tpu_status == 'READY':
            if inference_running:
                logger.info(f'{datetime.datetime.now()}- TPU RUNNING AS WELL AS YOUR INFERENCE...')
                time.sleep(120)
                continue
            else:
                logger.info(f'{datetime.datetime.now()} - Resuming Inference')
                inference_ssh = run_inference(tpu_name, tpu_type, zone, config, trainer_id, load_checkpoint_path)
                if inference_ssh.returncode == 0:
                    logger.info(f'{datetime.datetime.now()} - Inference Started Successfully')
                    inference_running = True
                else:
                    logger.error(f'{datetime.datetime.now()} - Error in inference: {inference_ssh.stderr}')                

        elif tpu_status == 'DELETING' or tpu_status == 'PREEMPTING':
            inference_running = False
            logger.warning(f'{datetime.datetime.now()} - TPU STATUS: {tpu_status}')
            time.sleep(120)
        
        elif tpu_status == 'PREEMPTED':
            inference_running = False
            delete_tpu_vm(tpu_name, zone)
            logger.info(f'{datetime.datetime.now()} - DELETING TPU')
            time.sleep(30)

        elif tpu_status == 'NOTFOUND' or tpu_status == "DELETED":
            create_tpu_vm(tpu_name, tpu_type, zone)
            time.sleep(30)
            logger.info(f'{datetime.datetime.now()} - Resuming Inference')
            inference_ssh = run_inference(tpu_name, tpu_type, zone, config, trainer_id, load_checkpoint_path)
            if inference_ssh.returncode == 0:
                logger.info(f'{datetime.datetime.now()} - Inference Started Successfully')
                inference_running = True
            else:
                logger.error(f'{datetime.datetime.now()} - Error in inference: {inference_ssh.stderr}')
            time.sleep(120)
        
