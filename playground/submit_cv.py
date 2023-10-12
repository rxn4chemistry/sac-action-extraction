from pyjbsub import PyJbsub
from rxn.onmt_utils.train_command import OnmtTrainCommand

base_dir = "/u/ava/data/ace_ethz/20230313_retraining_with_cv"

submitter = PyJbsub(
    conda_environment="rxn-onmt-utils",
    jbsub_args="-queue x86_6h -cores 1+1 -mem 16g  -require 'v100'",
    job_output_dir=f"{base_dir}/job_logs",
)

# Best models were ace10_0.6_4000_step_7000.pt, combined05_0.6_8000_2_1_step_11000.pt

# with multitask - reproduce combined05_0.6_8000_2_1_step_11000.pt
warmup = 8000
learning_rate = 0.6
ace_weight, org_weight = 2, 1
for cv in ["1", "2", "3", "4", "5"]:
    train_steps = 30000
    save_checkpoint = 500

    train_cmd = OnmtTrainCommand.finetune(
        batch_size=4096,
        data=f"{base_dir}/preprocessed/combined_cv_{cv}/data",
        dropout=0.1,
        learning_rate=learning_rate,
        save_model=f"{base_dir}/models/combined_cv_{cv}/model",
        seed=42,
        train_from=f"{base_dir}/original_models/pretrained.pt",
        train_steps=train_steps,
        warmup_steps=warmup,
        no_gpu=False,
        data_weights=(ace_weight, org_weight),
        report_every=500,
        save_checkpoint_steps=save_checkpoint,
    )
    train_cmd._kwargs.update({"keep_checkpoint": 100})
    command_str = " ".join(train_cmd.cmd())

    submitter.submit(command_str, dry_run=False)

# without multitask - reproduce ace10_0.6_4000_step_7000.pt
warmup = 4000
learning_rate = 0.6
for cv in ["1", "2", "3", "4", "5"]:
    train_steps = 30000
    save_checkpoint = 500

    train_cmd = OnmtTrainCommand.finetune(
        batch_size=4096,
        data=f"{base_dir}/preprocessed/ace_cv_{cv}/data",
        dropout=0.1,
        learning_rate=learning_rate,
        save_model=f"{base_dir}/models/ace_cv_{cv}/model",
        seed=42,
        train_from=f"{base_dir}/original_models/pretrained.pt",
        train_steps=train_steps,
        warmup_steps=warmup,
        no_gpu=False,
        data_weights=tuple(),
        report_every=500,
        save_checkpoint_steps=save_checkpoint,
    )
    train_cmd._kwargs.update({"keep_checkpoint": 100})
    command_str = " ".join(train_cmd.cmd())

    submitter.submit(command_str, dry_run=False)
