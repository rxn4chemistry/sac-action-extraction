from pyjbsub import PyJbsub
from rxn.onmt_utils.train_command import OnmtTrainCommand

base_dir = "/u/ava/data/ace_ethz/20230310_extended_retraining"

submitter = PyJbsub(
    conda_environment="rxn-onmt-utils",
    jbsub_args="-queue x86_6h -cores 1+1 -mem 16g  -require 'v100'",
    job_output_dir=f"{base_dir}/job_logs",
)

# with multitask
for warmup in [2000, 4000, 8000]:
    for dataset in ["combined00", "combined05", "combined10"]:
        for learning_rate in [0.60, 0.20, 0.06]:
            for ace_weight, org_weight in [(1, 1), (2, 1), (1, 2)]:
                train_steps = 30000
                save_checkpoint = 500

                train_cmd = OnmtTrainCommand.finetune(
                    batch_size=4096,
                    data=f"{base_dir}/preprocessed/{dataset}/data",
                    dropout=0.1,
                    learning_rate=learning_rate,
                    save_model=f"{base_dir}/models/{dataset}_{learning_rate}_{warmup}_{ace_weight}_{org_weight}/model",
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

# No multi-task
for warmup in [2000, 4000, 8000]:
    for dataset in ["ace00", "ace05", "ace10"]:
        for learning_rate in [0.60, 0.20, 0.06]:
            train_steps = 30000
            save_checkpoint = 500

            train_cmd = OnmtTrainCommand.finetune(
                batch_size=4096,
                data=f"{base_dir}/preprocessed/{dataset}/data",
                dropout=0.1,
                learning_rate=learning_rate,
                save_model=f"{base_dir}/models/{dataset}_{learning_rate}_{warmup}/model",
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
